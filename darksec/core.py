"""
Dark Sectioning core (GPU / CuPy).

The algorithm is the one in the legacy `dark_sectioning_gpu.py`
(`dark_sectioning_gpu_true_optimized`), i.e. a Python port of the MATLAB
Dark Sectioning code (Cao et al.): per z-slice

    1. split the image into Hi / Lo (Gaussian high/low-pass, cut-off set by
       the optical resolution) and EL (an even lower-pass copy),
    2. dehaze Lo with a dark-channel prior whose "atmosphere" A(x) follows EL,
    3. result = Lo_dehazed / hl + Hi,

repeated `maxtime` times with decreasing EL smoothness (deg) and depth (dep).

What is different from the legacy code
--------------------------------------
* Slices are independent, so the z loop is the outer loop and one slice at a
  time lives on the GPU. Memory is O(one padded slice), not O(stack), so
  74-slice 2304x2304 stacks no longer exhaust a 12 GB card.
* The image is NOT min-max normalised per stack. The caller supplies a
  linear map (lo, hi) -> [0, 255] that is shared by every image of a channel
  (see calibrate.py). `thres` and the atmosphere therefore mean the same
  absolute intensity in every image.
* The "atmosphere" statistics (A_min, A_max, EL_max) can be estimated
  per slice (legacy), per stack (median over slices), or fixed from a
  calibration (global). See `AtmosphereMode`.
* No clipping to [0, 255] inside the iterations (values above `hi` are
  legitimate bright signal and are kept); only the closing uint16 conversion
  clips at 0 and 65535.
* Filters, PSF block size and FFT plans are prepared once per image size.
* Two places where the legacy GPU port deviated from the published MATLAB
  (`MATLAB_Code/Dark.m` + helpfunctions, github.com/Cao-ruijie/Dark-sectioning)
  were corrected: `dehaze_fast2` subtracts min(EL) before normalising EL,
  and `confirm_block` scans the PSF profile from MATLAB index floor(Nx/2)
  (one row before / one column beside the peak). `DSParams.exact_upstream`
  additionally reproduces the integer-rounded cut-off and the exact
  floor(N/15)+1 padding of the MATLAB code for validation.

Everything is float64: the guided filter subtracts mean(g)^2 from mean(g^2)
and would lose precision in float32 on 2700x2700 slices.
"""
from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning, module="cupyx.jit._interface")

import cupy as cp  # noqa: E402
import scipy.fft  # noqa: E402
from cupyx.scipy.ndimage import (  # noqa: E402
    gaussian_filter as cp_gaussian_filter,
    minimum_filter as cp_minimum_filter,
)
from cupyx.scipy.special import j1 as cp_j1  # noqa: E402

DEVICE_NAME = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())["name"].decode()
# bump when a change alters numerical results; calibrations made by another version are redone
ALGORITHM_VERSION = "1.1"   # 1.0: legacy port; 1.1: EL min-subtraction + MATLAB confirm_block indexing (upstream-faithful)

# maxtime, deg (EL smoothness divisor), dep (atmosphere depth), hl
# 'mild'     = MATLAB background=0 : 1 iteration
# 'moderate' = MATLAB background=1 : 2 iterations
# 'severe'   = Python extension    : 3 iterations
BACKGROUND_PRESETS: Dict[str, dict] = {
    "severe":   dict(deg=[6.0, 3.0, 1.2], dep=[3.0, 3.0, 2.0], hl=[1.0, 1.0, 1.0]),
    "moderate": dict(deg=[6.0, 3.0],      dep=[3.0, 3.0],      hl=[1.0, 1.0]),
    "mild":     dict(deg=[6.0],           dep=[3.0],           hl=[1.0]),
}

# Continuous strength: the dark-channel prior weight omega (published value 0.95). omega = 0 leaves every
# slice unchanged (t = 1 -> J = I), omega -> 1 removes the whole estimated atmosphere. Exposed to the user as
# `processing.strength` = 100 * omega. The presets (iteration schedules) stay discrete; a continuous
# interpolation of their EL smoothness `deg` was tried and rejected: after the first iteration the atmosphere
# is spatially almost flat (a_min ~ a_max), so `deg` of later iterations barely changes the output.
OMEGA_DEFAULT = 0.95
STRENGTH_DEFAULT = 100.0 * OMEGA_DEFAULT


def omega_from_strength(strength) -> float:
    s = float(strength)
    if not (0.0 <= s <= 100.0):
        raise ValueError(f"strength must be within 0-100 (percent of the dark-channel prior weight), got {strength}")
    return s / 100.0


def resolve_preset(background) -> Tuple[str, dict]:
    """(label, schedule) for a preset name or an explicit dict(deg, dep[, hl])."""
    if isinstance(background, str):
        if background in BACKGROUND_PRESETS:
            return background, BACKGROUND_PRESETS[background]
        raise ValueError(f"background must be one of {list(BACKGROUND_PRESETS)} or a dict(deg, dep[, hl]); got {background!r}. "
                         f"The continuous knob is processing.strength (0-100), not background.")
    if isinstance(background, dict):
        deg = [float(v) for v in background["deg"]]; dep = [float(v) for v in background["dep"]]
        hl = [float(v) for v in background.get("hl", [1.0] * len(deg))]
        if not (len(deg) == len(dep) == len(hl) >= 1):
            raise ValueError("custom background: deg, dep and hl must have the same length >= 1")
        return "custom", dict(deg=deg, dep=dep, hl=hl)
    raise ValueError(f"unsupported background specification {background!r}")

T_MIN = 0.1           # transmission floor
GF_RADIUS = 15        # guided-filter radius
GF_EPS = 0.001        # guided-filter regulariser
DIVIDE = 0.5          # Hi/Lo split relative to sigmaLP
ATM_TOP_FRACTION = 0.01  # brightest 1 % of the dark channel define the atmosphere


# ----------------------------------------------------------------------------
# FFT backend. cuFFT (CuPy 13.6 / CUDA 11.8 build, RTX 4070 SUPER) returns
# garbage for 2-D complex128 transforms whose side has a large prime factor
# (2612 = 4*653, 2611, 2609, 3001, 4001, 4099, ...): the round trip
# ifft2(fft2(x)) comes back as zeros or 1e5 off. Production padding uses
# next_fast_len (smooth sizes) and never hits this, but exact_upstream mode pads
# 2304 -> 2612 -- which is exactly what the legacy GPU port did, and why it
# produced all-zero output on 2304x2304 fields. Every DarkSectioner self-tests
# its padded size and falls back to scipy.fft on the CPU when cuFFT is wrong.
# ----------------------------------------------------------------------------
def largest_prime_factor(n: int) -> int:
    p, d = 1, 2
    while d * d <= n:
        while n % d == 0:
            p, n = d, n // d
        d += 1
    return max(p, n) if n > 1 else p


def is_smooth(n: int, max_prime: int = 13) -> bool:
    """True if every prime factor of n is <= max_prime (the sizes scipy.fft.next_fast_len returns)."""
    return largest_prime_factor(n) <= max_prime


def cufft_ok(shape: Tuple[int, int], tol: float = 1e-8) -> bool:
    x = cp.random.default_rng(0).random(shape, dtype=cp.float64)
    err = float(cp.abs(cp.fft.ifft2(cp.fft.fft2(x)).real - x).max())
    return math.isfinite(err) and err < tol


_CUFFT_TRUSTED = None


def cufft_version() -> int:
    from cupy.cuda import cufft
    return int(cufft.getVersion())


def cufft_trusted() -> bool:
    """True if the loaded cuFFT can be used for every 2-D size.

    cuFFT 10.3 (CUDA 11.1) returns wrong, non-deterministic complex128 results for
    sizes with a large prime factor (Bluestein path). Trust requires a library
    newer than that AND three consecutive correct round trips on two such sizes;
    the result is cached per process.
    """
    global _CUFFT_TRUSTED
    if _CUFFT_TRUSTED is None:
        ok = cufft_version() >= 10900
        if ok:
            for n in (2612, 3001):
                ok = ok and all(cufft_ok((n, n)) for _ in range(3))
        _CUFFT_TRUSTED = bool(ok)
    return _CUFFT_TRUSTED


class FFT2:
    """fft2 / ifft2 on CuPy arrays, on the GPU or (fallback) via scipy on the CPU.

    Every padded size is round-trip tested. Sizes with a prime factor > 13 (cuFFT's
    Bluestein path) are additionally accepted only when the loaded cuFFT is trusted
    (`cufft_trusted`): with cuFFT 10.3 (CUDA 11.1) that path returned wrong results
    NON-deterministically -- the same 2612x2612 transform could be right in one call
    and garbage in the next -- so for that library a self-test alone is not a
    sufficient guard and such sizes go to scipy.fft on the CPU. next_fast_len
    padding (production) only produces smooth sizes and is unaffected either way.
    """

    def __init__(self, shape: Tuple[int, int]):
        self.shape = tuple(shape)
        smooth = all(is_smooth(int(n)) for n in self.shape)
        allowed = smooth or cufft_trusted()
        self.backend = "gpu" if (allowed and cufft_ok(self.shape)) else "cpu"
        if self.backend == "cpu":
            why = (f"has a prime factor > 13 (cuFFT Bluestein path) and the loaded cuFFT {cufft_version()} is not trusted for it"
                   if not allowed else "fails the cuFFT round-trip test")
            warnings.warn(f"FFT size {self.shape} {why}; using scipy.fft on the CPU for this size (slower). "
                          f"exact_upstream=False uses fast-length padding and stays on the GPU.", RuntimeWarning)

    def fft2(self, x: cp.ndarray) -> cp.ndarray:
        if self.backend == "gpu":
            return cp.fft.fft2(x)
        return cp.asarray(scipy.fft.fft2(cp.asnumpy(x), workers=-1))

    def ifft2(self, X: cp.ndarray) -> cp.ndarray:
        if self.backend == "gpu":
            return cp.fft.ifft2(X)
        return cp.asarray(scipy.fft.ifft2(cp.asnumpy(X), workers=-1))


# ----------------------------------------------------------------------------
# optics -> filters
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Optics:
    NA: float
    emission_nm: float
    pixel_nm: float
    factor: float = 2.0
    z_step_um: Optional[float] = None       # axial sampling of the stack (None for single planes)
    n_immersion: Optional[float] = None     # refractive index of the immersion medium (None -> guessed from NA)

    @property
    def resolution_px(self) -> float:
        """0.5 * lambda / NA / factor, in pixels (legacy definition)."""
        return 0.5 * self.emission_nm / self.NA / self.factor / self.pixel_nm

    # -- defocus geometry (used by cutoff='optical' and model='neighbour') --------------------------------
    @property
    def n(self) -> float:
        if self.n_immersion:
            return float(self.n_immersion)
        return 1.515 if self.NA > 1.33 else (1.33 if self.NA > 1.0 else 1.0)

    @property
    def dof_um(self) -> float:
        """Axial resolution / depth of field n * lambda / NA^2 (um)."""
        return self.n * self.emission_nm * 1e-3 / self.NA ** 2

    def defocus_radius_um(self, dz_um: float) -> float:
        """Radius of the geometric defocus disk at |dz|: dz * tan(asin(NA / n))."""
        return float(dz_um) * math.tan(math.asin(min(self.NA / self.n, 0.999)))

    @property
    def dz_eff_um(self) -> float:
        """Nearest plane that is out of focus: max(z step, depth of field)."""
        return max(float(self.z_step_um or 0.0), self.dof_um)

    @property
    def optical_cutoff_cycles_per_um(self) -> float:
        """First zero of the defocus-disk MTF of the nearest out-of-focus plane, 0.61 / r(dz_eff):
        above this frequency out-of-focus planes contribute (to first order) nothing."""
        return 0.61 / self.defocus_radius_um(self.dz_eff_um)

    @property
    def optical_cutoff_cycles_per_px(self) -> float:
        return self.optical_cutoff_cycles_per_um * self.pixel_nm * 1e-3

    @property
    def optical_window_px(self) -> int:
        """Dark-channel (minimum filter) window for cutoff='optical': the defocus radius of the nearest
        out-of-focus plane in pixels (odd, >= 3)."""
        r = int(round(self.defocus_radius_um(self.dz_eff_um) / (self.pixel_nm * 1e-3)))
        return max(3, r | 1)


def immersion_index(objective: str, NA: float) -> float:
    """Refractive index of the immersion medium from the objective name (Leica strings), else from the NA."""
    o = (objective or "").upper()
    if "WATER" in o or "W " in o or "IMM CORR" in o:
        return 1.33
    if "GLYC" in o:
        return 1.45
    if "OIL" in o:
        return 1.515
    if "DRY" in o or "AIR" in o:
        return 1.0
    return 1.515 if NA > 1.33 else (1.33 if NA > 1.0 else 1.0)


def _lpgauss(H: int, W: int, sigma: float) -> cp.ndarray:
    """Gaussian low-pass in Fourier space, DC at [0, 0] (legacy `lpgauss`)."""
    kcx, kcy = sigma, (H / W) * sigma
    x = cp.arange(-math.floor(W / 2), math.floor((W - 1) / 2) + 1, dtype=cp.float64)
    y = cp.arange(-math.floor(H / 2), math.floor((H - 1) / 2) + 1, dtype=cp.float64)
    X, Y = cp.meshgrid(x, y)
    return cp.fft.ifftshift(cp.exp(-(X ** 2 / kcx ** 2 + Y ** 2 / kcy ** 2)))


def _psf(optics: Optics, w: int) -> cp.ndarray:
    """Airy PSF on a w x w grid, centred (legacy `PSF_Generator`)."""
    X, Y = cp.meshgrid(cp.linspace(0, w - 1, w), cp.linspace(0, w - 1, w))
    scale = 2 * math.pi * optics.NA / optics.emission_nm * optics.pixel_nm * optics.factor
    R = cp.sqrt(cp.minimum(X, cp.abs(X - w)) ** 2 + cp.minimum(Y, cp.abs(Y - w)) ** 2)
    sr = scale * R + np.finfo(float).eps
    psf = cp.abs(2 * cp_j1(sr) / sr) ** 2
    return cp.fft.fftshift(psf / cp.sum(psf))


def block_size_from_psf(optics: Optics, lp: cp.ndarray, fft: Optional["FFT2"] = None) -> int:
    """Dark-channel window from the PSF (upstream `confirm_block.m`).

    MATLAB:  for count_x = floor(Nx/2):Nx
                 if PSF_Lo(count_x, floor(Nx/2)) < 0.01, break, end
             end;  block_size = count_x - floor(Nx/2)
    i.e. 1-based row/col floor(Nx/2) = 0-based Nx//2 - 1: the scan starts one
    row before the peak, one column beside it, and block_size is the offset
    from that start. (The legacy Python port scanned from the peak itself.)
    """
    Nx = lp.shape[0]
    fft = fft or FFT2((Nx, Nx))
    psf = _psf(optics, Nx)
    spectrum = cp.fft.fftshift(fft.fft2(psf)) * cp.fft.fftshift(lp[:Nx, :Nx] if lp.shape[1] >= Nx else lp)
    psf_lo = cp.abs(fft.ifft2(cp.fft.ifftshift(spectrum)))
    # the peak must be at (Nx//2, Nx//2); for even sizes it always is
    peak = cp.unravel_index(cp.argmax(psf_lo), psf_lo.shape)
    if int(peak[0]) != Nx // 2 or int(peak[1]) != Nx // 2:
        psf_lo = cp.roll(psf_lo, (Nx // 2 - int(peak[0]), Nx // 2 - int(peak[1])), axis=(0, 1))
    psf_lo = psf_lo / cp.max(psf_lo)
    start = Nx // 2 - 1
    line = psf_lo[start:, start]
    below = line < 0.01
    bs = int(cp.argmax(below)) if bool(cp.any(below)) else int(Nx - Nx // 2)
    return max(1, bs)


@dataclass
class SliceFilters:
    """Per-iteration Fourier filters for one padded slice size."""
    lp: cp.ndarray      # Hi/Lo split low-pass (DC at corner)
    hp: cp.ndarray
    elp: cp.ndarray     # EL low-pass
    block_size: int


class FilterBank:
    """Filters + block sizes for a given (padded shape, optics, preset). Built once."""

    def __init__(self, shape_pad: Tuple[int, int], optics: Optics, degs: Sequence[float],
                 kc_round: str = "exact", fft: Optional["FFT2"] = None, cutoff: str = "legacy"):
        H, W = shape_pad
        self.cutoff_mode = cutoff
        if cutoff == "legacy":
            # reference implementation: res in px, k_m = W / res_px, kc = floor(0.2 k_m) (or the
            # nearest integer), sigmaLP = 2 kc / 2.355. By default the rounding is dropped so
            # that the cut-off in cycles/px is exactly independent of the (padded) image
            # size (changes results by < 0.3 %).
            k_m = W / optics.resolution_px
            kc = {"exact": k_m * 0.2, "floor": math.floor(k_m * 0.2), "nearest": round(k_m * 0.2)}[kc_round]
            sigma_lp = kc * 2 / 2.355
        elif cutoff == "optical":
            # Hi/Lo split at the first zero of the defocus MTF of the nearest out-of-focus plane: frequencies above
            # it cannot come from other planes and pass untouched; the haze estimate works on the band below.
            sigma_lp = optics.optical_cutoff_cycles_per_px * W / (2 * DIVIDE)
        else:
            raise ValueError(f"cutoff must be 'legacy' or 'optical', got {cutoff!r}")
        self.sigma_lp = sigma_lp
        self.cutoff_cycles_per_px = sigma_lp * 2 * DIVIDE / W
        self.cutoff_cycles_per_um = self.cutoff_cycles_per_px / (optics.pixel_nm * 1e-3)
        lp = _lpgauss(H, W, sigma_lp * 2 * DIVIDE)
        hp = 1 - lp
        # block size needs a square grid; use the smaller side (result is a
        # physical size, independent of the grid as long as it is large)
        n_sq = min(H, W)
        lp_sq = lp if H == W else _lpgauss(n_sq, n_sq, sigma_lp * 2 * DIVIDE * n_sq / W)
        bs = block_size_from_psf(optics, lp_sq, fft if (fft is not None and H == W) else None) if cutoff == "legacy" else optics.optical_window_px
        self.block_size = bs
        self.per_iter: List[SliceFilters] = []
        for deg in degs:
            elp = _lpgauss(H, W, sigma_lp / deg)
            self.per_iter.append(SliceFilters(lp=lp, hp=hp, elp=elp, block_size=bs))
        self.el_cycles_per_px = [sigma_lp / d / W for d in degs]


# ----------------------------------------------------------------------------
# dehazing pieces
# ----------------------------------------------------------------------------
def _box_sum(image: cp.ndarray, r: int) -> cp.ndarray:
    """Legacy cumulative-sum box filter (window 2r+1, truncated at the borders)."""
    h, w = image.shape
    out = cp.empty_like(image)
    c = cp.cumsum(image, axis=0)
    out[:r + 1, :] = c[r:2 * r + 1, :]
    out[r + 1:h - r, :] = c[2 * r + 1:h, :] - c[:h - 2 * r - 1, :]
    out[h - r:h, :] = c[h - 1:h, :] - c[h - 2 * r - 1:h - r - 1, :]
    c = cp.cumsum(out, axis=1)
    out2 = cp.empty_like(image)
    out2[:, :r + 1] = c[:, r:2 * r + 1]
    out2[:, r + 1:w - r] = c[:, 2 * r + 1:w] - c[:, :w - 2 * r - 1]
    out2[:, w - r:w] = c[:, w - 1:w] - c[:, w - 2 * r - 1:w - r - 1]
    return out2


class GuidedFilter:
    def __init__(self, shape: Tuple[int, int], r: int = GF_RADIUS, eps: float = GF_EPS):
        self.r, self.eps = r, eps
        self.N = _box_sum(cp.ones(shape, dtype=cp.float64), r)

    def __call__(self, guide: cp.ndarray, target: cp.ndarray) -> cp.ndarray:
        r, N = self.r, self.N
        mean_g = _box_sum(guide, r) / N
        mean_t = _box_sum(target, r) / N
        var_g = _box_sum(guide * guide, r) / N - mean_g * mean_g
        cov_gt = _box_sum(guide * target, r) / N - mean_g * mean_t
        a = cov_gt / (var_g + self.eps)
        b = mean_t - a * mean_g
        return (_box_sum(a, r) / N) * guide + _box_sum(b, r) / N


def dark_channel(image: cp.ndarray, win: int) -> cp.ndarray:
    return cp_minimum_filter(image, size=win, mode="constant", cval=cp.inf)


def atmosphere(image: cp.ndarray, dc: cp.ndarray) -> float:
    """Mean of `image` over the 1 % of pixels with the highest dark channel."""
    n = max(1, int(math.floor(dc.size * ATM_TOP_FRACTION)))
    flat = dc.ravel()
    # k-th largest as threshold; ties are negligible on real data
    kth = cp.partition(flat, flat.size - n)[flat.size - n]
    sel = flat >= kth
    return float(cp.mean(image.ravel()[sel]))


@dataclass
class AtmosphereStats:
    a_min: float          # atmosphere of the masked (background) image
    a_max: float          # atmosphere of the whole image
    el_max: float         # max of EL
    el_min: float = 0.0   # min of EL (upstream: EL = EL - min(EL) before normalising)

    def alpha_beta(self) -> Tuple[float, float]:
        """A(x) = dep * (alpha * EL(x) + beta)."""
        rng = self.el_max - self.el_min
        alpha = (self.a_max - self.a_min) / rng if rng > 0 else 0.0
        return alpha, self.a_min - alpha * self.el_min


def estimate_atmosphere(lo_img: cp.ndarray, el: cp.ndarray, thres: float, win: int) -> AtmosphereStats:
    mask = (lo_img < thres).astype(cp.float64)
    masked = lo_img * mask
    a_min = atmosphere(masked, dark_channel(masked, win))
    a_max = atmosphere(lo_img, dark_channel(lo_img, win))
    return AtmosphereStats(a_min=a_min, a_max=a_max, el_max=float(cp.max(el)), el_min=float(cp.min(el)))


def dehaze(lo_img: cp.ndarray, el: cp.ndarray, stats: AtmosphereStats, dep: float,
           win: int, gf: GuidedFilter, omega: float = OMEGA_DEFAULT, additive: bool = False) -> cp.ndarray:
    """Upstream `dehaze_fast2.m` with the atmosphere statistics supplied:
        EL = EL - min(EL);  A = dep * (EL / max(EL) * (a_max - a_min) + a_min)
    omega is the dark-channel prior weight (0.95 upstream; 0 = identity).

    multiplicative (published, Koschmieder model I = J t + A (1 - t)):  J = (I - A) / t + A
    additive (widefield fluorescence is additive; same background estimate B = A (1 - t), applied
    linearly, no 1/t amplification):                                       J = I - A (1 - t)"""
    rng = stats.el_max - stats.el_min
    el_norm = (el - stats.el_min) / rng if rng > 0 else (el - stats.el_min)
    A = dep * (el_norm * (stats.a_max - stats.a_min) + stats.a_min)
    t_est = 1 - omega * dark_channel(lo_img / A, win)
    t = cp.maximum(gf(lo_img, t_est), T_MIN)
    if additive:
        return lo_img - A * (1 - t)
    return (lo_img - A) / t + A


def _disk_otf(H: int, W: int, r_px: float) -> cp.ndarray:
    """OTF (DC at corner) of a uniform disk of radius r_px: geometric defocus blur of a plane |dz| away."""
    y = cp.arange(H, dtype=cp.float64); x = cp.arange(W, dtype=cp.float64)
    yy = cp.minimum(y, H - y)[:, None]; xx = cp.minimum(x, W - x)[None, :]
    disk = (yy ** 2 + xx ** 2 <= r_px ** 2).astype(cp.float64)
    disk /= disk.sum()
    return cp.fft.fft2(disk)


# ----------------------------------------------------------------------------
# the sectioner
# ----------------------------------------------------------------------------
AtmosphereMode = str  # 'per_slice' | 'per_stack' | 'global'


@dataclass
class DSParams:
    background: Union[str, dict] = "severe"   # preset name (mild | moderate | severe) or dict(deg, dep[, hl])
    thres: float = 70.0            # in the [0, 255] working scale
    omega: float = OMEGA_DEFAULT   # dark-channel prior weight = strength / 100 (0 = no change, 0.95 = published)
    model: str = "multiplicative"  # multiplicative (published) | additive (J = I - A(1-t)) | neighbour (nearest-neighbour planes, no dark channel)
                                   # | hybrid (neighbour-plane subtraction first, then additive dark-channel iterations on the residual)
    cutoff: str = "legacy"         # legacy (kc = 0.2 k_m, factor) | optical (defocus-MTF first zero of the nearest out-of-focus plane)
    nn_alpha: float = 0.45         # neighbour model: weight of each blurred neighbour plane
    nn_delta: int = 1              # neighbour model: plane offset
    denoise: bool = True           # closing Gaussian, sigma = 1 px
    atmosphere_mode: AtmosphereMode = "per_stack"
    # for 'global': per-iteration AtmosphereStats (len == maxtime)
    global_atmosphere: Optional[List[AtmosphereStats]] = None
    # reproduce the MATLAB code exactly (integer-rounded cut-off, floor(N/15)+1
    # padding without next_fast_len); for validation against upstream output
    exact_upstream: bool = False
    kc_round: Optional[str] = None      # None -> "floor" if exact_upstream else "exact"; or "nearest"

    @property
    def preset(self) -> dict:
        return resolve_preset(self.background)[1]

    @property
    def preset_label(self) -> str:
        return resolve_preset(self.background)[0]

    @property
    def strength(self) -> float:
        return 100.0 * self.omega

    def __post_init__(self):
        if self.model not in ("multiplicative", "additive", "neighbour", "hybrid"):
            raise ValueError(f"model must be multiplicative | additive | neighbour | hybrid, got {self.model!r}")
        if self.cutoff not in ("legacy", "optical"):
            raise ValueError(f"cutoff must be legacy | optical, got {self.cutoff!r}")

    @property
    def uses_atmosphere(self) -> bool:
        return self.model != "neighbour"

    @property
    def schedule_json(self) -> str:
        """Canonical string of everything that changes the numerical result besides the calibration scale
        (hashable / comparable across calibrations)."""
        d = dict(self.preset, omega=self.omega, model=self.model, cutoff=self.cutoff)
        if self.model in ("neighbour", "hybrid"):
            d.update(nn_alpha=self.nn_alpha, nn_delta=self.nn_delta)
        return json.dumps(d, sort_keys=True)

    @property
    def maxtime(self) -> int:
        return len(self.preset["deg"])


@dataclass
class Padding:
    shape: Tuple[int, int]
    shape_pad: Tuple[int, int]
    before: Tuple[int, int]

    @classmethod
    def for_shape(cls, shape: Tuple[int, int], exact: bool = False) -> "Padding":
        H, W = shape
        if exact:   # upstream: padarray(..., [floor(Nx/15)+1, floor(Ny/15)+1], 'symmetric')
            Hp, Wp = H + 2 * (H // 15 + 1), W + 2 * (W // 15 + 1)
        else:
            Hp = scipy.fft.next_fast_len(H + 2 * (H // 15 + 1))
            Wp = scipy.fft.next_fast_len(W + 2 * (W // 15 + 1))
        return cls(shape=(H, W), shape_pad=(Hp, Wp), before=((Hp - H) // 2, (Wp - W) // 2))

    def pad(self, a: cp.ndarray) -> cp.ndarray:
        (H, W), (Hp, Wp), (by, bx) = self.shape, self.shape_pad, self.before
        return cp.pad(a, ((by, Hp - H - by), (bx, Wp - W - bx)), mode="symmetric")

    def crop(self, a: cp.ndarray) -> cp.ndarray:
        (H, W), (by, bx) = self.shape, self.before
        return a[by:by + H, bx:bx + W]


class DarkSectioner:
    """Runs Dark Sectioning on (Z, Y, X) float arrays already mapped to the
    [0, 255] working scale. Build one per (image shape, optics, params)."""

    def __init__(self, shape_yx: Tuple[int, int], optics: Optics, params: DSParams):
        self.optics, self.params = optics, params
        self.padding = Padding.for_shape(tuple(shape_yx), exact=params.exact_upstream)
        self.fft = FFT2(self.padding.shape_pad)
        kc_round = params.kc_round or ("floor" if params.exact_upstream else "exact")
        self.filters = FilterBank(self.padding.shape_pad, optics, params.preset["deg"], kc_round=kc_round, fft=self.fft, cutoff=params.cutoff)
        self.gf = GuidedFilter(self.padding.shape_pad)
        self.nn_otf = None
        if params.model in ("neighbour", "hybrid"):
            if not optics.z_step_um:
                raise ValueError(f"model={params.model!r} needs the z step (Optics.z_step_um); single planes have no neighbours")
            self.nn_radius_px = optics.defocus_radius_um(params.nn_delta * optics.z_step_um) / (optics.pixel_nm * 1e-3)
            self.nn_otf = _disk_otf(*self.padding.shape_pad, self.nn_radius_px)

    # -- one iteration on one padded slice ------------------------------------
    def _split(self, img: cp.ndarray, f: SliceFilters):
        F = self.fft.fft2(img)
        hi = cp.real(self.fft.ifft2(F * f.hp))
        lo = cp.real(self.fft.ifft2(F * f.lp))
        el = cp.real(self.fft.ifft2(F * f.elp))
        return hi, lo, el

    def _iterate(self, img: cp.ndarray, it: int, stats: Optional[AtmosphereStats]):
        f = self.filters.per_iter[it]
        hi, lo, el = self._split(img, f)
        if stats is None:
            stats = estimate_atmosphere(lo, el, self.params.thres, f.block_size)
        dep = self.params.preset["dep"][it]
        hl = self.params.preset["hl"][self.params.maxtime - 1]
        out = dehaze(lo, el, stats, dep, f.block_size, self.gf, self.params.omega, additive=(self.params.model in ("additive", "hybrid"))) / hl + hi
        return out, stats

    def _estimate_only(self, img: cp.ndarray, it: int) -> AtmosphereStats:
        f = self.filters.per_iter[it]
        F = self.fft.fft2(img)
        lo = cp.real(self.fft.ifft2(F * f.lp))
        el = cp.real(self.fft.ifft2(F * f.elp))
        return estimate_atmosphere(lo, el, self.params.thres, f.block_size)

    # -- public ---------------------------------------------------------------
    def run(self, stack_u: np.ndarray, return_stats: bool = True, collect_would_be_stats: bool = False):
        """stack_u: (Z, Y, X) float array in the working scale.

        Returns (result (Z, Y, X) float32, stats[z][iteration]) where stats are
        the AtmosphereStats actually used, or -- in global mode with
        collect_would_be_stats -- the per-slice values that would have been
        estimated (used by the calibration and for QC).
        """
        stack_u = np.asarray(stack_u)
        if stack_u.ndim == 2:
            stack_u = stack_u[None]
        Z = stack_u.shape[0]
        maxtime = self.params.maxtime
        mode = self.params.atmosphere_mode
        out = np.empty(stack_u.shape, dtype=np.float32)
        stats_log: List[List[Optional[AtmosphereStats]]] = [[None] * maxtime for _ in range(Z)]
        if self.params.model == "neighbour":
            out = self._run_neighbour(stack_u)
            return (out, stats_log) if return_stats else out
        if self.params.model == "hybrid":
            # 1. physics first: subtract the blurred neighbouring planes (large-scale out-of-focus light, no prior)
            # 2. prior second: additive dark-channel iterations on the residual (fine-scale haze the neighbours cannot explain)
            stack_u = np.clip(self._run_neighbour(stack_u), 0.0, None)

        if mode in ("per_slice", "global"):
            fixed = None
            if mode == "global":
                fixed = self.params.global_atmosphere
                if fixed is None or len(fixed) != maxtime:
                    raise ValueError("global atmosphere_mode needs global_atmosphere with one entry per iteration")
            for z in range(Z):
                img = self.padding.pad(cp.asarray(stack_u[z], dtype=cp.float64))
                for it in range(maxtime):
                    if fixed is not None and collect_would_be_stats:
                        stats_log[z][it] = self._estimate_only(img, it)
                    res, st = self._iterate(img, it, None if fixed is None else fixed[it])
                    if stats_log[z][it] is None:
                        stats_log[z][it] = st
                    img = self.padding.pad(self.padding.crop(res)) if it < maxtime - 1 else res
                out[z] = self._finish(img)
        elif mode == "per_stack":
            # iteration k statistics are the medians over all slices of the
            # actual iteration-k inputs, so the stack advances one iteration
            # at a time. Keep it on the GPU only if it comfortably fits.
            Hp, Wp = self.padding.shape_pad
            need = Z * Hp * Wp * 8
            free, _ = cp.cuda.runtime.memGetInfo()
            on_gpu = need < 0.5 * free
            store = [self.padding.pad(cp.asarray(stack_u[z], dtype=cp.float64)) for z in range(Z)] if on_gpu \
                else [None] * Z
            cpu_store = None if on_gpu else np.stack([cp.asnumpy(self.padding.pad(cp.asarray(s, dtype=cp.float64))) for s in stack_u])

            def get(z):
                return store[z] if on_gpu else cp.asarray(cpu_store[z])

            def put(z, arr):
                if on_gpu:
                    store[z] = arr
                else:
                    cpu_store[z] = cp.asnumpy(arr)

            for it in range(maxtime):
                per = [self._estimate_only(get(z), it) for z in range(Z)]
                st = AtmosphereStats(a_min=float(np.median([p.a_min for p in per])),
                                     a_max=float(np.median([p.a_max for p in per])),
                                     el_max=float(np.median([p.el_max for p in per])),
                                     el_min=float(np.median([p.el_min for p in per])))
                for z in range(Z):
                    res, _ = self._iterate(get(z), it, st)
                    stats_log[z][it] = st
                    put(z, self.padding.pad(self.padding.crop(res)) if it < maxtime - 1 else res)
            for z in range(Z):
                out[z] = self._finish(get(z))
                if on_gpu:
                    store[z] = None
        else:
            raise ValueError(f"unknown atmosphere_mode {mode!r}")
        cp.get_default_memory_pool().free_all_blocks()
        return (out, stats_log) if return_stats else out

    def _run_neighbour(self, stack_u: np.ndarray) -> np.ndarray:
        """Classic nearest-neighbour deblurring (Agard 1984; Castleman): the out-of-focus background of plane z
        is estimated from the neighbouring planes blurred by the defocus disk of |delta| planes,
            J(z) = I(z) - alpha * [ D * I(z - delta) + D * I(z + delta) ],
        a missing neighbour at the stack ends is replaced by the existing one (weight 2 alpha). Linear, uses the
        stack instead of a prior, no dark channel. Negative values are kept (uint16 output clips them)."""
        Z = stack_u.shape[0]; d = int(self.params.nn_delta); a = float(self.params.nn_alpha)
        out = np.empty(stack_u.shape, dtype=np.float32)
        cache: Dict[int, cp.ndarray] = {}

        def blurred(z):
            if z not in cache:
                if len(cache) > 2 * d + 2:
                    cache.pop(min(cache))
                F = self.fft.fft2(self.padding.pad(cp.asarray(stack_u[z], dtype=cp.float64)))
                cache[z] = cp.real(self.fft.ifft2(F * self.nn_otf))
            return cache[z]

        for z in range(Z):
            img = self.padding.pad(cp.asarray(stack_u[z], dtype=cp.float64))
            zm, zp = z - d, z + d
            if 0 <= zm and zp < Z:
                bg = a * (blurred(zm) + blurred(zp))
            elif zp < Z:
                bg = 2 * a * blurred(zp)
            elif 0 <= zm:
                bg = 2 * a * blurred(zm)
            else:
                bg = cp.zeros_like(img)
            out[z] = self._finish(img - bg)
        cp.get_default_memory_pool().free_all_blocks()
        return out

    def run_with_mode(self, stack_u, mode: AtmosphereMode, **kw):
        saved = self.params.atmosphere_mode
        self.params.atmosphere_mode = mode
        try:
            return self.run(stack_u, return_stats=True, **kw)
        finally:
            self.params.atmosphere_mode = saved

    def _finish(self, padded: cp.ndarray) -> np.ndarray:
        if self.params.denoise:
            padded = cp_gaussian_filter(padded, sigma=1)
        return cp.asnumpy(self.padding.crop(padded)).astype(np.float32)


# ----------------------------------------------------------------------------
# intensity scaling helpers
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Scale:
    """Linear map raw counts -> working scale: u = 255 (I - lo) / (hi - lo)."""
    lo: float
    hi: float

    @property
    def counts_per_unit(self) -> float:
        return (self.hi - self.lo) / 255.0

    def to_working(self, raw: np.ndarray) -> np.ndarray:
        u = (raw.astype(np.float64) - self.lo) / (self.hi - self.lo) * 255.0
        return np.clip(u, 0.0, None)

    def to_counts(self, u: np.ndarray) -> np.ndarray:
        """Inverse map WITHOUT adding `lo` back: output is 'counts above lo'."""
        return u * self.counts_per_unit

    def thres_counts(self, thres: float) -> float:
        return self.lo + thres * self.counts_per_unit


def to_uint16(counts: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(counts), 0, 65535).astype(np.uint16)
