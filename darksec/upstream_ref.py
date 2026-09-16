"""
Independent NumPy implementation of the published Dark sectioning algorithm
(Cao et al., Nature Methods 2025), written to reproduce the documented behaviour
of the reference implementation (github.com/Cao-ruijie/Dark-sectioning, commit
f131db3). CPU only, float64. It runs the published pipeline on our own data
without MATLAB and is the CPU reference for `darksec.core`; when the reference
repository is available locally, tests/test_upstream_reference.py compares both
with the input/output pair it ships.

Behaviour reproduced:
  * min-max normalisation of the input to [0, 255] (per stack or per slice)
  * zero-padding to a square, then symmetric padding by floor(N/15) + 1
  * Hi/Lo split by Fourier Gaussians at kc = 0.2 k_m (floor or nearest integer),
    very-low-pass EL at kc/deg
  * dark-channel window = half-width of the low-passed PSF (first drop below 1 %)
  * dehazing of Lo with the dark-channel prior: atmosphere from the top 1 % of
    the dark channel, interpolated between a masked minimum and the maximum by
    EL - min(EL), transmission smoothed by a guided filter (r 15, eps 1e-3), t >= 0.1
  * background = 1 -> 2 iterations, deg [6, 3], dep [3, 3], hl = 1
  * output: uint16(65535 * result / max(result))

The dark-channel and guided-filter functions follow the MIT-licensed MATLAB code
of Stephen Tierney (2014, github.com/sjtrny/Dark-Channel-Haze-Removal), which the
reference implementation also builds on; see NOTICE.

Choices that do not change the result: the dark channel is a separable minimum
filter (O(N^2)) rather than an explicit patch loop, the atmosphere uses a partial
sort, and PSF and filters are cached per padded size.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Tuple

import numpy as np
import scipy.fft as sfft
from scipy.ndimage import minimum_filter, correlate
from scipy.special import j1

OMEGA = 0.95          # dark-channel prior weight of the published algorithm
GUIDED_R, GUIDED_EPS = 15, 0.001
T_MIN = 0.1           # transmission floor
SCHEDULES = {1: dict(deg=[6.0, 3.0], dep=[3.0, 3.0]),    # background = 1: two iterations
             0: dict(deg=[6.0], dep=[3.0])}              # background = 0: one iteration
HL = 1.0              # weight of the dehazed low-pass part when recombining (1 in the published settings)


# ---------------------------------------------------------------- Fourier filters and PSF
def _fft_index(n: int) -> np.ndarray:
    """Integer frequency indices in FFT order: 0, 1, ..., ceil(n/2)-1, -floor(n/2), ..., -1."""
    return np.concatenate([np.arange(0, (n - 1) // 2 + 1), np.arange(-(n // 2), 0)]).astype(np.float64)


def gaussian_lowpass(n_rows: int, n_cols: int, sigma: float) -> np.ndarray:
    """Fourier-space Gaussian in FFT order; width `sigma` along columns and (n_rows/n_cols)*sigma along rows."""
    kx, ky = _fft_index(n_cols), _fft_index(n_rows)
    sx, sy = sigma, (n_rows / n_cols) * sigma
    return np.exp(-(kx[None, :] ** 2 / sx ** 2 + ky[:, None] ** 2 / sy ** 2))


def airy_psf(wavelength_nm: float, pixel_nm: float, NA: float, n: int, factor: float) -> np.ndarray:
    """Normalised Airy intensity |2 J1(v)/v|^2 on an n x n periodic grid, centred with fftshift."""
    d = np.arange(n, dtype=np.float64)
    d = np.minimum(d, n - d)                                   # periodic distance to the origin
    radius = np.sqrt(d[None, :] ** 2 + d[:, None] ** 2)
    v = (2 * np.pi * NA / wavelength_nm * pixel_nm * factor) * radius + np.finfo(float).eps
    psf = (2 * j1(v) / v) ** 2
    return np.fft.fftshift(psf / psf.sum())


@lru_cache(maxsize=16)
def fourier_filters(n_rows, n_cols, NA, emwavelength, pixelsize, factor, deg, divide, kc_round):
    """(lp, hp, elp): low-pass / high-pass split at kc = 0.2 k_m and the very-low-pass for EL."""
    resolution = 0.5 * emwavelength / NA / factor
    k_m = n_cols / (resolution / pixelsize)
    kc = math.floor(k_m * 0.2) if kc_round == "floor" else round(k_m * 0.2)
    sigma = kc * 2 / 2.355
    lp = gaussian_lowpass(n_rows, n_cols, sigma * 2 * divide)
    return lp, 1 - lp, gaussian_lowpass(n_rows, n_cols, sigma / deg)


@lru_cache(maxsize=16)
def dark_channel_window(n_rows, n_cols, NA, emwavelength, pixelsize, factor, deg, divide, kc_round) -> int:
    """Window of the dark-channel minimum filter: distance from the centre at which the low-passed PSF
    first drops below 1 % of its peak (measured along one column through the centre)."""
    lp, _, _ = fourier_filters(n_rows, n_cols, NA, emwavelength, pixelsize, factor, deg, divide, kc_round)
    psf_lo = np.abs(sfft.ifft2(sfft.fft2(airy_psf(emwavelength, pixelsize, NA, n_rows, factor)) * lp))
    c = n_rows // 2
    profile = psf_lo[c - 1:, c - 1] / psf_lo.max()
    below = np.flatnonzero(profile < 0.01)
    return int(below[0]) if below.size else n_rows - c


def split_bands(image: np.ndarray, filters) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(Hi, Lo, EL) of one plane for the filters of fourier_filters()."""
    lp, hp, elp = filters
    F = sfft.fft2(image)
    return (np.real(sfft.ifft2(F * hp)), np.real(sfft.ifft2(F * lp)), np.real(sfft.ifft2(F * elp)))


# ---------------------------------------------------------------- dark-channel prior (after S. Tierney, MIT; see NOTICE)
def window_sum_filter(image: np.ndarray, r: int) -> np.ndarray:
    """Sum over the (2r+1)^2 window around every pixel, windows clipped at the border (cumulative sums).
    After window_sum_filter.m, Copyright (c) 2014 Stephen Tierney (MIT)."""
    h, w = image.shape
    s = np.zeros_like(image)
    c = np.cumsum(image, axis=0)
    s[:r + 1, :] = c[r:2 * r + 1, :]
    s[r + 1:h - r, :] = c[2 * r + 1:h, :] - c[:h - 2 * r - 1, :]
    s[h - r:h, :] = c[h - 1:h, :] - c[h - 2 * r - 1:h - r - 1, :]
    c = np.cumsum(s, axis=1)
    out = np.zeros_like(image)
    out[:, :r + 1] = c[:, r:2 * r + 1]
    out[:, r + 1:w - r] = c[:, 2 * r + 1:w] - c[:, :w - 2 * r - 1]
    out[:, w - r:w] = c[:, w - 1:w] - c[:, w - 2 * r - 1:w - r - 1]
    return out


def guided_filter(guide: np.ndarray, target: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """He's guided filter with box sums. After guided_filter.m, Copyright (c) 2014 Stephen Tierney (MIT)."""
    N = window_sum_filter(np.ones_like(guide), radius)
    mean_g = window_sum_filter(guide, radius) / N
    mean_t = window_sum_filter(target, radius) / N
    var_g = window_sum_filter(guide * guide, radius) / N - mean_g * mean_g
    cov_gt = window_sum_filter(guide * target, radius) / N - mean_g * mean_t
    a = cov_gt / (var_g + eps)
    b = mean_t - a * mean_g
    return window_sum_filter(a, radius) / N * guide + window_sum_filter(b, radius) / N


def get_dark_channel(image: np.ndarray, win_size: int) -> np.ndarray:
    """Minimum over a win_size x win_size window (border padded with +inf). After get_dark_channel.m (Tierney, MIT)."""
    return minimum_filter(image, size=win_size, mode="constant", cval=np.inf)


def get_atmosphere(image: np.ndarray, dark_channel: np.ndarray) -> float:
    """Mean of the image over the brightest 1 % of the dark channel. After get_atmosphere.m (Tierney, MIT)."""
    n = math.floor(dark_channel.size * 0.01)
    flat = dark_channel.ravel()
    idx = np.argpartition(flat, flat.size - n)[flat.size - n:]
    return float(image.ravel()[idx].mean())


def dehaze_lowpass(lo: np.ndarray, omega: float, win_size: int, el: np.ndarray, dep: float, thres: float) -> np.ndarray:
    """Dark-channel dehazing of the low-pass plane with a spatially varying atmosphere.
    After dehaze_fast2.m / get_transmission_estimate.m / get_radiance.m (Tierney, MIT) as used by the reference
    implementation: the atmosphere is interpolated between its estimate on the sub-threshold pixels (minimum)
    and on the whole plane (maximum) by the very-low-pass image EL."""
    mask = (lo < thres).astype(np.float64)
    a_min = get_atmosphere(lo * mask, get_dark_channel(lo * mask, win_size))
    a_max = get_atmosphere(lo, get_dark_channel(lo, win_size))
    el = el - el.min()
    atmosphere = dep * (el / el.max() * (a_max - a_min) + a_min)
    transmission = guided_filter(lo, 1 - omega * get_dark_channel(lo / atmosphere, win_size), GUIDED_R, GUIDED_EPS)
    return (lo - atmosphere) / np.maximum(transmission, T_MIN) + atmosphere


# ---------------------------------------------------------------- pipeline
def _normalised_square(stack_zyx: np.ndarray, normalize: str):
    """(Nz, N, N) float64 in [0, 255] (per stack or per slice), zero-padded to a square; plus the original (rows, cols)."""
    x = np.asarray(stack_zyx, dtype=np.float64)
    if x.ndim == 2:
        x = x[None]
    if normalize == "slice":
        mn = x.min(axis=(1, 2), keepdims=True); mx = x.max(axis=(1, 2), keepdims=True)
    elif normalize == "stack":
        mn, mx = x.min(), x.max()
    else:
        raise ValueError("normalize must be 'stack' or 'slice'")
    x = 255 * (x - mn) / (mx - mn)
    rows, cols = x.shape[1:]
    n = max(rows, cols)
    return np.pad(x, ((0, 0), (0, n - rows), (0, n - cols))), (rows, cols)


def _pad(plane: np.ndarray, p: int) -> np.ndarray:
    return np.pad(plane, p, mode="symmetric")


def _process_plane(plane: np.ndarray, p: int, optics: dict, schedule: dict, thres: float, divide: float, omega: float,
                   kc_round: str) -> np.ndarray:
    """All iterations of the algorithm on one (unpadded, square) plane."""
    n = plane.shape[0]
    npad = n + 2 * p
    for deg, dep in zip(schedule["deg"], schedule["dep"]):
        key = (npad, npad, optics["NA"], optics["emwavelength"], optics["pixelsize"], optics["factor"], deg, divide, kc_round)
        hi, lo, el = split_bands(_pad(plane, p), fourier_filters(*key))
        lo = dehaze_lowpass(lo, omega, dark_channel_window(*key), el, dep, thres)
        plane = (lo / HL + hi)[p:p + n, p:p + n]
    return plane


def _denoise_plane(plane: np.ndarray, p: int) -> np.ndarray:
    """Optional closing smoothing: 2 x 2 Gaussian (sigma 1) with replicated borders, as in the reference settings."""
    g = np.exp(-(0.5 ** 2 + 0.5 ** 2) / 2) * np.ones((2, 2))
    g /= g.sum()
    n = plane.shape[0]
    return correlate(_pad(plane, p), g, mode="nearest")[p:p + n, p:p + n]


def _finish(planes, rows: int, cols: int) -> Tuple[np.ndarray, np.ndarray]:
    """Crop to the original size; float result and the uint16 output scaled to 65535/max."""
    out = np.stack(planes)[:, :rows, :cols]
    u16 = np.clip(np.rint(65535 * out / out.max()), 0, 65535).astype(np.uint16)
    return out, u16


def dark_sectioning_upstream(stack_zyx: np.ndarray, *, NA: float, emwavelength: float, pixelsize: float,
                             factor: float = 2.0, background: int = 1, thres: float = 70.0, pad_size: int = 15,
                             denoise: int = 0, divide: float = 0.5, verbose: bool = False,
                             dep_matrix=None, deg_matrix=None, omega: float = OMEGA,
                             normalize: str = "stack", kc_round: str = "floor") -> Tuple[np.ndarray, np.ndarray]:
    """Published algorithm on a (Z, Y, X) stack. Returns (float64 result, uint16 output scaled to 65535/max).

    normalize  'stack' (one min/max for the stack) or 'slice' (every plane stretched to [0, 255] on its own)
    kc_round   'floor' or 'nearest' integer for the cut-off kc = 0.2 k_m
    """
    x, (rows, cols) = _normalised_square(stack_zyx, normalize)
    n = x.shape[1]
    p = n // pad_size + 1
    schedule = dict(SCHEDULES[1 if background == 1 else 0])
    if deg_matrix is not None:
        schedule["deg"] = [float(v) for v in deg_matrix]
    if dep_matrix is not None:
        schedule["dep"] = [float(v) for v in dep_matrix]
    optics = dict(NA=NA, emwavelength=emwavelength, pixelsize=pixelsize, factor=factor)
    if verbose:
        for i, deg in enumerate(schedule["deg"]):
            w = dark_channel_window(n + 2 * p, n + 2 * p, NA, emwavelength, pixelsize, factor, deg, divide, kc_round)
            print(f"  reference implementation, iteration {i + 1}/{len(schedule['deg'])}: dark-channel window {w}")
    planes = [_process_plane(x[z], p, optics, schedule, thres, divide, omega, kc_round) for z in range(x.shape[0])]
    if denoise:
        planes = [_denoise_plane(pl, p) for pl in planes]
    return _finish(planes, rows, cols)


def _process_plane_pinned(plane, p, optics, schedule, thres, divide, omega, kc_round):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):          # one BLAS/OpenMP thread per worker process
        return _process_plane(plane, p, optics, schedule, thres, divide, omega, kc_round)


def dark_sectioning_upstream_parallel(stack_zyx, *, NA, emwavelength, pixelsize, factor=2.0, background=1, thres=70.0,
                                      pad_size=15, divide=0.5, omega=OMEGA, normalize="stack", kc_round="floor",
                                      n_jobs=-1):
    """Same result as dark_sectioning_upstream (denoise=0); planes are independent once normalised, so they are
    distributed over processes."""
    from joblib import Parallel, delayed
    x, (rows, cols) = _normalised_square(stack_zyx, normalize)
    n = x.shape[1]
    p = n // pad_size + 1
    schedule = SCHEDULES[1 if background == 1 else 0]
    optics = dict(NA=NA, emwavelength=emwavelength, pixelsize=pixelsize, factor=factor)
    planes = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_process_plane_pinned)(x[z], p, optics, schedule, thres, divide, omega, kc_round) for z in range(x.shape[0]))
    return _finish(planes, rows, cols)
