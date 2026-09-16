"""Comparison arm: 3-D Richardson-Lucy deconvolution with a theoretical widefield PSF, on the GPU.

This is NOT a darksec mode. It is the textbook answer to "thick sample, widefield, sectioning and quantitation",
kept here so that the same metrics can be run on it. Requirements it has and dark sectioning does not: a PSF
(here theoretical: scalar Richards-Wolf / Debye integral for the objective NA, emission wavelength, immersion
index; no aberrations, index-matched sample), a z-stack, an iteration count. Non-negative RL conserves the
total intensity of the stack (up to the padded borders), which is why it is the reference for linearity.

    psf = widefield_psf(optics, shape_zyx=(21, 65, 65))          # sums to 1
    out = richardson_lucy(stack, psf, n_iter=30)                  # stack (Z, Y, X) >= 0, float; returns float32
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import cupy as cp
from scipy.special import j0

from .core import Optics


def widefield_psf(optics: Optics, shape_zyx: Tuple[int, int, int], n_theta: int = 400) -> np.ndarray:
    """Scalar widefield PSF |h(r, z)|^2 = |int_0^alpha sqrt(cos t) J0(k n r sin t) exp(i k n z cos t) sin t dt|^2,
    sampled on the voxel grid (pixel_nm, z_step_um), centred, normalised to sum 1. alpha = asin(NA / n)."""
    if not optics.z_step_um:
        raise ValueError("widefield_psf needs Optics.z_step_um")
    Z, Y, X = shape_zyx
    n = optics.n; k = 2 * math.pi / (optics.emission_nm * 1e-3)         # rad / um
    alpha = math.asin(min(optics.NA / n, 0.999))
    th = np.linspace(0.0, alpha, n_theta); w = np.gradient(th)
    st, ct = np.sin(th), np.cos(th); apod = np.sqrt(ct) * st * w
    yy, xx = np.mgrid[0:Y, 0:X]
    r_um = np.hypot(yy - Y // 2, xx - X // 2) * optics.pixel_nm * 1e-3
    r_unique, inv = np.unique(np.round(r_um, 6), return_inverse=True)
    bess = j0(np.outer(r_unique, k * n * st))                              # (R, T)
    psf = np.empty((Z, Y, X), dtype=np.float64)
    for iz in range(Z):
        z_um = (iz - Z // 2) * optics.z_step_um
        phase = np.exp(1j * k * n * z_um * ct)
        amp = bess @ (apod * phase)
        psf[iz] = (np.abs(amp) ** 2)[inv].reshape(Y, X)
    psf /= psf.sum()
    return psf


def _pad_reflect(a: cp.ndarray, pad: Tuple[int, int, int]) -> cp.ndarray:
    pz, py, px = pad
    return cp.pad(a, ((pz, pz), (py, py), (px, px)), mode="reflect")


def richardson_lucy(stack: np.ndarray, psf: np.ndarray, n_iter: int = 30, pad_xy: int = 32, pad_z: int | None = None,
                    pedestal: float | None = None, dtype=cp.float32) -> np.ndarray:
    """Richardson-Lucy with non-negativity, FFT convolutions on the GPU, reflect padding (pad_z defaults to the PSF
    half-depth). Input (Z, Y, X) in counts above the camera floor (noise around 0 allowed); output float32, same shape.

    A flat pedestal (default 3 x the robust noise sigma of the input, at least 1e-3 of its maximum) is added before the
    iterations and subtracted afterwards: RL needs a strictly positive image, otherwise pixels where the estimate goes
    to zero blow up (ratio = image / ~0). The total intensity above the pedestal is conserved up to the padding."""
    Z, Y, X = stack.shape
    pz = psf.shape[0] // 2 if pad_z is None else pad_z
    a = np.asarray(stack, dtype=np.float32)
    if pedestal is None:
        med = float(np.median(a)); mad = float(np.median(np.abs(a - med))) * 1.4826
        pedestal = max(3.0 * mad - min(med, 0.0), 1e-3 * float(a.max()), 1e-6)
    img = _pad_reflect(cp.asarray(np.clip(a + pedestal, 0, None), dtype=dtype), (pz, pad_xy, pad_xy))
    eps = 1e-6 * float(img.max())
    shape = img.shape
    # PSF -> OTF on the padded grid (centre at the origin)
    h = cp.zeros(shape, dtype=dtype); kz, ky, kx = psf.shape
    h[:kz, :ky, :kx] = cp.asarray(psf, dtype=dtype)
    h = cp.roll(h, (-(kz // 2), -(ky // 2), -(kx // 2)), axis=(0, 1, 2))
    otf = cp.fft.rfftn(h); otf_conj = cp.conj(otf)
    del h
    est = cp.full(shape, float(img.mean()), dtype=dtype)
    for _ in range(int(n_iter)):
        conv = cp.fft.irfftn(cp.fft.rfftn(est) * otf, s=shape)
        ratio = img / cp.maximum(conv, eps)
        est *= cp.fft.irfftn(cp.fft.rfftn(ratio) * otf_conj, s=shape)
        cp.maximum(est, 0, out=est)
    out = cp.asnumpy(est[pz:pz + Z, pad_xy:pad_xy + Y, pad_xy:pad_xy + X]).astype(np.float32) - np.float32(pedestal)
    del img, est, otf, otf_conj
    cp.get_default_memory_pool().free_all_blocks()
    return out
