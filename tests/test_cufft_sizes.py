"""Document the cuFFT failure that broke the legacy GPU port on 2304x2304 fields.

2-D complex128 FFTs on sizes with a large prime factor return garbage on this
GPU/CuPy build (round trip ifft2(fft2(x)) != x), while 1-D and float32 are
fine. 2304 + 2*(2304//15 + 1) = 2612 = 4 * 653 is such a size -- the exact
padding of the MATLAB code and of the legacy dark_sectioning_gpu.py. darksec
pads with scipy.fft.next_fast_len in production and self-tests every padded
size (core.FFT2), falling back to scipy.fft on the CPU when cuFFT is wrong.
With cuFFT >= 10.9 (CUDA 11.8; installed into the development env on 2026-09-10 via the
nvidia-cufft-cu11 wheel) every size is correct and core.cufft_trusted() lets
non-smooth sizes stay on the GPU.

    python tests/test_cufft_sizes.py
"""
import sys
import numpy as np, cupy as cp, scipy.fft as sf
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from darksec.core import cufft_ok, cufft_trusted, cufft_version, FFT2


def factors(n):
    f, d = [], 2
    while d * d <= n:
        while n % d == 0:
            f.append(d); n //= d
        d += 1
    return f + ([n] if n > 1 else [])


def main():
    print("cupy", cp.__version__, "cuda runtime", cp.cuda.runtime.runtimeGetVersion(), "driver", cp.cuda.runtime.driverGetVersion(),
          cp.cuda.runtime.getDeviceProperties(0)["name"].decode(), "cuFFT", cufft_version(), "trusted for all sizes:", cufft_trusted())
    rng = np.random.default_rng(0)
    bad = []
    for n in [512, 582, 588, 1170, 2304, 2306, 2600, 2609, 2611, 2612, 2617, 2625, 2660, 3001, 4001, 4099, 6002]:
        ok = cufft_ok((n, n))
        x = rng.random((n, n))
        err = np.abs(cp.asnumpy(cp.fft.fft2(cp.asarray(x))) - sf.fft2(x)).max() / np.abs(sf.fft2(x)).max()
        print(f"  n={n:5d} factors={str(factors(n)):>22}  round-trip {'OK ' if ok else 'BAD'}   |cuFFT - scipy| / max = {err:.1e}")
        if not ok:
            bad.append(n)
    print("broken sizes on this machine:", bad)
    print("2612 (legacy padding of 2304):", "BROKEN -> explains the legacy all-zero output" if 2612 in bad else "ok with this cuFFT")
    f = FFT2((2612, 2612)); print("FFT2 backend chosen for 2612:", f.backend)
    print("PASS" if 2625 not in bad and 588 not in bad else "FAIL (production sizes broken?)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
