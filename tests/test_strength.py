"""`processing.strength` (0-100) = 100 x omega, the dark-channel prior weight. Checks on the example field in data/
(512 x 512 x 21 crop, per_stack atmosphere):

  * strength 95 (default) reproduces the published algorithm: identical to DSParams() with the default omega
  * strength 0 is the identity (output == input up to float32 rounding)
  * the median output level decreases monotonically with strength (more atmosphere removed)
  * an explicit dict schedule {deg, dep} is accepted; a numeric `background` is rejected (the knob is `strength`)

    python tests/test_strength.py
"""
import sys
from pathlib import Path
import numpy as np, tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from darksec.core import DSParams, DarkSectioner, Optics, Scale, omega_from_strength, OMEGA_DEFAULT

REF = Path(__file__).resolve().parent.parent / "data" / "A488_raw_1024x1024x21.tif"


def run(u, optics, background, strength):
    ds = DarkSectioner(u.shape[1:], optics, DSParams(background=background, thres=70.0, denoise=False, atmosphere_mode="per_stack", omega=omega_from_strength(strength)))
    return np.asarray(ds.run(u, return_stats=False))


def main():
    ok = True
    dct = DSParams(background={"deg": [8, 4], "dep": [3, 3]}); print(f"dict schedule -> {dct.preset_label} {dct.preset}")
    try:
        DSParams(background=50).preset; print("numeric background accepted (should be rejected)"); ok = False
    except ValueError:
        print("numeric background rejected (use strength)")
    print(f"default omega {OMEGA_DEFAULT} = strength {100 * OMEGA_DEFAULT:g}; strength 0 -> omega {omega_from_strength(0)}")
    if not REF.exists():
        print(f"example stack {REF} not found; parameter checks only"); print("PASS" if ok else "FAIL"); return 0 if ok else 1
    raw = tifffile.imread(str(REF))[:, 0:512, 512:1024].astype(np.float64)
    optics = Optics(NA=1.4, emission_nm=520.0, pixel_nm=103.554, factor=2.0)
    u = Scale(float(raw.min()), float(raw.max())).to_working(raw)
    for name in ("mild", "moderate", "severe"):
        a = np.asarray(DarkSectioner(u.shape[1:], optics, DSParams(background=name, thres=70.0, denoise=False, atmosphere_mode="per_stack")).run(u, return_stats=False))
        b = run(u, optics, name, 95.0)
        d = float(np.abs(a - b).max()); print(f"  {name}: default vs strength 95 max |diff| = {d:.3e}"); ok &= d == 0.0
    ident = float(np.abs(run(u, optics, "moderate", 0.0) - u).max()); print(f"  strength 0: max |output - input| = {ident:.2e} (float32 rounding)"); ok &= ident < 1e-3
    for name in ("mild", "moderate"):
        levels = []
        for s in (0, 30, 60, 80, 95, 100):
            out = run(u, optics, name, s); levels.append(float(np.median(out)))
            print(f"  {name} strength {s:3d}: median output {levels[-1]:8.3f}  p99 {np.percentile(out, 99):7.2f}  fraction <= 0: {100 * (out <= 0).mean():5.1f} %")
        mono = all(levels[i + 1] <= levels[i] + 1e-9 for i in range(len(levels) - 1)); print(f"  {name}: median decreases with strength: {mono}"); ok &= mono
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
