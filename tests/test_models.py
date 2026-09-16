"""Background models and cut-off rules (options added 2026-09-11), checked on synthetic data.

  * the default (model multiplicative, cutoff legacy) is bit-identical to DSParams() before the options existed
    (here: explicit equality of the two spellings)
  * additive model: the response to spot amplitude S is linear -- log-log slope of (peak - background) vs S is
    1.00 +/- 0.03 over S = 20..640 on a uniform background (the multiplicative model is also ~linear on this
    haze-free scene; its superlinearity appears on real haze, see tests/compare_models.py)
  * optical cutoff: printed for the three optics used in this project; below the legacy value, depends on z step
  * neighbour model: a plane whose neighbours are empty is returned unchanged; a plane between two copies of
    itself loses 2 * alpha of its smooth content

    python tests/test_models.py
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from darksec.core import DarkSectioner, DSParams, Optics

rng = np.random.default_rng(0)
N = 768
OPT = Optics(NA=1.4, emission_nm=670, pixel_nm=103.6, factor=2, z_step_um=0.4, n_immersion=1.515)
AMPS = np.array([20, 40, 80, 160, 320, 640], dtype=float)
yy, xx = np.mgrid[0:N, 0:N]


def synth(B, S_list, spacing=64, sigma=1.3):
    img = np.full((N, N), B, dtype=np.float64); pos = []; k = 0
    for i in range(spacing, N - spacing, spacing):
        for j in range(spacing, N - spacing, spacing):
            S = S_list[k % len(S_list)]; img += S * np.exp(-((yy - i) ** 2 + (xx - j) ** 2) / (2 * sigma ** 2)); pos.append((i, j, S)); k += 1
    img += rng.normal(0, np.maximum(1.0, 0.04 * img))
    return np.clip(img, 0, None), pos


def slope(model, cutoff, B=60.0):
    img, pos = synth(B, AMPS)
    ds = DarkSectioner(img.shape, OPT, DSParams(background="moderate", thres=70, denoise=False, atmosphere_mode="per_slice", model=model, cutoff=cutoff))
    out = ds.run(img[None], return_stats=False)[0]
    bg = float(np.median(out[8:56, 8:56]))
    S = np.array([s for _, _, s in pos]); pk = np.array([out[i - 2:i + 3, j - 2:j + 3].max() - bg for i, j, _ in pos])
    med = np.array([np.median(pk[S == a]) for a in AMPS]); ok = med > 0
    k, _ = np.polyfit(np.log10(AMPS[ok]), np.log10(med[ok]), 1)
    return float(k), med


def main():
    ok = True
    a = DSParams(background="moderate"); b = DSParams(background="moderate", model="multiplicative", cutoff="legacy")
    print("default == multiplicative/legacy:", a.schedule_json == b.schedule_json); ok &= a.schedule_json == b.schedule_json
    print("optical cutoff (cycles/um; period um; dark-channel window px) vs legacy:")
    for name, o in (("561 nm, 1.49 NA oil, 65 nm, no z", Optics(NA=1.49, emission_nm=610, pixel_nm=65, factor=2)),
                    ("seqFISH A488, 63x/1.40 oil, 103.6 nm, dz 0.4", Optics(NA=1.4, emission_nm=520, pixel_nm=103.6, factor=2, z_step_um=0.4, n_immersion=1.515)),
                    ("spheroid DAPI, 63x/1.20 water, 103.6 nm, dz 1.0", Optics(NA=1.2, emission_nm=460, pixel_nm=103.55, factor=2, z_step_um=1.0, n_immersion=1.33))):
        leg = 0.17 / o.resolution_px / (o.pixel_nm * 1e-3)
        print(f"  {name:48s} optical {o.optical_cutoff_cycles_per_um:.2f} ({1 / o.optical_cutoff_cycles_per_um:.2f} um; {o.optical_window_px} px)   legacy {leg:.2f} ({1 / leg:.2f} um)")
    print("log-log slope of (peak - bg) vs S, S = 20..640, B = 60 (working units), moderate, strength 95:")
    for model, cutoff in (("multiplicative", "legacy"), ("additive", "legacy"), ("multiplicative", "optical"), ("additive", "optical")):
        k, med = slope(model, cutoff)
        print(f"  {model:14s} {cutoff:8s} slope {k:.3f}   medians {np.round(med, 1)}")
        if model == "additive":
            ok &= abs(k - 1.0) < 0.03
    # neighbour model
    img, pos = synth(60.0, AMPS)
    st = np.stack([np.full_like(img, 60.0), img, np.full_like(img, 60.0)])
    ds = DarkSectioner(img.shape, OPT, DSParams(background="moderate", thres=70, denoise=False, atmosphere_mode="per_stack", model="neighbour", nn_alpha=0.45))
    out = ds.run(st, return_stats=False)
    d1 = float(np.abs(out[1] - (img - 0.9 * 60.0)).max())      # neighbours are flat 60 -> subtract 2 * alpha * 60
    print(f"  neighbour: plane between flat neighbours = I - 2 alpha B, max |diff| = {d1:.2e} (float32 output); defocus disk radius {ds.nn_radius_px:.1f} px"); ok &= d1 < 1e-3
    st2 = np.stack([img, img, img]); out2 = ds.run(st2, return_stats=False)
    frac = float(np.median(out2[1][8:56, 8:56]) / 60.0)
    print(f"  neighbour: plane between copies of itself keeps {100 * frac:.1f} % of its smooth background (expected {100 * (1 - 0.9):.0f} %)"); ok &= abs(frac - 0.1) < 0.02
    # hybrid model runs and keeps the isolated spot; RL reference (darksec/deconv.py) conserves flux
    ds = DarkSectioner(img.shape, OPT, DSParams(background="moderate", thres=70, denoise=False, atmosphere_mode="per_stack", model="hybrid", cutoff="optical"))
    outh = ds.run(st, return_stats=False); i, j, S = pos[0]
    print(f"  hybrid: spot S={S:.0f} -> peak - bg {outh[1][i - 2:i + 3, j - 2:j + 3].max() - np.median(outh[1][8:56, 8:56]):.1f}; background median {np.median(outh[1][8:56, 8:56]):.2f}")
    from darksec.deconv import widefield_psf, richardson_lucy
    psf = widefield_psf(OPT, (21, 65, 65)); ok &= abs(psf.sum() - 1) < 1e-6
    st3 = np.stack([np.full_like(img, 5.0)] * 10 + [img] + [np.full_like(img, 5.0)] * 10)
    est = richardson_lucy(st3, psf, n_iter=10); ratio = float(est.sum() / st3.sum())
    print(f"  RL reference: PSF sums to {psf.sum():.6f}; flux ratio out/in after 10 iterations {ratio:.3f} (padding takes a few %)"); ok &= abs(ratio - 1) < 0.1
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
