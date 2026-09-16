"""Optional validation against the reference implementation's own published input/output pair, if you have that
repository locally (expected at upstream/Dark-sectioning; see README). Dark.m rescales its output to 65535/max, so
everything is compared with scale-invariant statistics (Pearson r, the scale-only fit y = a*x, the affine intercept).
Settings that reproduce the shipped output: per-slice normalisation, thres = 60, kc = nearest(0.2 k_m).

  A0  darksec/upstream_ref.py (CPU)                 vs Dark.tif
  B0  darksec exact_upstream (GPU), same settings   vs Dark.tif
  B0 vs A0 in float (GPU vs CPU numerics, no quantisation)

Skipped (exit 0) when upstream/Dark-sectioning is absent.

    python tests/test_upstream_reference.py
"""
import json, sys, time
from pathlib import Path
import numpy as np, tifffile
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; sys.path.insert(0, str(ROOT))
from darksec.core import DarkSectioner, DSParams, Optics
from darksec.upstream_ref import dark_sectioning_upstream
from darksec.qc import savefig

UP = ROOT / "upstream" / "Dark-sectioning" / "MATLAB_Code"
IN, OUT = UP / "input" / "Mousekidney_561nm_1.49NA_65nm.tif", UP / "output" / "Dark.tif"
OPT = dict(NA=1.49, emwavelength=610, pixelsize=65, factor=2)        # acquisition parameters given with the pair
SETTINGS = dict(background=1, thres=60, normalize="slice", kc_round="nearest")


def to_u16_like_matlab(x):
    return np.clip(np.rint(65535 * x / x.max()), 0, 65535).astype(np.uint16)


def compare(ref, x, name):
    """ref, x: same shape, any float. Scale-invariant statistics."""
    r = ref.astype(np.float64).ravel(); m = x.astype(np.float64).ravel()
    pear = float(np.corrcoef(r, m)[0, 1])
    a = float((r @ m) / (m @ m))                                   # y = a x
    rmse_scale = float(np.sqrt(np.mean((r - a * m) ** 2)) / np.sqrt(np.mean(r ** 2)))
    A = np.vstack([m, np.ones_like(m)]).T                          # y = a x + b
    (a2, b2), *_ = np.linalg.lstsq(A, r, rcond=None)
    per_slice_r = [float(np.corrcoef(ref[z].ravel(), x[z].ravel())[0, 1]) for z in range(ref.shape[0])]
    per_slice_a = [float((ref[z].ravel() @ x[z].ravel()) / (x[z].ravel() @ x[z].ravel())) for z in range(ref.shape[0])]
    out = dict(name=name, pearson_r=pear, scale_a=a, scale_rmse_rel=rmse_scale, affine_a=float(a2),
               affine_b_rel=float(b2 / np.sqrt(np.mean(r ** 2))), per_slice_r_min=min(per_slice_r),
               per_slice_r_median=float(np.median(per_slice_r)), per_slice_scale_cv=float(np.std(per_slice_a) / np.mean(per_slice_a)))
    print(f"{name:34s} r={pear:.6f}  scale-only RMSE={100*rmse_scale:.3f}%  a={a:.4f}  affine b/rms={100*out['affine_b_rel']:.3f}%  "
          f"slice r min/med={min(per_slice_r):.5f}/{np.median(per_slice_r):.5f}  slice-scale CV={100*out['per_slice_scale_cv']:.2f}%")
    return out, per_slice_r


def main():
    if not (IN.exists() and OUT.exists()):
        print(f"SKIPPED: reference repository not found at {UP.parent} (optional check; see README)")
        return 0
    raw = tifffile.imread(str(IN)).astype(np.float64)          # (31, 512, 512)
    ref = tifffile.imread(str(OUT)).astype(np.float64)
    print(f"reference input {raw.shape} [{raw.min():.0f},{raw.max():.0f}]  output {ref.shape} [{ref.min():.0f},{ref.max():.0f}]")

    t0 = time.time(); cpu_f, cpu_u16 = dark_sectioning_upstream(raw, **OPT, **SETTINGS, denoise=0, verbose=True); t_cpu = time.time() - t0
    optics = Optics(NA=OPT["NA"], emission_nm=OPT["emwavelength"], pixel_nm=OPT["pixelsize"], factor=OPT["factor"])
    u_slice = np.stack([255 * (s - s.min()) / (s.max() - s.min()) for s in raw])      # per-slice min-max
    p = DSParams(background="moderate", thres=SETTINGS["thres"], denoise=False, atmosphere_mode="per_slice",
                 exact_upstream=True, kc_round=SETTINGS["kc_round"])
    ds = DarkSectioner(raw.shape[1:], optics, p)
    t0 = time.time(); gpu_f, _ = ds.run(u_slice); t_gpu = time.time() - t0
    gpu_f = gpu_f.astype(np.float64)
    print(f"darksec exact: pad={ds.padding.shape_pad} block={ds.filters.per_iter[0].block_size} "
          f"cutoff={ds.filters.cutoff_cycles_per_px:.5f} c/px  {t_gpu:.1f}s GPU; CPU reference {t_cpu:.1f}s")

    rows, slices = [], {}
    o, s = compare(ref, cpu_u16.astype(np.float64), "A0 upstream_ref (CPU) vs Dark.tif"); rows.append(o); slices["A0"] = s
    o, s = compare(ref, to_u16_like_matlab(gpu_f), "B0 darksec exact (GPU) vs Dark.tif"); rows.append(o); slices["B0"] = s
    o, _ = compare(cpu_f, gpu_f, "B0 vs A0 (float, no quantisation)"); rows.append(o)
    mine16 = to_u16_like_matlab(gpu_f)
    (ROOT / "results").mkdir(exist_ok=True)
    json.dump(dict(rows=rows, seconds=dict(cpu=t_cpu, gpu=t_gpu)), open(ROOT / "results" / "upstream_reference_validation.json", "w"), indent=2)

    # ---- figure: slices, scatter, per-slice r
    zs = [3, 15, 27]
    fig = plt.figure(figsize=(16, 9.5))
    gs = fig.add_gridspec(3, 5, width_ratios=[1, 1, 1, 1, 1.1])
    vmax = np.percentile(ref, 99.9)
    for i, z in enumerate(zs):
        for j, (img, t) in enumerate([(raw[z], "input (uint8)"), (ref[z], "reference output (Dark.tif)"),
                                      (mine16[z], "darksec exact"), (mine16[z].astype(float) - ref[z], "darksec − reference")]):
            ax = fig.add_subplot(gs[i, j])
            if j == 0:
                ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            elif j == 3:
                ax.imshow(img, cmap="RdBu_r", vmin=-0.02 * vmax, vmax=0.02 * vmax)
            else:
                ax.imshow(img, cmap="gray", vmin=0, vmax=vmax)
            ax.set_title(f"z={z}  {t}" + ("  (±2 % of p99.9)" if j == 3 else ""), fontsize=9); ax.axis("off")
    ax = fig.add_subplot(gs[0, 4])
    sel = np.random.default_rng(0).choice(ref.size, 60000, replace=False)
    ax.plot(ref.ravel()[sel], mine16.ravel()[sel], ".", ms=1.5, alpha=0.4)
    ax.plot([0, 65535], [0, 65535], "k--", lw=0.8)
    ax.set_xlabel("reference output (Dark.tif)"); ax.set_ylabel("darksec exact (same 65535/max scaling)")
    ax.set_title(f"r = {rows[1]['pearson_r']:.6f}, scale-only RMSE {100*rows[1]['scale_rmse_rel']:.2f} %", fontsize=9)
    ax = fig.add_subplot(gs[1, 4])
    for k, lab in (("A0", "upstream_ref (CPU)"), ("B0", "darksec exact (GPU)")):
        ax.plot(slices[k], "-o", ms=3, label=lab)
    ax.set_xlabel("z"); ax.set_ylabel("Pearson r vs Dark.tif (per slice)"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    ax = fig.add_subplot(gs[2, 4])
    prof_ref = [np.percentile(ref[z], 99.9) for z in range(ref.shape[0])]
    prof_me = [np.percentile(mine16[z], 99.9) for z in range(ref.shape[0])]
    ax.plot(prof_ref, "k-o", ms=3, label="reference p99.9"); ax.plot(prof_me, "r-o", ms=3, label="darksec p99.9")
    ax.set_xlabel("z"); ax.set_ylabel("uint16 value"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.suptitle("Validation against the reference implementation's input/output pair", fontsize=12)
    fig.tight_layout(); savefig(fig, ROOT / "figs" / "validation" / "upstream_reference.png")

    ok = all(rows[i]["pearson_r"] > 0.999 and rows[i]["scale_rmse_rel"] < 0.01 for i in (0, 1)) and rows[2]["pearson_r"] > 0.99999
    print("PASS" if ok else "FAIL", "(criteria: A0 and B0: r > 0.999 and scale-only RMSE < 1 %; B0 vs A0: r > 0.99999)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
