"""Characterise the DS transfer function on synthetic data.

Uniform background B (working-scale units) + Gaussian spots (sigma 1.3 px) of
amplitude S, plus Poisson-like noise. Measures the DS output at the spot peak
and in the background as a function of S and B, for the calibrated-scale
pipeline (fixed thres = 70, severe). This is the plot that says what
"quantitative" can mean for Dark Sectioning: the output is a monotonic but
non-linear function of S whose shape depends on the local background B.

    python tests/response_curve.py   -> figs/response_curve.png + results/response_curve.csv
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
from darksec.core import DarkSectioner, DSParams, Optics, AtmosphereStats
from darksec.qc import savefig

rng = np.random.default_rng(0)
N = 768
optics = Optics(NA=1.4, emission_nm=670, pixel_nm=103.6, factor=2)
amps = np.array([2, 5, 10, 20, 40, 80, 160, 320, 640], dtype=float)
bgs = [5.0, 20.0, 60.0, 120.0]
sigma_spot = 1.3
yy, xx = np.mgrid[0:N, 0:N]


def synth(B, S_list, spacing=64):
    img = np.full((N, N), B, dtype=np.float64)
    pos = []
    k = 0
    for i in range(spacing, N - spacing, spacing):
        for j in range(spacing, N - spacing, spacing):
            S = S_list[k % len(S_list)]
            img += S * np.exp(-((yy - i) ** 2 + (xx - j) ** 2) / (2 * sigma_spot ** 2))
            pos.append((i, j, S)); k += 1
    # read-noise-like scatter: 4 % of level, min 1 unit
    img += rng.normal(0, np.maximum(1.0, 0.04 * img))
    return np.clip(img, 0, None), pos


rows = []
CASES = [("per_slice", "severe"), ("global", "severe"), ("global", "moderate"), ("global", "mild")]
for mode, bgp in CASES:
    for B in bgs:
        img, pos = synth(B, amps)
        params = DSParams(background=bgp, thres=70, denoise=False, atmosphere_mode="per_slice")
        ds = DarkSectioner(img.shape, optics, params)
        if mode == "global":
            # a "dataset" atmosphere: medians over the four background levels' per-slice stats
            st_all = []
            for Bk in bgs:
                im_k, _ = synth(Bk, amps)
                _, st = ds.run_with_mode(im_k[None], "per_slice"); st_all.append(st[0])
            ga = [AtmosphereStats(a_min=float(np.median([s[it].a_min for s in st_all])),
                                  a_max=float(np.median([s[it].a_max for s in st_all])),
                                  el_max=float(np.median([s[it].el_max for s in st_all])),
                                  el_min=float(np.median([s[it].el_min for s in st_all]))) for it in range(ds.params.maxtime)]
            ds.params.global_atmosphere = ga
            out, _ = ds.run_with_mode(img[None], "global")
        else:
            out, _ = ds.run_with_mode(img[None], "per_slice")
        out = out[0]
        bg_out = float(np.median(out[8:56, 8:56]))
        for (i, j, S) in pos:
            peak = float(out[i - 2:i + 3, j - 2:j + 3].max())
            rows.append(dict(mode=f"{mode}/{bgp}", B=B, S=S, out_peak=peak, out_bg=bg_out, in_peak=B + S))
df = pd.DataFrame(rows)
(HERE.parent / "results").mkdir(exist_ok=True)
df.to_csv(HERE.parent / "results" / "response_curve.csv", index=False)
g = df.groupby(["mode", "B", "S"]).agg(out=("out_peak", "median"), sd=("out_peak", "std"), bg=("out_bg", "median")).reset_index()

fig, axes = plt.subplots(1, len(CASES), figsize=(5.2 * len(CASES), 4.6))
for ax, (mode, bgp) in zip(axes, CASES):
    mode = f"{mode}/{bgp}"
    for B in bgs:
        s = g[(g["mode"] == mode) & (g["B"] == B)]
        ax.errorbar(s["S"], s["out"] - s["bg"], yerr=s["sd"], marker="o", ms=4, capsize=2, label=f"background B = {B:.0f}")
    ax.plot(amps, amps, "k--", lw=0.8, label="identity (out = S)")
    ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel("spot amplitude S above background (working units, 255 = hi)")
    ax.set_ylabel("DS output at spot peak − DS background")
    ax.set_title(f"atmosphere_mode = {mode}"); ax.grid(alpha=.3); ax.legend(fontsize=8)
fig.suptitle("Dark Sectioning transfer function (thres=70, Gaussian spots σ=1.3 px); dashed = identity")
fig.tight_layout(); savefig(fig, HERE.parent / "figs" / "response_curve.png")
print(g.pivot_table(index=["mode", "S"], columns="B", values="out").round(1).to_string())
