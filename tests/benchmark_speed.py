"""Speed for Figure 1 panel D -- the published algorithm (CPU implementation, darksec/upstream_ref.py)
versus darksec, same settings (background=1 / moderate, thres 70, no denoise),
same inputs, per-stack scaling, compute only (in memory):

  transcript_1thread   darksec/upstream_ref.py, 1 thread            (CSV labels kept for the figure scripts)
  transcript_parallel  same code, slices over 20 processes, BLAS/OpenMP pinned to 1 thread per worker
  darksec              GPU, fast-length padding

Also a stage breakdown of the darksec pipeline. NOT measured: the MATLAB
parfor implementation and the Fiji plugin (GUI only). The lab's earlier
in-house ports are not baselines for the figure (results/figure1/speed_legacy_ports.csv
keeps that earlier run for the record).

    python tests/benchmark_speed.py [out_dir] -> <out_dir>/speed.csv, speed_breakdown.csv (default results/figure1)
"""
import os, sys, time
from pathlib import Path
import numpy as np, pandas as pd, tifffile

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; sys.path.insert(0, str(ROOT))
from darksec.paths import RAW_ROOT, DS_ROOT, DS_TELO_ROOT, PROJECTS_ROOT
from darksec.core import DarkSectioner, DSParams, Optics, Scale, to_uint16
from darksec.leica import index_dataset
from darksec.upstream_ref import dark_sectioning_upstream, dark_sectioning_upstream_parallel
import cupy as cp

RAW = RAW_ROOT / Path("2026_09_08_14_31_32--Project001")
DATA = ROOT / "data" / "A488_raw_1024x1024x21.tif"                  # example field; the 512^2 case is its top-right crop (background + spots)
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "figure1"
N_CPU = os.cpu_count()
SCR = ROOT / "results" / "bench_tmp"; SCR.mkdir(parents=True, exist_ok=True)
GPU_NAME = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()


def timed(fn, reps):
    ts = []
    for _ in range(reps):
        cp.cuda.Device().synchronize(); t0 = time.perf_counter(); out = fn(); cp.cuda.Device().synchronize(); ts.append(time.perf_counter() - t0)
    return ts, out


def main():
    cases = {}
    cases["example_512x512x21"] = dict(raw=tifffile.imread(str(DATA))[:, 0:512, 512:1024], NA=1.4, em=520, px=103.554, src="TIFF (data/ example field, 512^2 crop)")
    im = mm = None
    if RAW.exists():
        im = [i for i in index_dataset(RAW) if i.rel.endswith("A/3/P1")][0]; mm = im.memmap()
        cases["leica_2304x2304x26"] = dict(raw=im.read_stack(c=0, mm=mm), NA=im.NA, em=670, px=im.pixel_nm, src=f"{im.rel} 635nm (.lof)")
    else:
        print(f"dataset {RAW} not found (set DARKSEC_RAW_ROOT): the 2304x2304x26 case is skipped", flush=True)
    rows, breakdown = [], []
    for name, c in cases.items():
        raw, Z, iters = c["raw"], c["raw"].shape[0], 2
        heavy = raw.shape[-1] > 1000
        kw = dict(NA=c["NA"], emwavelength=c["em"], pixelsize=c["px"], factor=2, background=1, thres=70)
        print(f"== {name}  Z={Z} {raw.shape[1:]}  {c['src']}", flush=True)
        ts, up = timed(lambda: dark_sectioning_upstream(raw, denoise=0, **kw)[0], 1 if heavy else 3)
        rows.append(dict(case=name, impl="transcript_1thread", device="CPU, 1 thread (NumPy/SciPy)", seconds=min(ts), reps=len(ts), Z=Z, iters=iters))
        print(f"  CPU reference, 1 thread  {min(ts):8.1f} s", flush=True)
        ts, upp = timed(lambda: dark_sectioning_upstream_parallel(raw, n_jobs=N_CPU, **kw)[0], 2 if heavy else 3)
        rows.append(dict(case=name, impl="transcript_parallel", device=f"CPU, {N_CPU} processes x 1 thread (joblib + threadpoolctl)", seconds=min(ts), reps=len(ts), Z=Z, iters=iters,
                         max_abs_diff_vs_1thread=float(np.abs(up - upp).max())))
        print(f"  CPU reference, parallel  {min(ts):8.1f} s   (max|diff| vs 1 thread {np.abs(up-upp).max():.1e})", flush=True)
        optics = Optics(NA=c["NA"], emission_nm=c["em"], pixel_nm=c["px"], factor=2)
        scale = Scale(lo=float(raw.min()), hi=float(raw.max())); u = scale.to_working(raw)
        p = DSParams(background="moderate", thres=70, denoise=False, atmosphere_mode="per_slice", exact_upstream=False)
        ds = DarkSectioner(raw.shape[1:], optics, p); ds.run(u[:1], return_stats=False)   # warm-up
        ts, out = timed(lambda: ds.run(u, return_stats=False), 3)
        r = float(np.corrcoef(up.ravel()[::7], out.astype(np.float64).ravel()[::7])[0, 1])
        rows.append(dict(case=name, impl="darksec", device=f"GPU ({GPU_NAME}), pad {ds.padding.shape_pad[0]}", seconds=float(np.median(ts)), reps=len(ts), Z=Z, iters=iters, r_vs_cpu_reference=r))
        print(f"  darksec GPU              {np.median(ts):8.2f} s   (r vs CPU reference {r:.6f})", flush=True)
        t = {}
        t0 = time.perf_counter(); rr = im.read_stack(c=0, mm=mm) if heavy else tifffile.imread(str(DATA))[:, 0:512, 512:1024]; t["read"] = time.perf_counter() - t0
        t0 = time.perf_counter(); uu = scale.to_working(rr); t["scale"] = time.perf_counter() - t0
        cp.cuda.Device().synchronize(); t0 = time.perf_counter(); oo = ds.run(uu, return_stats=False); cp.cuda.Device().synchronize(); t["gpu"] = time.perf_counter() - t0
        t0 = time.perf_counter(); u16 = to_uint16(scale.to_counts(oo)); t["uint16"] = time.perf_counter() - t0
        t0 = time.perf_counter(); tifffile.imwrite(str(SCR / f"{name}_darksec.tif"), u16); t["write"] = time.perf_counter() - t0
        breakdown.append(dict(case=name, **t, total=sum(t.values()), read_src="9p .lof memmap (cached)" if heavy else "local TIFF"))
    df = pd.DataFrame(rows); df["ms_per_slice_iter"] = df["seconds"] / (df["Z"] * df["iters"]) * 1000
    base = df[df.impl == "transcript_1thread"].set_index("case")["seconds"]
    df["speedup_vs_1thread"] = [base[c] / s for c, s in zip(df["case"], df["seconds"])]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "speed.csv", index=False); pd.DataFrame(breakdown).to_csv(OUT_DIR / "speed_breakdown.csv", index=False)
    print(df.round(3).to_string(index=False)); print(pd.DataFrame(breakdown).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
