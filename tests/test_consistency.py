"""Context independence: with a global calibration, a 1024x1024 crop processed
on its own must give the same DS values as the same region inside the full
2304x2304 field (per-image normalisation would break this). Also checks that
the output does not depend on how many slices are in the stack (global mode).

    python tests/test_consistency.py [--dataset <Leica project folder name>] [--field B/3/P1] [--channel 635nm]

The dataset must have been calibrated and processed by run_ds.py (calibration.json under output_root).
"""
import argparse, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
from darksec.paths import RAW_ROOT, DS_ROOT, DS_TELO_ROOT, PROJECTS_ROOT
from darksec.calibrate import Calibration
from darksec.core import DarkSectioner, DSParams
from darksec.leica import index_dataset



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="2026_09_04_11_46_43--Project003", help="Leica project folder name under DARKSEC_RAW_ROOT")
    ap.add_argument("--field", default="B/3/P1", help="suffix of the image's relative path")
    ap.add_argument("--channel", default="635nm")
    a = ap.parse_args()
    RAW = RAW_ROOT / a.dataset
    cal = Calibration.load(DS_ROOT / RAW.name / "calibration.json")
    im = [i for i in index_dataset(RAW) if i.rel.endswith(a.field)][0]
    ccal = cal.channels[a.channel]
    bg = cal.settings.get("background", "moderate")   # the calibration is preset-specific
    raw = im.read_stack(c=0)[8:13]                     # 5 slices
    u = ccal.scale.to_working(raw)
    results = {}; extra = {}; small_rows = []
    for mode in ["global", "per_slice"]:
        p = DSParams(background=bg, thres=ccal.thres, denoise=False, atmosphere_mode=mode,
                     global_atmosphere=ccal.atmosphere if mode == "global" else None)
        full = DarkSectioner(u.shape[1:], ccal.optics, p).run(u, return_stats=False)
        for size in (1024, 512, 256):          # ROI sizes; the margin excludes the reflection-padding zone
            y0 = x0 = 1152 - size // 2
            crop = DarkSectioner((size, size), ccal.optics, p).run(u[:, y0:y0 + size, x0:x0 + size], return_stats=False)
            m = max(16, size // 16)
            a = full[:, y0 + m:y0 + size - m, x0 + m:x0 + size - m]
            b = crop[:, m:-m, m:-m]
            d = np.abs(a - b); ref = np.percentile(a, 99.9) + 1e-9
            rel = float(np.percentile(d, 99.9)) / ref
            r = np.corrcoef(a.ravel()[::13], b.ravel()[::13])[0, 1]
            small_rows.append(dict(mode=mode, crop_px=size, margin_px=m, diff_p99_9_pct=rel * 100, diff_max_pct=float(d.max() / ref * 100),
                                   diff_rms_pct=float(np.sqrt((d ** 2).mean()) / ref * 100), r=r))
            print(f"{mode:10s} {size:4d}² crop vs full: |diff| p99.9 = {rel*100:.3f} %, max = {d.max()/ref*100:.2f} %, r = {r:.6f}")
            if size == 1024:
                results[mode] = (rel, r)
                extra.setdefault(mode, {}).update(diff_max_pct=float(d.max() / ref * 100), diff_rms_pct=float(np.sqrt((d ** 2).mean()) / ref * 100))
    import pandas as pd
    rows = []
    for mode, (rel, r) in results.items():
        rows.append(dict(mode=mode, dataset=RAW.name, field=a.field, channel=a.channel, background=bg, crop_px=1024,
                         diff_p99_9_pct=rel * 100, r=r))
    # stack-length independence (global only): 1 slice vs inside 5 slices
    p = DSParams(background=bg, thres=ccal.thres, denoise=False, atmosphere_mode="global", global_atmosphere=ccal.atmosphere)
    ds = DarkSectioner(u.shape[1:], ccal.optics, p)
    one = ds.run(u[2:3], return_stats=False)[0]
    five = ds.run(u, return_stats=False)[2]
    print(f"global     1-slice vs 5-slice: max|diff| = {np.abs(one-five).max():.2e}")
    for r_ in rows:
        r_.update(extra[r_["mode"]])
    (HERE.parent / "results").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(HERE.parent / "results" / "consistency.csv", index=False)
    pd.DataFrame(small_rows).to_csv(HERE.parent / "results" / "consistency_roi_sizes.csv", index=False)
    ok = results["global"][0] < 0.01 and np.abs(one - five).max() < 1e-4
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
