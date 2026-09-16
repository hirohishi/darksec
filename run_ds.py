#!/usr/bin/env python
"""
Dark Sectioning batch CLI.

    python run_ds.py index                       # list image files + channels of every dataset
    python run_ds.py calibrate [--recalibrate]   # calibration.json per dataset
    python run_ds.py run [--files A/2 B/3] [--limit 2] [--mode per_stack] [--overwrite]
    python run_ds.py qc                          # figures (needs processed outputs)

    --datasets <dir> ...   restrict to these project folders (default: all in config.yaml)
    --config path.yaml     (default: config.yaml next to this script)
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["index", "calibrate", "run", "qc"])
    ap.add_argument("--config", type=Path, default=HERE / "config.yaml")
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--files", nargs="*", help="substring filters on the relative path")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--mode", choices=["global", "per_stack", "per_slice"])
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--recalibrate", action="store_true")
    a = ap.parse_args()

    from darksec.batch import load_config, run_dataset, get_calibration, index_entry, dataset_name
    from darksec.leica import describe
    cfg = load_config(a.config)
    if a.datasets:   # a name or path given on the command line selects/overrides config entries
        by_name = {dataset_name(e): e for e in cfg["datasets"]}
        datasets = [by_name.get(d, by_name.get(Path(d).name, d)) for d in a.datasets]
    else:
        datasets = cfg["datasets"]
    out_root = Path(cfg["output_root"])

    if a.command == "index":
        for d in datasets:
            ims = index_entry(d, include_icc=bool(cfg.get("include_icc", False)))
            print(f"=== {dataset_name(d)}: {len(ims)} image files"); print(describe(ims))
    elif a.command == "calibrate":
        for d in datasets:
            ims = index_entry(d, include_icc=bool(cfg.get("include_icc", False)))
            cal = get_calibration(ims, cfg, out_root, dataset_name(d), force=a.recalibrate)
            print(cal.summary())
    elif a.command == "run":
        for d in datasets:
            run_dataset(d, cfg, file_filter=a.files, limit=a.limit, overwrite=a.overwrite,
                        recalibrate=a.recalibrate, mode_override=a.mode)
    elif a.command == "qc":
        from darksec.qc import qc_dataset, summarize
        for d in datasets:
            qc_dataset(d, cfg, HERE / "figs")
        summarize(datasets, cfg, HERE / "results")


if __name__ == "__main__":
    main()
