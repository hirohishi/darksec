"""
QC figures for a processed dataset. Two families, kept in separate folders:

figs/<dataset>/unified/   every panel of a channel shares ONE display range
                          (raw: [lo, hi] of the calibration; DS: [0, p99.9 of the
                          DS output over all files]) -> brightness IS comparable
figs/<dataset>/autoscale/ each panel scaled on its own -> morphology only

Plus per-channel z-profiles (raw vs DS) and DS histograms per file.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile

from .calibrate import Calibration, channel_key
from .leica import LofImage, index_dataset
from .pipeline import output_paths


def savefig(fig, path_png: Path) -> None:
    """Every figure is written as PNG (quick look) and SVG (editable)."""
    path_png = Path(path_png)
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png, dpi=110)
    plt.rcParams["svg.hashsalt"] = path_png.stem          # stable element ids / no timestamp -> clean git diffs
    fig.savefig(path_png.with_suffix(".svg"), metadata={"Date": None})
    plt.close(fig)


def _mip_raw(im: LofImage, c: int, t: int = 0) -> np.ndarray:
    return im.read_stack(c=c, t=t).max(axis=0)


def _load_ds_mip(paths: Dict[str, Path]) -> Optional[np.ndarray]:
    if paths["mip"].exists():
        return tifffile.imread(str(paths["mip"]))            # (T, C, Y, X) or squeezed
    if paths["tif"].exists():
        return tifffile.imread(str(paths["tif"])).max(axis=-4) if paths["tif"].exists() else None
    return None


def _as_tcyx(a: np.ndarray, T: int, C: int) -> np.ndarray:
    return a.reshape(T, C, a.shape[-2], a.shape[-1])


def _grid(panels: List[np.ndarray], titles: List[str], vmin, vmax, suptitle: str, out: Path, ncol: int = 4,
          cmap="gray", per_panel: bool = False):
    n = len(panels)
    ncol = min(ncol, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.4 * nrow), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, p, t in zip(axes.ravel(), panels, titles):
        if per_panel:
            lo, hi = np.percentile(p, [0.5, 99.9])
            ax.imshow(p, cmap=cmap, vmin=lo, vmax=hi)
            ax.set_title(f"{t}\n[{lo:.0f}, {hi:.0f}] (own)", fontsize=9)
        else:
            ax.imshow(p, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(t, fontsize=9)
    rng = "per-panel autoscale" if per_panel else f"common display range [{vmin:.0f}, {vmax:.0f}]"
    fig.suptitle(f"{suptitle}\n{rng}", fontsize=11)
    fig.tight_layout()
    savefig(fig, out)


def qc_dataset(entry, cfg: dict, fig_root: Path, max_files: int = 8, log=print) -> None:
    from .batch import dataset_name, index_entry
    name = dataset_name(entry)
    out_root = Path(cfg["output_root"])
    cal_path = out_root / name / "calibration.json"
    if not cal_path.exists():
        log(f"{name}: no calibration.json, skip QC")
        return
    cal = Calibration.load(cal_path)
    images = index_entry(entry, include_icc=False)
    class _D:  # keep the old variable name used below
        pass
    dataset_dir = _D(); dataset_dir.name = name
    done = [im for im in images if output_paths(out_root, im)["tif"].exists()]
    if not done:
        log(f"{dataset_dir.name}: nothing processed yet")
        return
    step = max(1, len(done) // max_files)
    sel = done[::step][:max_files]
    fdir = fig_root / dataset_dir.name
    log_df = pd.read_csv(out_root / dataset_dir.name / "ds_log.csv") if (out_root / dataset_dir.name / "ds_log.csv").exists() else None

    for key, ccal in cal.channels.items():
        raw_mips, ds_mips, titles = [], [], []
        for im in sel:
            side = json.loads(output_paths(out_root, im)["json"].read_text(encoding="utf-8"))
            cmeta = [c for c in side["channels"] if c["key"] == key]
            if not cmeta:
                continue
            c_in, c_out = cmeta[0]["c_in"], cmeta[0]["c_out"]
            mip = _load_ds_mip(output_paths(out_root, im))
            if mip is None:
                continue
            mip = _as_tcyx(mip, side["shape_out"][0], side["shape_out"][2])
            raw_mips.append(_mip_raw(im, c_in).astype(np.float32))
            ds_mips.append(mip[0, c_out].astype(np.float32))
            titles.append(im.rel)
        if not ds_mips:
            continue
        # shared DS display range: 25th percentile over files of the per-file p99.9.
        # Files with bright debris / dead cells have a p99.9 set by the blob and
        # would otherwise hide the spots of every other file; a low quantile over
        # files keeps the range at the level of the typical blob-free field.
        ds_hi = float(np.percentile([np.percentile(m, 99.9) for m in ds_mips], 25))
        _grid(raw_mips, titles, ccal.scale.lo, ccal.scale.hi, f"{dataset_dir.name}  {key}  RAW MIP",
              fdir / "unified" / f"raw_mip_{key}.png")
        _grid(ds_mips, titles, 0, ds_hi, f"{dataset_dir.name}  {key}  DS MIP ({cal.settings.get('background')}, mode={side['atmosphere_mode']}); "
              f"range = 25th pct over files of per-file p99.9",
              fdir / "unified" / f"ds_mip_{key}.png")
        _grid(raw_mips, titles, None, None, f"{dataset_dir.name}  {key}  RAW MIP", fdir / "autoscale" / f"raw_mip_{key}.png", per_panel=True)
        _grid(ds_mips, titles, None, None, f"{dataset_dir.name}  {key}  DS MIP", fdir / "autoscale" / f"ds_mip_{key}.png", per_panel=True)

        # histograms of the DS MIPs (consistency between files)
        fig, ax = plt.subplots(figsize=(7, 4))
        bins = np.linspace(0, ds_hi * 1.5, 200)
        for m, t in zip(ds_mips, titles):
            ax.hist(m.ravel(), bins=bins, histtype="step", label=t, log=True)
        ax.set_xlabel("DS MIP value (counts above lo)"); ax.set_ylabel("pixels"); ax.legend(fontsize=6)
        ax.set_title(f"{dataset_dir.name} {key}: DS MIP histograms (same scale for all files)")
        fig.tight_layout(); savefig(fig, fdir / f"ds_hist_{key}.png")

    # z-profiles from the first selected stack per channel: p99.9 per slice raw vs DS
    for im in sel[:3]:
        side = json.loads(output_paths(out_root, im)["json"].read_text(encoding="utf-8"))
        if side["shape_out"][1] < 3:
            continue
        ds = tifffile.imread(str(output_paths(out_root, im)["tif"]))
        ds = ds.reshape(side["shape_out"])
        fig, axes = plt.subplots(1, len(side["channels"]), figsize=(4.5 * len(side["channels"]), 3.6), squeeze=False)
        for ax, cm in zip(axes[0], side["channels"]):
            raw = im.read_stack(c=cm["c_in"], t=0)
            lo = cm["scale"]["lo"]
            zr = [np.percentile(raw[z], 99.9) - lo for z in range(raw.shape[0])]
            zd = [np.percentile(ds[0, z, cm["c_out"]], 99.9) for z in range(ds.shape[1])]
            ax.plot(zr, "k-o", ms=3, label="raw p99.9 - lo")
            ax.plot(zd, "r-o", ms=3, label="DS p99.9")
            ax.set_title(f"{im.rel}  {cm['key']}", fontsize=9); ax.set_xlabel("z"); ax.set_ylabel("counts"); ax.legend(fontsize=7)
        fig.tight_layout(); savefig(fig, fdir / f"zprofile_{im.rel.replace('/', '_')}.png")
    log(f"{dataset_dir.name}: QC figures -> {fdir}")


def summarize(datasets, cfg: dict, results_dir: Path, log=print) -> None:
    """results/calibration_summary.csv (one row per dataset x channel) and
    results/ds_log_all.csv (every processed stack) across all datasets."""
    out_root = Path(cfg["output_root"])
    cal_rows, logs = [], []
    from .batch import dataset_name
    for d in datasets:
        class _D:
            pass
        dn = dataset_name(d); d = _D(); d.name = dn
        cpath = out_root / d.name / "calibration.json"
        if cpath.exists():
            cal = Calibration.load(cpath)
            for k, c in cal.channels.items():
                row = dict(dataset=d.name, channel=k, lo=c.scale.lo, hi=c.scale.hi, counts_per_unit=c.scale.counts_per_unit,
                           thres=c.thres, thres_counts=c.thres_counts, fraction_below_thres=c.fraction_below_thres,
                           n_slices=c.n_slices, emission_nm=c.optics.emission_nm, NA=c.optics.NA, pixel_nm=c.optics.pixel_nm,
                           background=cal.settings.get("background"), pooling=cal.settings.get("pooling"))
                for it, (a, (al, be)) in enumerate(zip(c.atmosphere, c.alpha_beta())):
                    row.update({f"a_min_it{it}": a.a_min, f"a_max_it{it}": a.a_max, f"el_max_it{it}": a.el_max, f"el_min_it{it}": a.el_min,
                                f"alpha_it{it}": al, f"beta_it{it}": be})
                cal_rows.append(row)
        lpath = out_root / d.name / "ds_log.csv"
        if lpath.exists():
            logs.append(pd.read_csv(lpath))
    results_dir.mkdir(parents=True, exist_ok=True)
    if cal_rows:
        pd.DataFrame(cal_rows).to_csv(results_dir / "calibration_summary.csv", index=False)
    if logs:
        allog = pd.concat(logs, ignore_index=True)
        allog.to_csv(results_dir / "ds_log_all.csv", index=False)
        allog["flag_mostly_below_thres"] = allog["u_frac_below_thres"] > 0.95   # stack is almost entirely "background" for thres
        allog.to_csv(results_dir / "ds_log_all.csv", index=False)
        per = allog.groupby(["dataset", "channel"]).agg(n_stacks=("file", "size"), nz=("nz", "first"),
                                                          sec_per_stack=("seconds", "mean"),
                                                          raw_p99_9=("raw_p99_9", "median"), ds_p99_9=("ds_p99_9_counts", "median"),
                                                          frac_below_thres=("u_frac_below_thres", "median"),
                                                          frac_clipped_hi=("frac_clipped_hi", "max"),
                                                          n_mostly_below_thres=("flag_mostly_below_thres", "sum")).reset_index()
        per.to_csv(results_dir / "per_channel_summary.csv", index=False)
        log(per.round(3).to_string(index=False))
    log(f"summary tables -> {results_dir}")
