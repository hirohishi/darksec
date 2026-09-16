"""Config loading and dataset-level orchestration (calibrate -> process -> log)."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml

from .calibrate import Calibration, calibrate
from .core import DSParams
from .leica import LofImage, index_dataset, describe
from .pipeline import SectionerCache, output_paths, process_image
from .tiffio import index_tiffs


def dataset_name(entry) -> str:
    if isinstance(entry, dict):
        p = Path(entry["path"])
        return entry.get("name") or (p.stem if p.is_file() else p.name)
    return Path(entry).name


def index_entry(entry, include_icc: bool = False) -> List[LofImage]:
    """A config `datasets` entry is either a Leica project folder (str) or a dict:
    {path: <Leica project folder>, include: [substrings of the relative path]} restricts BOTH the
    calibration and the processing to the matching files (e.g. include: ["TileScan 9", "TileScan 10"]);
    {path: ..., type: tiff, ...} indexes TIFF files (see tiffio)."""
    if isinstance(entry, dict):
        if entry.get("type", "leica") == "tiff":
            return index_tiffs(entry)
        ims = index_dataset(Path(entry["path"]), include_icc=include_icc)
        inc = entry.get("include")
        if inc:
            ims = [im for im in ims if any(f in im.rel for f in inc)]
        return ims
    return index_dataset(Path(entry), include_icc=include_icc)


def load_config(path: Path) -> dict:
    from .paths import expand
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg["datasets"] = [expand(d) if isinstance(d, str) else {**d, "path": expand(d["path"])} for d in cfg.get("datasets", [])]
    if cfg.get("output_root"):
        cfg["output_root"] = expand(cfg["output_root"])
    if cfg.get("calibration", {}).get("from_file"):
        cfg["calibration"]["from_file"] = expand(cfg["calibration"]["from_file"])
    cfg.setdefault("channels", {}).setdefault("exclude", [])
    cfg["channels"].setdefault("overrides", {})
    return cfg


def base_params(cfg: dict) -> DSParams:
    from .core import omega_from_strength, STRENGTH_DEFAULT
    p = cfg["processing"]
    nn = p.get("neighbour", {}) or {}
    return DSParams(background=p.get("background", "severe"), thres=float(p.get("thres", 70)),
                    denoise=bool(p.get("denoise", True)), atmosphere_mode=p.get("atmosphere_mode", "global"),
                    omega=omega_from_strength(p.get("strength", STRENGTH_DEFAULT)),
                    model=p.get("model", "multiplicative"), cutoff=p.get("cutoff", "legacy"),
                    nn_alpha=float(nn.get("alpha", 0.45)), nn_delta=int(nn.get("delta", 1)))


def calibration_path(out_root: Path, name: str) -> Path:
    return Path(out_root) / name / "calibration.json"


def get_calibration(images: List[LofImage], cfg: dict, out_root: Path, name: str, *, force=False, log=print) -> Calibration:
    c = cfg.get("calibration", {})
    if c.get("from_file"):
        cal = Calibration.load(Path(c["from_file"]))
        log(f"calibration loaded from {c['from_file']} (dataset {cal.dataset})")
        return cal
    cpath = calibration_path(out_root, name)
    params = base_params(cfg)
    if cpath.exists() and c.get("reuse", True) and not force:
        cal = Calibration.load(cpath)
        key = cal.settings.get("schedule_key")
        if key is not None:
            same_preset = key == params.schedule_json and all(
                (len(ch.atmosphere) == params.maxtime) or not params.uses_atmosphere for ch in cal.channels.values())
        else:   # calibrations written before schedule_key existed
            stored = cal.settings.get("schedule")
            same_preset = ((stored == params.preset) if stored is not None else (cal.settings.get("background") == params.preset_label)) and all(
                len(ch.atmosphere) == params.maxtime for ch in cal.channels.values()) and params.model == "multiplicative" and params.cutoff == "legacy"
        same_thres = float(cal.settings.get("thres", params.thres)) == params.thres and float(cal.settings.get("omega", 0.95)) == params.omega
        from .core import ALGORITHM_VERSION
        same_algo = cal.settings.get("algorithm_version") == ALGORITHM_VERSION
        if same_preset and same_thres and same_algo:
            log(f"calibration reused: {cpath}")
            return cal
        log(f"calibration at {cpath} was made for background={cal.settings.get('background')}, "
            f"thres={cal.settings.get('thres')}, algorithm {cal.settings.get('algorithm_version', '1.0')}; "
            f"strength={100*float(cal.settings.get('omega', 0.95)):g}, model={cal.settings.get('model', 'multiplicative')}, cutoff={cal.settings.get('cutoff', 'legacy')}; "
            f"recalibrating for {params.preset_label} {params.preset['deg']}, thres={params.thres}, strength={params.strength:g}, model={params.model}, cutoff={params.cutoff}, "
            f"algorithm {ALGORITHM_VERSION}")
    opt = cfg.get("optics", {})
    cal = calibrate(images, params, lo_percentile=float(c.get("lo_percentile", 0.1)),
                    hi_percentile=float(c.get("hi_percentile", 99.9)), lo_fixed=c.get("lo_fixed"), hi_fixed=c.get("hi_fixed"),
                    n_z=int(c.get("n_z", 5)), n_t=int(c.get("n_t", 2)), factor=float(opt.get("factor", 2)),
                    emission_table=opt.get("emission_nm"), NA_override=opt.get("NA"),
                    atmosphere_rounds=int(c.get("atmosphere_rounds", 2)),
                    include_keys=None, per_channel_params=cfg["channels"].get("overrides"),
                    pooling=c.get("pooling", "median_of_slices"),
                    log=log, slice_table_path=cpath.with_name("calibration_slices.csv"))
    cal.save(cpath)
    log(f"calibration saved: {cpath}\n" + cal.summary())
    return cal


def run_dataset(entry, cfg: dict, *, file_filter: Optional[List[str]] = None, limit: Optional[int] = None,
                overwrite: bool = False, recalibrate: bool = False, mode_override: Optional[str] = None,
                log=print) -> pd.DataFrame:
    name = dataset_name(entry)
    out_root = Path(cfg["output_root"])
    images = index_entry(entry, include_icc=bool(cfg.get("include_icc", False)))
    if not images:
        log(f"{name}: no image files found")
        return pd.DataFrame()
    log(f"=== {name}: {len(images)} image files\n" + describe(images))
    cal = get_calibration(images, cfg, out_root, name, force=recalibrate, log=log)
    params = base_params(cfg)
    if mode_override:
        params.atmosphere_mode = mode_override
    todo = images
    if file_filter:
        todo = [im for im in todo if any(f in im.rel for f in file_filter)]
    if limit:
        todo = todo[:limit]
    cache = SectionerCache()
    rows: List[dict] = []
    t0 = time.time()
    for i, im in enumerate(todo):
        paths = output_paths(out_root, im)
        if paths["tif"].exists() and cfg.get("output", {}).get("skip_existing", True) and not overwrite:
            log(f"[{i+1}/{len(todo)}] {im.rel}: exists, skipped")
            continue
        log(f"[{i+1}/{len(todo)}] {im.rel}")
        rows += process_image(im, cal, params, out_root, dtype=cfg.get("output", {}).get("dtype", "uint16"),
                              write_mip=bool(cfg.get("output", {}).get("write_mip", True)),
                              exclude_keys=cfg["channels"].get("exclude", []), overrides=cfg["channels"].get("overrides"),
                              cache=cache, log=log)
    df = pd.DataFrame(rows)
    if not df.empty:
        log_path = out_root / name / "ds_log.csv"
        if log_path.exists() and not overwrite:
            old = pd.read_csv(log_path)
            keys = ["file", "channel", "t"]
            old = old[~old.set_index(keys).index.isin(df.set_index(keys).index)]
            df = pd.concat([old, df], ignore_index=True)
        df.to_csv(log_path, index=False)
        log(f"{name}: {len(rows)} stacks in {time.time()-t0:.0f}s -> {log_path}")
    return df
