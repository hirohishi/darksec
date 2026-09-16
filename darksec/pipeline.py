"""
One .lof -> Dark-Sectioned TIFF(s), using a dataset Calibration.

Output per input file (under <output_root>/<dataset>/<rel>...):
    <rel>_DS.tif        ImageJ hyperstack TZCYX (uint16 or float32), only the
                        processed fluorescence channels, in "counts above lo"
                        (= u_out * (hi - lo) / 255, see core.Scale)
    <rel>_DS_MIP.tif    z max-projection, TCYX
    <rel>_DS.json       everything needed to interpret the numbers: channel
                        keys, scale, thres in counts, atmosphere used, params
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import tifffile

from .calibrate import Calibration, ChannelCalibration, channel_key
from .core import DEVICE_NAME, DSParams, DarkSectioner, AtmosphereStats, to_uint16
from . import core
from .leica import LofImage

IMAGEJ_LIMIT = 4 * 1024 ** 3 - 64 * 1024 ** 2   # classic TIFF limit


class SectionerCache:
    """DarkSectioner objects are expensive (filters ~3 s); reuse per (shape, optics, params)."""

    def __init__(self):
        self._d: Dict[tuple, DarkSectioner] = {}

    def get(self, shape_yx, cal: ChannelCalibration, params: DSParams) -> DarkSectioner:
        key = (tuple(shape_yx), cal.optics, params.schedule_json, params.thres, params.denoise)
        ds = self._d.get(key)
        if ds is None:
            ds = self._d[key] = DarkSectioner(shape_yx, cal.optics, params)
        ds.params = params
        return ds


def params_for(cal: ChannelCalibration, base: DSParams, overrides: Optional[dict]) -> DSParams:
    o = overrides or {}
    p = DSParams(background=o.get("background", base.background), thres=cal.thres,
                 denoise=bool(o.get("denoise", base.denoise)), atmosphere_mode=base.atmosphere_mode,
                 omega=core.omega_from_strength(o["strength"]) if "strength" in o else base.omega,
                 model=o.get("model", base.model), cutoff=o.get("cutoff", base.cutoff), nn_alpha=base.nn_alpha, nn_delta=base.nn_delta,
                 global_atmosphere=cal.atmosphere if base.atmosphere_mode == "global" else None)
    if base.atmosphere_mode == "global" and p.uses_atmosphere and len(cal.atmosphere) != p.maxtime:
        raise ValueError(f"channel {cal.key}: calibration has {len(cal.atmosphere)} iterations, "
                         f"background={p.preset_label} needs {p.maxtime}; recalibrate")
    return p


def output_paths(out_root: Path, im: LofImage) -> Dict[str, Path]:
    base = Path(out_root) / im.dataset / im.rel
    return dict(tif=base.with_name(base.name + "_DS.tif"), mip=base.with_name(base.name + "_DS_MIP.tif"),
                json=base.with_name(base.name + "_DS.json"))


def _write_stack(path: Path, arr: np.ndarray, axes: str, im: LofImage, description: str) -> None:
    """ImageJ hyperstack when it fits, OME BigTIFF otherwise."""
    path.parent.mkdir(parents=True, exist_ok=True)
    res = (1e4 / (im.pixel_nm / 1e3 * 1e0), 1e4 / (im.pixel_nm / 1e3 * 1e0))  # pixels per cm
    px_um = im.pixel_nm / 1e3
    if arr.nbytes < IMAGEJ_LIMIT:
        md = dict(axes=axes, unit="um", spacing=float(im.z_step_um or 1.0), Info=description)
        if im.t_interval_s and "T" in axes:
            md["finterval"] = float(im.t_interval_s)
        if "C" in axes and arr.shape[axes.index("C")] > 1:
            md["mode"] = "composite"
        tifffile.imwrite(str(path), arr, imagej=True, resolution=(1 / px_um, 1 / px_um), metadata=md)
    else:
        md = dict(axes=axes, PhysicalSizeX=px_um, PhysicalSizeXUnit="um", PhysicalSizeY=px_um, PhysicalSizeYUnit="um",
                  Description=description)
        if im.z_step_um and "Z" in axes:
            md.update(PhysicalSizeZ=float(im.z_step_um), PhysicalSizeZUnit="um")
        tifffile.imwrite(str(path), arr, bigtiff=True, ome=True, metadata=md)


def process_image(im: LofImage, cal: Calibration, base_params: DSParams, out_root: Path, *,
                  dtype: str = "uint16", write_mip: bool = True, exclude_keys=(), overrides: Optional[dict] = None,
                  cache: Optional[SectionerCache] = None, log=print, log_would_be_stats: bool = False) -> List[dict]:
    """Process every calibrated fluorescence channel / timepoint of one file.
    Returns one log row per (channel, timepoint)."""
    cache = cache or SectionerCache()
    paths = output_paths(out_root, im)
    chans = [(ch, channel_key(ch)) for ch in im.channels if ch.is_fluo]
    chans = [(ch, k) for ch, k in chans if k in cal.channels and k not in set(exclude_keys or ())]
    if not chans:
        log(f"  {im.rel}: no calibrated fluorescence channel, skipped")
        return []
    frames = im.frames()                      # (t, m): mosaic tiles become extra frames on the output T axis
    T, Z = len(frames), im.nZ
    H, W = im.shape_yx
    out_dtype = np.uint16 if dtype == "uint16" else np.float32
    out = np.zeros((T, Z, len(chans), H, W), dtype=out_dtype)
    mm = im.memmap()
    rows: List[dict] = []
    t_file = time.time()
    chan_meta = []
    for ci, (ch, key) in enumerate(chans):
        ccal = cal.channels[key]
        p = params_for(ccal, base_params, (overrides or {}).get(key))
        ds = cache.get((H, W), ccal, p)
        chan_meta.append(dict(c_in=ch.index, c_out=ci, key=key, lut=ch.lut, exposure_s=ch.exposure_s,
                              scale=asdict(ccal.scale), counts_per_unit=ccal.scale.counts_per_unit,
                              thres=ccal.thres, thres_counts=ccal.thres_counts, background=p.preset_label, schedule=p.preset, omega=p.omega, strength=p.strength,
                              model=p.model, cutoff=p.cutoff, cutoff_cycles_per_um=ds.filters.cutoff_cycles_per_um,
                              nn_alpha=p.nn_alpha if p.model == "neighbour" else None, nn_delta=p.nn_delta if p.model == "neighbour" else None,
                              optics=asdict(ccal.optics),
                              denoise=p.denoise, emission_nm=ccal.optics.emission_nm,
                              block_size=ds.filters.per_iter[0].block_size, padded_shape=list(ds.padding.shape_pad),
                              fft_backend=ds.fft.backend, cufft_version=core.cufft_version(),
                              atmosphere_global=[asdict(a) for a in ccal.atmosphere]))
        for fi, (t_in, m_in) in enumerate(frames):
            t = fi
            t0 = time.time()
            raw = im.read_stack(c=ch.index, t=t_in, mm=mm, m=m_in)
            u = ccal.scale.to_working(raw)
            res_u, stats = ds.run(u, return_stats=True, collect_would_be_stats=log_would_be_stats)
            counts = ccal.scale.to_counts(res_u)
            out[t, :, ci] = to_uint16(counts) if out_dtype == np.uint16 else counts.astype(np.float32)
            dt = time.time() - t0
            r = dict(dataset=im.dataset, file=im.rel, channel=key, c=ch.index, t=t, t_in=t_in, m=m_in, nz=Z,
                     raw_min=int(raw.min()), raw_p50=float(np.median(raw[Z // 2])), raw_p99_9=float(np.percentile(raw, 99.9)),
                     raw_max=int(raw.max()), n_saturated=int((raw >= 65535).sum()),
                     u_frac_above_255=float((u > 255).mean()), u_frac_below_thres=float((u < ccal.thres).mean()),
                     ds_min_counts=float(counts.min()), ds_p50_counts=float(np.median(counts)),
                     ds_p99_9_counts=float(np.percentile(counts, 99.9)), ds_max_counts=float(counts.max()),
                     frac_clipped_neg=float((counts < 0).mean()), frac_clipped_hi=float((counts > 65535).mean()) if out_dtype == np.uint16 else 0.0,
                     flag_mostly_below_thres=bool((u < ccal.thres).mean() > 0.95),
                     seconds=dt, mode=p.atmosphere_mode)
            for it in range(p.maxtime):
                r[f"a_min_it{it}_med"] = float(np.median([s[it].a_min for s in stats]))
                r[f"a_max_it{it}_med"] = float(np.median([s[it].a_max for s in stats]))
                r[f"el_max_it{it}_med"] = float(np.median([s[it].el_max for s in stats]))
            rows.append(r)
            log(f"  {im.rel} {key} t{t}: {Z} z in {dt:.1f}s  raw p99.9={r['raw_p99_9']:.0f}  DS p99.9={r['ds_p99_9_counts']:.1f} counts")
    desc = json.dumps(dict(software="darksec 1.0", mode=base_params.atmosphere_mode, channels=chan_meta,
                           units="counts above lo (u_out*(hi-lo)/255)"), ensure_ascii=False)
    _write_stack(paths["tif"], out, "TZCYX", im, desc)
    if write_mip:
        _write_stack(paths["mip"], out.max(axis=1), "TCYX", im, desc)
    side = dict(input=str(im.path), dataset=im.dataset, rel=im.rel, dims_in=list(im.dims), shape_in=list(im.shape),
                axes_out="TZCYX", frames_tm=frames, shape_out=list(out.shape), dtype=dtype, pixel_nm=im.pixel_nm, z_step_um=im.z_step_um,
                t_interval_s=im.t_interval_s, NA=im.NA, objective=im.objective, device=DEVICE_NAME,
                atmosphere_mode=base_params.atmosphere_mode, channels=chan_meta, per_stack_log=rows,
                seconds_total=time.time() - t_file, created=time.strftime("%Y-%m-%d %H:%M:%S"))
    paths["json"].write_text(json.dumps(side, indent=2, ensure_ascii=False), encoding="utf-8")
    return rows
