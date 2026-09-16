"""
Global calibration of the Dark Sectioning intensity scale and atmosphere.

Why
---
The legacy code min-max normalised every stack to [0, 255] before dehazing,
so `thres` and the estimated "atmosphere" A(x) meant a different absolute
intensity in every image (and one hot pixel could squeeze a whole stack
into a few grey levels). Here every channel of a dataset gets ONE linear
map raw counts -> [0, 255] (`Scale`), and optionally ONE set of atmosphere
statistics per iteration, so the same transform is applied to every slice,
z-stack, timepoint and field.

Channel identity is the LED line name from the .xlif ('635nm', '470nm', ...),
so 'C0' of one file and 'C1' of another are pooled when they are the same
LED.

Outputs
-------
Calibration (JSON): per channel key
    scale.lo / scale.hi          raw counts mapped to 0 / 255
    atmosphere[it] = {a_min, a_max, el_max}
        equivalently A(x) = dep * (alpha * EL(x) + beta) with
        alpha = (a_max - a_min) / el_max,  beta = a_min
    thres_counts                 `thres` expressed in raw counts
    sample statistics            n slices, percentiles, fraction below thres
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .core import ALGORITHM_VERSION, AtmosphereStats, DSParams, DarkSectioner, Optics, Scale
from .leica import LofImage, ChannelInfo

# LED line (nm) -> emission wavelength used for the optics (nm).
# LAS X stores EmissionWavelength=0, so this is a lookup for the common dyes
# imaged with the DFT51010 quad-band cube (DAPI / FITC / TRITC / Cy5) and the
# CYR71010 cube (Cy3 / Cy5 / Cy7).
DEFAULT_EMISSION_NM: Dict[str, float] = {
    "400nm": 460, "405nm": 460, "DAPI": 460,
    "470nm": 520, "488nm": 520,
    "542nm": 590, "555nm": 590,
    "588nm": 620,
    "635nm": 670, "640nm": 670,
    "740nm": 780,
}
LUT_TO_LED: Dict[str, str] = {"Blue": "400nm", "Green": "470nm", "Red": "542nm",
                              "Yellow": "588nm", "Magenta": "635nm"}


def channel_key(ch: ChannelInfo) -> str:
    """Stable identity of a channel across files: LED name, LUT as fallback."""
    if ch.led_nm is not None:
        return f"{ch.led_nm}nm"
    if ch.led_name and ch.led_name != f"C{ch.index}":
        return ch.led_name
    return LUT_TO_LED.get(ch.lut, f"C{ch.index}")


def emission_for(key: str, table: Optional[Dict[str, float]] = None) -> float:
    table = {**DEFAULT_EMISSION_NM, **(table or {})}
    if key in table:
        return float(table[key])
    raise KeyError(f"no emission wavelength for channel {key!r}; add it to optics.emission_nm in config.yaml")


# ----------------------------------------------------------------------------
@dataclass
class SampleSlice:
    image: LofImage
    c: int
    t: int
    z: int
    key: str
    m: int = 0


def sample_plan(images: Sequence[LofImage], n_z: int = 5, n_t: int = 2,
                include_keys: Optional[Sequence[str]] = None) -> List[SampleSlice]:
    """Evenly spaced z (and t) slices of every fluorescence channel of every file."""
    plan: List[SampleSlice] = []
    for im in images:
        zs = np.unique(np.linspace(0, im.nZ - 1, min(n_z, im.nZ)).round().astype(int))
        frames = im.frames()
        fsel = np.unique(np.linspace(0, len(frames) - 1, min(n_t, len(frames))).round().astype(int))
        for ch in im.channels:
            if not ch.is_fluo:
                continue
            key = channel_key(ch)
            if include_keys and key not in include_keys:
                continue
            for fi in fsel:
                t, m = frames[int(fi)]
                for z in zs:
                    plan.append(SampleSlice(im, ch.index, int(t), int(z), key, m=int(m)))
    return plan


def _read_slice(s: SampleSlice, mm_cache: dict) -> np.ndarray:
    mm = mm_cache.get(s.image.path)
    if mm is None:
        mm = mm_cache[s.image.path] = s.image.memmap()
    idx = []
    for d in s.image.dims:
        idx.append({"T": s.t, "C": s.c, "Z": s.z, "M": s.m}.get(d, slice(None)))
    return np.asarray(mm[tuple(idx)])


# ----------------------------------------------------------------------------
@dataclass
class ChannelCalibration:
    key: str
    scale: Scale
    optics: Optics
    n_slices: int
    percentiles: Dict[str, float]          # pooled raw percentiles
    atmosphere: List[AtmosphereStats] = field(default_factory=list)
    thres: float = 70.0
    fraction_below_thres: Optional[float] = None
    fraction_above_255: Optional[float] = None

    @property
    def thres_counts(self) -> float:
        return self.scale.thres_counts(self.thres)

    def alpha_beta(self) -> List[Tuple[float, float]]:
        """A(x) = dep * (alpha * EL(x) + beta) per iteration."""
        return [a.alpha_beta() for a in self.atmosphere]

    def to_dict(self) -> dict:
        d = dict(key=self.key, scale=asdict(self.scale), optics=asdict(self.optics), n_slices=self.n_slices,
                 percentiles=self.percentiles, thres=self.thres, thres_counts=self.thres_counts,
                 fraction_below_thres=self.fraction_below_thres, fraction_above_255=self.fraction_above_255,
                 atmosphere=[asdict(a) for a in self.atmosphere],
                 atmosphere_alpha_beta=[dict(alpha=a, beta=b) for a, b in self.alpha_beta()])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ChannelCalibration":
        return cls(key=d["key"], scale=Scale(**d["scale"]), optics=Optics(**d["optics"]), n_slices=d["n_slices"],
                   percentiles=d["percentiles"], atmosphere=[AtmosphereStats(**a) for a in d.get("atmosphere", [])],
                   thres=d.get("thres", 70.0), fraction_below_thres=d.get("fraction_below_thres"),
                   fraction_above_255=d.get("fraction_above_255"))


@dataclass
class Calibration:
    dataset: str
    created: str
    settings: dict
    channels: Dict[str, ChannelCalibration]

    def save(self, path: Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(dataset=self.dataset, created=self.created, settings=self.settings,
                                        channels={k: v.to_dict() for k, v in self.channels.items()}),
                                   indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Calibration":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(dataset=d["dataset"], created=d["created"], settings=d.get("settings", {}),
                   channels={k: ChannelCalibration.from_dict(v) for k, v in d["channels"].items()})

    def summary(self) -> str:
        rows = []
        for k, c in self.channels.items():
            ab = c.alpha_beta()
            rows.append(f"  {k:8s} lo={c.scale.lo:7.1f} hi={c.scale.hi:8.1f} (1 unit = {c.scale.counts_per_unit:.2f} counts)"
                        f"  thres={c.thres:.0f} -> {c.thres_counts:.0f} counts  below-thres {100*(c.fraction_below_thres or 0):.1f}%"
                        f"  n={c.n_slices} slices  em={c.optics.emission_nm:.0f}nm NA={c.optics.NA}")
            for it, ((al, be), a) in enumerate(zip(ab, c.atmosphere)):
                rows.append(f"           iter{it}: a_min={a.a_min:6.2f} a_max={a.a_max:7.2f} el=[{a.el_min:.2f}, {a.el_max:.2f}]"
                            f"   A(x) = dep*({al:.3f}*EL + {be:.2f})")
        return "\n".join(rows)


# ----------------------------------------------------------------------------
def _percentiles_from_hist(hist: np.ndarray, pcts: Sequence[float]) -> List[float]:
    cdf = np.cumsum(hist, dtype=np.float64)
    total = cdf[-1]
    out = []
    for p in pcts:
        k = int(np.searchsorted(cdf, p / 100.0 * total, side="left"))
        out.append(float(min(k, hist.size - 1)))
    return out


def calibrate(images: Sequence[LofImage], params: DSParams, *, lo_percentile: float = 0.1,
              hi_percentile: float = 99.9, lo_fixed: Optional[float] = None, hi_fixed: Optional[float] = None,
              n_z: int = 5, n_t: int = 2, factor: float = 2.0, emission_table: Optional[Dict[str, float]] = None,
              NA_override: Optional[float] = None, atmosphere_rounds: int = 2, include_keys=None,
              per_channel_params: Optional[Dict[str, dict]] = None, log=print,
              slice_table_path: Optional[Path] = None, pooling: str = "median_of_slices") -> Calibration:
    """Two-stage calibration.

    Stage 1  lo / hi per channel (Scale):
             pooling='median_of_slices' (default): the percentile is taken in
             every sampled slice and the MEDIAN over slices is used, so one
             field with a bright debris blob cannot set the scale for the
             whole dataset.
             pooling='pooled': percentile of the pooled 16-bit histogram
             (dominated by the brightest field when fields differ a lot).
    Stage 2  run the sampled slices through DS in per_slice mode with that Scale
             and take the median a_min / a_max / el_max per iteration. Repeat
             `atmosphere_rounds` times with the medians fixed (the inputs of
             iteration k depend on the statistics used in iterations < k).
    """
    t0 = time.time()
    plan = sample_plan(images, n_z=n_z, n_t=n_t, include_keys=include_keys)
    keys = sorted({s.key for s in plan})
    for k in list(keys):
        try:
            emission_for(k, emission_table)
        except KeyError as e:
            log(f"  WARNING: {e}; channel {k!r} skipped")
            keys.remove(k); plan = [s for s in plan if s.key != k]
    log(f"calibration: {len(images)} files, {len(plan)} sampled slices, channels {keys}")
    mm_cache: dict = {}
    hists = {k: np.zeros(65536, dtype=np.int64) for k in keys}
    slice_rows = []
    per_slice_lo = {k: [] for k in keys}
    per_slice_hi = {k: [] for k in keys}
    for s in plan:
        a = _read_slice(s, mm_cache)
        hists[s.key] += np.bincount(a.ravel(), minlength=65536)[:65536]
        plo, phi, p999 = np.percentile(a, [lo_percentile, hi_percentile, 99.9])
        per_slice_lo[s.key].append(float(plo)); per_slice_hi[s.key].append(float(phi))
        slice_rows.append(dict(dataset=s.image.dataset, file=s.image.rel, channel=s.key, c=s.c, t=s.t, m=s.m, z=s.z,
                               raw_min=int(a.min()), raw_p50=float(np.median(a)), raw_p99_9=float(p999),
                               raw_max=int(a.max()), n_saturated=int((a >= 65535).sum()),
                               slice_lo=float(plo), slice_hi=float(phi)))
    log(f"  histograms done ({time.time()-t0:.0f}s)")

    chans: Dict[str, ChannelCalibration] = {}
    for k in keys:
        pcts = [0.01, 0.1, 1, 50, 99, 99.9, 99.99]
        vals = _percentiles_from_hist(hists[k], pcts)
        pdict = {f"p{p}": v for p, v in zip(pcts, vals)}
        if pooling == "pooled":
            lo_auto = _percentiles_from_hist(hists[k], [lo_percentile])[0]
            hi_auto = _percentiles_from_hist(hists[k], [hi_percentile])[0]
        elif pooling == "median_of_slices":
            lo_auto = float(np.median(per_slice_lo[k]))
            hi_auto = float(np.median(per_slice_hi[k]))
        else:
            raise ValueError(f"unknown pooling {pooling!r}")
        lo = float(lo_fixed) if lo_fixed is not None else lo_auto
        hi = float(hi_fixed) if hi_fixed is not None else hi_auto
        if hi <= lo + 1:
            hi = lo + 1.0
        # optics: pixel size / NA from the first file that has this channel
        im0 = next(s.image for s in plan if s.key == k)
        NA = NA_override if NA_override is not None else (im0.NA or 1.4)
        from .core import immersion_index
        optics = Optics(NA=float(NA), emission_nm=emission_for(k, emission_table), pixel_nm=float(im0.pixel_nm), factor=factor,
                        z_step_um=(float(im0.z_step_um) if im0.z_step_um else None), n_immersion=immersion_index(getattr(im0, "objective", ""), float(NA)))
        thres = float((per_channel_params or {}).get(k, {}).get("thres", params.thres))
        chans[k] = ChannelCalibration(key=k, scale=Scale(lo=lo, hi=hi), optics=optics,
                                      n_slices=sum(1 for s in plan if s.key == k), percentiles=pdict, thres=thres)
        below = hists[k][: int(round(lo + thres * (hi - lo) / 255.0)) + 1].sum() / hists[k].sum()
        above = hists[k][int(round(hi)):].sum() / hists[k].sum()
        chans[k].fraction_below_thres = float(below)
        chans[k].fraction_above_255 = float(above)
        hi_spread = (max(per_slice_hi[k]) / max(hi, 1.0))
        if below > 0.95:
            log(f"  WARNING {k}: {100*below:.1f} % of all sampled pixels are below thres ({lo + thres*(hi-lo)/255:.0f} counts). "
                f"thres is probably too high for this channel, or hi is dominated by outliers (brightest slice p{hi_percentile} = "
                f"{max(per_slice_hi[k]):.0f} vs used hi {hi:.0f}). Consider a lower thres or hi_fixed.")
        elif hi_spread > 3:
            log(f"  note {k}: the brightest sampled slice reaches {hi_spread:.1f}x the calibration hi; DS values of such "
                f"extended bright objects will be strongly amplified (and clipped in uint16).")

    # ---- stage 2: atmosphere statistics --------------------------------------
    for k in keys:
        cal = chans[k]
        o = (per_channel_params or {}).get(k, {})
        from .core import omega_from_strength
        p = DSParams(background=o.get("background", params.background), thres=cal.thres, denoise=params.denoise, atmosphere_mode="per_slice",
                     omega=omega_from_strength(o["strength"]) if "strength" in o else params.omega,
                     model=o.get("model", params.model), cutoff=o.get("cutoff", params.cutoff), nn_alpha=params.nn_alpha, nn_delta=params.nn_delta)
        if not p.uses_atmosphere:
            log(f"  {k}: model={p.model} uses no atmosphere statistics; stage 2 skipped")
            cal.atmosphere = []
            continue
        ss = [s for s in plan if s.key == k]
        shapes = {s.image.shape_yx for s in ss}
        if len(shapes) != 1:
            raise ValueError(f"channel {k}: mixed image sizes {shapes} cannot share one calibration")
        ds = DarkSectioner(next(iter(shapes)), cal.optics, p)
        u = np.stack([cal.scale.to_working(_read_slice(s, mm_cache)) for s in ss]).astype(np.float32)
        current: Optional[List[AtmosphereStats]] = None
        for rnd in range(max(1, atmosphere_rounds)):
            if current is None:
                _, stats = ds.run_with_mode(u, "per_slice")
            else:
                ds.params.global_atmosphere = current
                _, stats = ds.run_with_mode(u, "global", collect_would_be_stats=True)
            new = []
            for it in range(ds.params.maxtime):
                new.append(AtmosphereStats(
                    a_min=float(np.median([st[it].a_min for st in stats])),
                    a_max=float(np.median([st[it].a_max for st in stats])),
                    el_max=float(np.median([st[it].el_max for st in stats])),
                    el_min=float(np.median([st[it].el_min for st in stats]))))
            for j, s in enumerate(ss):   # keep the last round's per-slice values for QC
                for it in range(ds.params.maxtime):
                    slice_rows_idx = next(i for i, r in enumerate(slice_rows)
                                          if r["file"] == s.image.rel and r["channel"] == k and r["c"] == s.c and r["t"] == s.t and r["z"] == s.z and r.get("m", 0) == s.m)
                    slice_rows[slice_rows_idx][f"a_min_it{it}"] = stats[j][it].a_min
                    slice_rows[slice_rows_idx][f"a_max_it{it}"] = stats[j][it].a_max
                    slice_rows[slice_rows_idx][f"el_max_it{it}"] = stats[j][it].el_max
                    slice_rows[slice_rows_idx][f"el_min_it{it}"] = stats[j][it].el_min
            current = new
        ds.params.global_atmosphere = None
        cal.atmosphere = current
        log(f"  {k}: atmosphere calibrated on {len(ss)} slices ({time.time()-t0:.0f}s)")

    if slice_table_path is not None:
        Path(slice_table_path).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(slice_rows).to_csv(slice_table_path, index=False)

    settings = dict(algorithm_version=ALGORITHM_VERSION, lo_percentile=lo_percentile, hi_percentile=hi_percentile, lo_fixed=lo_fixed, hi_fixed=hi_fixed, pooling=pooling,
                    n_z=n_z, n_t=n_t, factor=factor, background=params.preset_label, schedule=params.preset, schedule_key=params.schedule_json,
                    omega=params.omega, strength=params.strength, model=params.model, cutoff=params.cutoff, nn_alpha=params.nn_alpha, nn_delta=params.nn_delta, thres=params.thres,
                    denoise=params.denoise, atmosphere_rounds=atmosphere_rounds, per_channel_params=per_channel_params or {})
    return Calibration(dataset=images[0].dataset if images else "", created=time.strftime("%Y-%m-%d %H:%M:%S"),
                       settings=settings, channels=chans)
