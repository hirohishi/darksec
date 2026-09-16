"""
Leica LAS X project (.xlef / .lof / .xlif) access without loading whole files.

A Leica project folder looks like

    <project>/Project001.xlef
    <project>/TileScan 2/<row>/<col>/P1.lof          (image data)
    <project>/TileScan 2/<row>/<col>/P1_ICC.lof      (Instant Computational Clearing copy)
    <project>/TileScan 2/<row>/<col>/Metadata/P1.xlif (per-image XML)
    <project>/Image 2.lof + <project>/Metadata/Image 2.xlif
    <project>/Pyramid/*, <project>/Additional Data/*  (viewer pyramids, frame properties: ignored)

`index_dataset()` returns one `LofImage` per image .lof (ICC copies excluded by
default) with the axis order, shape, pixel size, z step, objective NA and the
per-channel LED line taken from the sidecar .xlif.

Channel identity: LAS X does not store the emission wavelength (it is 0 in the
XML). The `WideFieldChannelInfo` entries carry the user-defined LED name
('635nm', '470nm', '400nm', ...); their order matches the C axis. Fluorescence
entries (ContrastingMethodName == 'FLUO') are kept, BF / AutoFocus entries and
the trailing 'New 1' template are dropped.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_LED_RE = re.compile(r"(\d{3})\s*nm", re.I)


@dataclass
class ChannelInfo:
    index: int
    led_name: str          # e.g. '635nm' (as typed by the user, '*' stripped)
    led_nm: Optional[int]  # 635
    lut: str               # 'Magenta'
    exposure_s: Optional[float] = None
    intensity: Optional[float] = None
    is_fluo: bool = True


@dataclass
class LofImage:
    path: Path
    xlif: Optional[Path]
    dims: Tuple[str, ...]          # e.g. ('C','Z','Y','X')
    shape: Tuple[int, ...]
    dtype: str
    offset: int
    pixel_nm: float
    z_step_um: Optional[float]
    t_interval_s: Optional[float]
    NA: Optional[float]
    objective: str
    camera: str
    channels: List[ChannelInfo]
    dataset: str                   # project folder name
    rel: str                       # path relative to the project folder, without suffix

    # -- axes helpers ---------------------------------------------------------
    @property
    def nT(self) -> int: return self.shape[self.dims.index("T")] if "T" in self.dims else 1
    @property
    def nM(self) -> int: return self.shape[self.dims.index("M")] if "M" in self.dims else 1

    def frames(self):
        """(t, m) index pairs; a mosaic tile (M axis) is treated as an extra frame."""
        return [(t, m) for t in range(self.nT) for m in range(self.nM)]
    @property
    def nC(self) -> int: return self.shape[self.dims.index("C")] if "C" in self.dims else 1
    @property
    def nZ(self) -> int: return self.shape[self.dims.index("Z")] if "Z" in self.dims else 1
    @property
    def shape_yx(self) -> Tuple[int, int]:
        return self.shape[self.dims.index("Y")], self.shape[self.dims.index("X")]

    def memmap(self) -> np.memmap:
        return np.memmap(str(self.path), dtype=np.dtype(self.dtype), mode="r",
                         offset=self.offset, shape=self.shape)

    def read_stack(self, c: int = 0, t: int = 0, mm: Optional[np.memmap] = None, m: int = 0) -> np.ndarray:
        """(Z, Y, X) for one channel / timepoint / mosaic tile; a 2-D image comes back as (1, Y, X)."""
        mm = self.memmap() if mm is None else mm
        idx = []
        for d in self.dims:
            if d == "T": idx.append(t)
            elif d == "C": idx.append(c)
            elif d == "M": idx.append(m)
            else: idx.append(slice(None))
        arr = np.asarray(mm[tuple(idx)])
        if "Z" not in self.dims:
            arr = arr[None]
        return arr

    def to_record(self) -> dict:
        d = asdict(self)
        d["path"] = str(self.path); d["xlif"] = str(self.xlif) if self.xlif else None
        d["channels"] = [asdict(c) for c in self.channels]
        return d


# ----------------------------------------------------------------------------
def _attrs(tag: str) -> Dict[str, str]:
    return dict(re.findall(r'(\w+)="([^"]*)"', tag))


def parse_xlif(xlif: Path) -> dict:
    """Pull what we need from the per-image XML with regexes (the files are
    <100 kB; a full XML parse is not worth the namespace trouble)."""
    s = xlif.read_text(encoding="utf-8", errors="ignore")
    out: dict = {}
    dims = {}
    for m in re.finditer(r"<DimensionDescription[^>]*>", s):
        a = _attrs(m.group(0))
        dims[int(a["DimID"])] = (int(a["NumberOfElements"]), float(a["Length"]))
    # DimID 1 = X, 2 = Y, 3 = Z, 4 = T (Leica convention); Length is the full extent in m (Z, X, Y) or s (T)
    if 1 in dims:
        n, L = dims[1]
        out["pixel_nm"] = L / n * 1e9 if n > 0 else None
    if 3 in dims and dims[3][0] > 1:
        n, L = dims[3]
        out["z_step_um"] = L / (n - 1) * 1e6
    if 4 in dims and dims[4][0] > 1:
        n, L = dims[4]
        out["t_interval_s"] = L / (n - 1)
    m = re.search(r'NumericalAperture="([^"]*)"', s)
    out["NA"] = float(m.group(1)) if m else None
    m = re.search(r'ObjectiveName="([^"]*)"', s)
    out["objective"] = m.group(1).strip() if m else ""
    m = re.search(r'CameraName="([^"]*)"', s)
    out["camera"] = m.group(1) if m else ""
    luts = [_attrs(t).get("LUTName", "") for t in re.findall(r"<ChannelDescription[^>]*>", s)]
    wf = [_attrs(t) for t in re.findall(r"<WideFieldChannelInfo[^>]*/?>", s)]
    # Acquisition channels are the entries with Channel="2xxx" in order
    # (fluorescence AND brightfield interleaved as acquired); "New 1" is an
    # unused template and Channel="3xxx" is the autofocus camera.
    acq = [w for w in wf if w.get("UserDefName", "") != "New 1" and w.get("Channel", "2").startswith("2")]
    if len(acq) < len(luts):          # metadata lists fewer entries than channels: fall back to FLUO-first order
        acq = [w for w in wf if w.get("UserDefName", "") != "New 1"]
    chans: List[ChannelInfo] = []
    for i, lut in enumerate(luts):
        w = acq[i] if i < len(acq) else {}
        name = w.get("UserDefName", "").replace("*", "").strip()
        mm_ = _LED_RE.search(name)
        method = w.get("ContrastingMethodName")
        if method is None:            # no entry: infer from the LUT (Gray is what LAS X gives brightfield)
            is_fluo = lut not in ("Gray", "Grey", "")
        else:
            is_fluo = method == "FLUO"
        chans.append(ChannelInfo(
            index=i, led_name=name or f"C{i}", led_nm=int(mm_.group(1)) if mm_ else None, lut=lut,
            exposure_s=float(w["ExposureTime"]) if w.get("ExposureTime") else None,
            intensity=float(w["Intensity"]) if w.get("Intensity") else None,
            is_fluo=is_fluo,
        ))
    out["channels"] = chans
    return out


def _is_image_lof(p: Path, include_icc: bool) -> bool:
    parts = set(p.parts)
    if "Pyramid" in parts or "Additional Data" in parts:
        return False
    if p.stem.endswith("_ICC") and not include_icc:
        return False
    if "FrameProperties" in p.stem or p.stem.endswith("_histo") or re.search(r"_pmd_\d+$", p.stem):
        return False
    return True


def index_dataset(project_dir: Path, include_icc: bool = False) -> List[LofImage]:
    """All image .lof files in a Leica project folder, with metadata."""
    from liffile import LifFile
    project_dir = Path(project_dir)
    images: List[LofImage] = []
    for lof in sorted(project_dir.rglob("*.lof")):
        if not _is_image_lof(lof, include_icc):
            continue
        xlif = lof.parent / "Metadata" / (lof.stem + ".xlif")
        meta = parse_xlif(xlif) if xlif.exists() else {}
        with LifFile(str(lof)) as f:
            im = f.images[0]
            dims, shape, dtype, offset = tuple(im.dims), tuple(im.shape), str(np.dtype(im.dtype)), int(im.memory_block.offset)
        nC = shape[dims.index("C")] if "C" in dims else 1
        chans = meta.get("channels") or []
        if len(chans) < nC:   # metadata incomplete: pad with anonymous channels
            chans = chans + [ChannelInfo(index=i, led_name=f"C{i}", led_nm=None, lut="") for i in range(len(chans), nC)]
        images.append(LofImage(
            path=lof, xlif=xlif if xlif.exists() else None, dims=dims, shape=shape, dtype=dtype, offset=offset,
            pixel_nm=meta.get("pixel_nm") or float("nan"), z_step_um=meta.get("z_step_um"),
            t_interval_s=meta.get("t_interval_s"), NA=meta.get("NA"), objective=meta.get("objective", ""),
            camera=meta.get("camera", ""), channels=chans[:nC], dataset=project_dir.name,
            rel=str(lof.relative_to(project_dir).with_suffix("")),
        ))
    return images


def describe(images: List[LofImage]) -> str:
    lines = []
    for im in images:
        ch = ", ".join(f"C{c.index}:{c.led_name}/{c.lut}{'' if c.is_fluo else '(BF)'}" for c in im.channels)
        lines.append(f"{im.rel:40s} {im.dims} {im.shape}  px={im.pixel_nm:.1f}nm NA={im.NA}  [{ch}]")
    return "\n".join(lines)
