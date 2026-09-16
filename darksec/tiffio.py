"""
TIFF stacks as inputs (same interface as leica.LofImage).

Needed for anything that is not a Leica .lof: the example field in data/,
registered seqFISH TIFFs, exports from other microscopes. Metadata that a TIFF cannot supply (pixel size, NA, channel
identity) comes from the config entry:

    datasets:
      - path: /some/dir_or_file.tif      # a file or a directory (all *.tif/*.tiff)
        type: tiff
        name: example                     # dataset name used in output paths
        pixel_nm: 65
        NA: 1.49
        channels: ["561nm"]               # channel keys, in C order; emission from optics.emission_nm
        z_step_um: null

Axes are taken from tifffile (ImageJ / OME metadata when present). A bare
multi-page TIFF (axes 'I' or 'Q') is treated as a z-stack.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import tifffile

from .leica import ChannelInfo, LofImage


class TiffStack(LofImage):
    """LofImage duck-type backed by a TIFF file."""

    def memmap(self):
        try:
            arr = tifffile.memmap(str(self.path))
        except (ValueError, TypeError):
            arr = tifffile.imread(str(self.path))
        return np.asarray(arr).reshape(self.shape)


def _axes_to_dims(axes: str, shape: Sequence[int]):
    """Map tifffile axes to our ('T','C','Z','Y','X') subset, squeezing length-1 extras."""
    dims, shp = [], []
    for a, n in zip(axes, shape):
        a = a.upper()
        if a in ("I", "Q", "S") and n > 1:
            a = "Z"
        if a in ("T", "C", "Z", "Y", "X"):
            if a in ("T", "C", "Z") and n == 1:
                continue
            dims.append(a); shp.append(int(n))
        elif n != 1:
            raise ValueError(f"unsupported TIFF axis {a!r} with length {n}")
    if "Y" not in dims or "X" not in dims:
        raise ValueError(f"TIFF axes {axes!r} lack Y/X")
    return tuple(dims), tuple(shp)


def tiff_image(path: Path, *, pixel_nm: float, NA: float, channels: Sequence[str], dataset: str,
               rel: Optional[str] = None, z_step_um: Optional[float] = None,
               t_interval_s: Optional[float] = None) -> TiffStack:
    path = Path(path)
    with tifffile.TiffFile(str(path)) as tf:
        ser = tf.series[0]
        dims, shape = _axes_to_dims(ser.axes, ser.shape)
        dtype = str(np.dtype(ser.dtype))
    nC = shape[dims.index("C")] if "C" in dims else 1
    keys = list(channels) if channels else [f"C{i}" for i in range(nC)]
    if len(keys) != nC:
        raise ValueError(f"{path.name}: {nC} channels in file but {len(keys)} channel keys given")
    chans = [ChannelInfo(index=i, led_name=k, led_nm=int(k[:-2]) if k.endswith("nm") and k[:-2].isdigit() else None,
                         lut="", is_fluo=True) for i, k in enumerate(keys)]
    return TiffStack(path=path, xlif=None, dims=dims, shape=shape, dtype=dtype, offset=0,
                     pixel_nm=float(pixel_nm), z_step_um=z_step_um, t_interval_s=t_interval_s, NA=float(NA),
                     objective="", camera="", channels=chans, dataset=dataset, rel=rel or path.stem)


def index_tiffs(entry: dict) -> List[TiffStack]:
    """All TIFFs described by one config entry (file or directory)."""
    p = Path(entry["path"])
    files = [p] if p.is_file() else sorted(list(p.glob("*.tif")) + list(p.glob("*.tiff")))
    name = entry.get("name") or (p.stem if p.is_file() else p.name)
    out = []
    for f in files:
        rel = f.stem if p.is_file() else str(f.relative_to(p).with_suffix(""))
        out.append(tiff_image(f, pixel_nm=entry["pixel_nm"], NA=entry["NA"], channels=entry.get("channels") or [],
                              dataset=name, rel=rel, z_step_um=entry.get("z_step_um"),
                              t_interval_s=entry.get("t_interval_s")))
    return out
