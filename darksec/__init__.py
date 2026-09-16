"""darksec — Dark Sectioning (GPU) with globally calibrated intensity scaling.

Public entry points
-------------------
core.DarkSectioner       : the slice-wise GPU algorithm (array in, array out)
leica.index_dataset      : find image .lof files + metadata in a Leica project
calibrate.calibrate      : global lo/hi (and atmosphere) per channel
pipeline.process_file    : one .lof -> DS TIFF (+ MIP, JSON sidecar)
"""
__version__ = "1.0.0"
