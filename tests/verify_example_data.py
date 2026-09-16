"""Verify the installation on the example field shipped in data/.

data/A488_raw_1024x1024x21.tif is a 1024x1024 crop (21 planes) of a widefield z-stack with single-mRNA spots (A488);
data/A488_darksec_1024x1024x21.tif is the same region of the darksec output of the FULL 2304x2304 field, processed with
the dataset calibration data/A488_calibration.json. This script processes the crop alone through the command-line tool
(run_ds.py, TIFF input, calibration.from_file) and compares. With a dataset-level calibration the result must not depend
on the field of view, so the two must agree except for a narrow border (reflective padding): p99.9 of |difference| in
the interior below 0.01 % of the p99.9 intensity, Pearson r above 0.9999.

Second check: the independent CPU implementation of the published algorithm (darksec/upstream_ref.py) and darksec on
the GPU, run on the same crop with the same per-stack scaling and the published padding, must agree to r > 0.99999.

    python tests/verify_example_data.py
"""
import subprocess, sys, tempfile
from pathlib import Path
import numpy as np, tifffile, yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data"
RAW, REF, CAL = DATA / "A488_raw_1024x1024x21.tif", DATA / "A488_darksec_1024x1024x21.tif", DATA / "A488_calibration.json"
PIXEL_NM, NA, Z_STEP_UM = 103.554, 1.40, 0.3995


def main():
    for p in (RAW, REF, CAL):
        if not p.exists():
            print(f"missing {p}"); return 1
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    with tempfile.TemporaryDirectory(prefix="darksec_example_") as tmp:
        cfg["datasets"] = [dict(path=str(RAW), type="tiff", name="example", pixel_nm=PIXEL_NM, NA=NA, z_step_um=Z_STEP_UM, channels=["470nm"])]
        cfg["output_root"] = tmp
        cfg["calibration"] = {**cfg.get("calibration", {}), "from_file": str(CAL), "reuse": False}
        cfg["output"] = {**cfg.get("output", {}), "dtype": "uint16", "skip_existing": False, "write_mip": False}
        cpath = Path(tmp) / "config_example.yaml"; yaml.safe_dump(cfg, open(cpath, "w"))
        r = subprocess.run([sys.executable, str(ROOT / "run_ds.py"), "run", "--config", str(cpath)], capture_output=True, text=True)
        tail = "\n".join(r.stdout.strip().splitlines()[-4:])
        print(tail)
        if r.returncode != 0:
            print(r.stderr[-2000:]); return 1
        out = np.squeeze(tifffile.imread(str(Path(tmp) / "example" / f"{RAW.stem}_DS.tif"))).astype(np.float64)
    ref = np.squeeze(tifffile.imread(str(REF))).astype(np.float64)
    if out.shape != ref.shape:
        print(f"shape mismatch {out.shape} vs {ref.shape}"); return 1
    m = 64                                                  # border affected by the padding of the crop
    a, b = out[:, m:-m, m:-m], ref[:, m:-m, m:-m]
    scale = np.percentile(b, 99.9); d = np.abs(a - b)
    p999, mx = np.percentile(d, 99.9) / scale * 100, d.max() / scale * 100
    r = np.corrcoef(a.ravel()[::7], b.ravel()[::7])[0, 1]
    print(f"crop processed alone vs shipped output of the full field (interior, {m} px border excluded):")
    print(f"  |diff| p99.9 = {p999:.4f} % of p99.9 intensity, max = {mx:.3f} %, Pearson r = {r:.6f}")
    ok = p999 < 0.01 and r > 0.9999
    ok &= cpu_vs_gpu()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def cpu_vs_gpu():
    """CPU reference implementation vs darksec (GPU) on the example crop, identical inputs and settings."""
    import time
    from darksec.core import DarkSectioner, DSParams, Optics, Scale
    from darksec.upstream_ref import dark_sectioning_upstream
    raw = np.squeeze(tifffile.imread(str(RAW))).astype(np.float64)
    t0 = time.time(); cpu, _ = dark_sectioning_upstream(raw, NA=NA, emwavelength=520.0, pixelsize=PIXEL_NM, background=1, thres=70); t_cpu = time.time() - t0
    u = Scale(float(raw.min()), float(raw.max())).to_working(raw)                      # the same per-stack min-max scaling
    ds = DarkSectioner(raw.shape[1:], Optics(NA=NA, emission_nm=520.0, pixel_nm=PIXEL_NM, factor=2.0),
                       DSParams(background="moderate", thres=70, denoise=False, atmosphere_mode="per_slice", exact_upstream=True))
    t0 = time.time(); gpu = np.asarray(ds.run(u, return_stats=False)).astype(np.float64); t_gpu = time.time() - t0
    a, b = cpu.ravel()[::7], gpu.ravel()[::7]
    r = np.corrcoef(a, b)[0, 1]; scale = float(a @ b / (b @ b)); rmse = np.sqrt(np.mean((a - scale * b) ** 2)) / np.sqrt(np.mean(a ** 2))
    print(f"CPU reference implementation vs darksec GPU on the example crop (same scaling, published padding):")
    print(f"  Pearson r = {r:.6f}, scale-only residual = {100 * rmse:.3f} %   (CPU {t_cpu:.1f} s, GPU {t_gpu:.2f} s)")
    return bool(r > 0.99999 and rmse < 0.001)


if __name__ == "__main__":
    sys.exit(main())
