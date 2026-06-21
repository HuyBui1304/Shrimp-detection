#!/usr/bin/env python3
import os, time, hashlib, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).resolve().parent.parent
WEIGHT = Path(os.getenv("YOLO_MODEL", "models/best.pt")).expanduser()
if not WEIGHT.is_absolute():
    WEIGHT = PROJECT_DIR / WEIGHT
IMGSZ = int(os.getenv("IMG_SIZE", 320))
TRIALS = int(os.getenv("TRIALS", 10))
WARMUP = int(os.getenv("WARMUP", 3))

# Prefer CUDA, then Apple Silicon MPS, then CPU.
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

OUTCSV = PROJECT_DIR / "runs" / "benchmarks" / "benchmark_model.csv"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True

def log(s): print(f"[bench] {s}")

def bytes_mb(n): return round(n/(1024**2), 2)
def file_size_mb(p): return bytes_mb(p.stat().st_size)
def file_md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""): h.update(chunk)
    return h.hexdigest()

def count_params(model):  # -> M params
    return round(sum(p.numel() for p in model.parameters())/1e6, 3)

def try_gmacs(model, imgsz, batch=1, half=True):
    try:
        from thop import profile
    except Exception:
        return None
    dev = torch.device(DEVICE)
    m = model.eval()

    # Keep MPS inference in float32.
    m = m.to(torch.float32).to(dev)
    use_half = (half and DEVICE == "cuda")
    if use_half:
        m = m.half()

    x = torch.randn(batch, 3, imgsz, imgsz, device=dev, dtype=torch.float16 if use_half else torch.float32)
    with torch.no_grad():
        macs, _ = profile(m, inputs=(x,), verbose=False)
    return round(macs / 1e9, 3)  # GMACs


def latency_and_vram(core, imgsz, batch=1, half=True, trials=10, warmup=3):
    dev = torch.device(DEVICE)

    # Convert to float32 before moving the model to MPS.
    core = core.to(torch.float32).to(dev).eval()

    use_half = (half and DEVICE == "cuda")
    dtype = torch.float16 if use_half else torch.float32
    x = torch.rand(batch, 3, imgsz, imgsz, device=dev, dtype=dtype)
    if use_half:
        core = core.half()

    with torch.no_grad():
        # warmup
        for _ in range(warmup):
            _ = core(x)
        if DEVICE == "cuda":
            torch.cuda.synchronize()

        times = []
        if DEVICE == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for _ in range(trials):
            t0 = time.perf_counter()
            _ = core(x)
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
        vram = bytes_mb(torch.cuda.max_memory_allocated()) if DEVICE == "cuda" else 0.0

    lat = float(np.mean(times))
    fps = 1000.0 / lat if lat > 0 else float("inf")
    p50 = float(np.percentile(times, 50))
    p95 = float(np.percentile(times, 95))
    return round(lat, 3), round(p50, 3), round(p95, 3), round(fps, 2), vram

def main():
    assert WEIGHT.exists(), f"Model weights not found: {WEIGHT}"
    OUTCSV.parent.mkdir(parents=True, exist_ok=True)
    log(f"Device   : {DEVICE}")
    log(f"Weight   : {WEIGHT}")
    log(f"Img size : {IMGSZ}")
    log(f"Trials   : {TRIALS}, Warmup: {WARMUP}")

    log("Loading model ...")
    y = YOLO(str(WEIGHT))
    core = y.model
    log("Loaded.")

    params_M = count_params(core)
    size_MB  = file_size_mb(WEIGHT)
    md5      = file_md5(WEIGHT)
    log(f"Params   : {params_M} M")
    log(f"File size: {size_MB} MB")
    try:
        gmacs = try_gmacs(core, IMGSZ, batch=1, half=True)
        log(f"GMACs@{IMGSZ}: {gmacs}")
    except Exception as e:
        gmacs = None
        log(f"GMACs skipped: {e}")

    rows=[]
    for bs in (1, 8):
        log(f"Measuring latency @batch={bs} ...")
        lat, p50, p95, fps, vram = latency_and_vram(core, IMGSZ, batch=bs, half=True, trials=TRIALS, warmup=WARMUP)
        rows.append({
            "model": WEIGHT.name,
            "md5": md5,
            "file_size_MB": size_MB,
            "params_M": params_M,
            f"GMACs@{IMGSZ}": gmacs,
            "batch": bs,
            "imgsz": IMGSZ,
            "precision": "FP16" if DEVICE=="cuda" else ("MPS" if DEVICE=="mps" else "FP32"),
            "latency_ms_mean": lat,
            "latency_ms_p50": p50,
            "latency_ms_p95": p95,
            "FPS": fps,
            "peak_VRAM_MB": vram,
            "device": DEVICE
        })
        log(f" -> mean {lat} ms | p95 {p95} ms | FPS {fps} | VRAM {vram} MB")

    df = pd.DataFrame(rows)
    print("\n===== SUMMARY =====")
    print(df.to_string(index=False))
    df.to_csv(OUTCSV, index=False)
    log(f"Saved CSV: {OUTCSV}")

if __name__ == "__main__":
    main()
