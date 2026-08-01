# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A FastAPI web app that serves shrimp-disease detection using a custom Ultralytics YOLO model (`models/best.pt`). It supports single-image inference, uploaded-video inference (MJPEG stream), and a browser-based live-camera UI. Training happens separately in Colab notebooks (`training_notebooks/`), not in this app.

## Commands

Run everything from the repo root.

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env

# Run the app (dev)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000`; Swagger docs at `/docs`; readiness at `/health`.

There is no test suite, linter, or CI config in this repo — do not invent commands for them.

### Benchmarks

```bash
IMG_SIZE=640 TRIALS=50 WARMUP=10 python scripts/benchmark_model.py   # local model latency (writes runs/benchmarks/benchmark_model.csv)
python scripts/benchmark_api.py --image path/to/image.jpg --num-requests 100  # end-to-end /predict latency
./scripts/run_benchmark.sh local   # or `ngrok`, or an explicit base URL — wraps benchmark_api.py
```

## Architecture

Everything server-side lives in the single file [app/main.py](app/main.py); there's no package split by concern (routes, inference, and utils are all in one module).

- **Model lifecycle**: the YOLO model is loaded once in the FastAPI `lifespan` context into the module-level `MODEL`/`DEVICE` globals (CUDA if available, else CPU). `MODEL_LOCK` (a `threading.Lock`) serializes calls to `model.predict(...)` since Ultralytics YOLO is not safely reentrant across concurrent requests.
- **Config is env-driven**: `YOLO_MODEL`, `YOLO_IMGSZ`, `YOLO_CONF`, `CORS_ORIGINS`, `MAX_IMAGE_UPLOAD_MB`, `MAX_VIDEO_UPLOAD_MB` are read once at import time from `.env`-style environment variables (see `.env.example`). Other tuning constants (IoU, max detections, distance/confidence-boost params) are hardcoded near the top of `main.py`, not env-configurable.
- **Distance-based post-filter**: `_filter_detect_only` is the shared detection pipeline used by both the image and video endpoints. It runs `model.predict`, estimates a pseudo-distance for each box from its pixel height via a pinhole-camera approximation (`distance_from_bbox_h`, calibrated by the `REF_*`/`FOCAL_PX` constants), boosts/decays confidence by that distance (`distance_conf_boost`, sigmoid-shaped), and drops boxes that are too far (`MAX_DISTANCE_TO_KEEP`) or whose distance-adjusted confidence is too low (`ADJ_CONF_MIN`). This is a legacy "lane-aware" filter with the actual lane/polygon geometry stripped out (see the `# Lane ... removed` comments) — only the distance/confidence filtering remains, `_draw_hud` is now a no-op passthrough.
- **Three inference surfaces**, all going through `_filter_detect_only`:
  - `POST /predict` — single image upload, returns JSON boxes plus (optionally) a base64 PNG with detections drawn.
  - `POST /upload_video` — saves an uploaded video to `uploads/`, transcodes to browser-compatible H.264 MP4 via `ffmpeg` if available (`ensure_web_mp4`; silently no-ops if `ffmpeg` is missing), and registers it in the in-memory `VIDEO_STORE` dict keyed by a generated UUID.
  - `GET /mjpeg/{vid_id}` — streams the stored video back frame-by-frame as `multipart/x-mixed-replace` MJPEG, running inference on each (strided) frame on the fly.
- **Static frontend**: `app/static/` (`index.html`, `scripts.js`, `styles.css`, PWA manifest/service worker) is a plain JS SPA served via `StaticFiles`, calling `/predict`, `/upload_video`, and `/mjpeg/{id}` directly with `fetch`. `runs/` and `uploads/` are also mounted as static dirs so generated/uploaded files are directly fetchable.
- **State is in-process and ephemeral**: `VIDEO_STORE` is a plain dict with no persistence or eviction — restarting the server loses knowledge of previously uploaded videos (though the files remain in `uploads/`).

## Flutter mobile app (`mobile_app/`)

A separate on-device Android app that runs the same model via TFLite — no server, no network. It does **not** call the FastAPI service; the two are independent consumers of `models/best.pt`. See [mobile_app/README.md](mobile_app/README.md) for the full workflow.

```bash
cd mobile_app && flutter run                                    # needs an emulator/device
flutter test                                                    # letterbox + NMS math, no device needed
flutter test integration_test/detector_test.dart -d <device-id> # real model on real device
.venv-export/bin/python scripts/export_tflite.py                # re-export after retraining (run from repo root)
.venv-export/bin/python scripts/verify_tflite.py                # PyTorch vs TFLite cross-check
```

Things that will bite you here:

- **`yolo export format=tflite` no longer works.** Ultralytics 8.4.83+ redirects it to `litert-torch`, which breaks on torch 2.13 with `ImportError: get_cuda_generator_meta_val`. [scripts/export_tflite.py](scripts/export_tflite.py) goes the old `.pt → ONNX → onnx2tf → TFLite` route instead, and needs `output_signaturedefs=True` because YOLO11's C2PSA attention block emits OP names starting with `/`.
- **Letterbox padding must round down**, not `.round()`. When the pad is odd (e.g. 777×333 → 512×219 leaves 293px), Ultralytics puts 146 on top and 147 on the bottom; rounding 146.5 up shifts every box by several pixels with no error raised. Cost real debugging time — see the comment in [mobile_app/lib/services/letterbox.dart](mobile_app/lib/services/letterbox.dart).
- **Comparing against `ultralytics.predict()` requires `rect=False`.** By default predict pads only to a multiple of 32 (a 900×500 image enters the model as 288×512), while the TFLite model has a fixed 512×512 input. Without that flag the two disagree by ~6px and it looks like a decode bug.
- **Class names are hardcoded** in [mobile_app/lib/core/constants.dart](mobile_app/lib/core/constants.dart) — `best.pt` stores them as `'0'`/`'1'`; the real meaning (`0: Healthy, 1: Diseased`) only exists in the notebook. Retraining with a different class order silently mislabels everything.
- `android/build.gradle.kts` has a `subprojects` block forcing JVM target 17 — `tflite_flutter` declares Java 11 while Kotlin follows Android Studio's JDK 21, and Gradle rejects the mismatch.

## Datasets and training

Training is not run from this app — it happens in Colab notebooks under `training_notebooks/`:
- `shrimp.ipynb` — YOLOv8/YOLOv11 comparison (needs a Roboflow `YOLOv8`/`YOLOv11` export).
- `train_fasterrcnn_effdet_rtdetr.ipynb` — Faster R-CNN, EfficientDet, RT-DETR (COCO JSON export for Faster R-CNN/EfficientDet; YOLO export for RT-DETR).

`data/` holds local Roboflow exports in both COCO (`tom_benh.v1i.coco`) and YOLOv11 (`tom_benh.v1i.yolov11`) formats; `data/` and `datasets/` are gitignored. See the README's "Training dataset export formats" table before adding new export logic.

`runs/recovered_results/` contains prior model-comparison CSVs/markdown summaries (Faster R-CNN vs EfficientDet vs RT-DETR vs YOLO11n) — treat these as historical results, not something to regenerate automatically.
