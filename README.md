# Shrimp Detection

Web application for shrimp detection using a custom Ultralytics YOLO model.
The FastAPI service supports image inference, uploaded-video streaming, and
live camera inference from the browser.

## Repository structure

```text
shrimp-detection/
|-- app/
|   |-- main.py                  # FastAPI application
|   `-- static/                  # Web interface and PWA assets
|-- docs/images/                 # Documentation images
|-- models/best.pt               # YOLO model weights
|-- scripts/
|   |-- benchmark_api.py         # End-to-end API benchmark
|   |-- benchmark_model.py       # Local model benchmark
|   `-- run_benchmark.sh         # Benchmark helper
|-- uploads/                     # Runtime uploads, ignored by Git
|-- runs/                        # Generated output, ignored by Git
|-- .env.example                 # Environment configuration example
`-- requirements.txt             # Project dependencies
```

## Requirements

- Python 3.10 or newer
- A compatible Ultralytics YOLO `.pt` model
- FFmpeg, optionally, for browser-compatible uploaded videos

## Local setup

Run every command from the repository root, `shrimp-detection`:

```bash
cd shrimp-detection
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open <http://localhost:8000>. API documentation is available at `/docs`, and
service readiness is available at `/health`.

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `YOLO_MODEL` | `models/best.pt` | Absolute path or path relative to repository root |
| `YOLO_IMGSZ` | `1280` | Default inference image size |
| `YOLO_CONF` | `0.10` | Default confidence threshold |
| `CORS_ORIGINS` | `http://localhost:8000` | Comma-separated allowed origins |
| `MAX_IMAGE_UPLOAD_MB` | `20` | Maximum image upload size in MB |
| `MAX_VIDEO_UPLOAD_MB` | `500` | Maximum video upload size in MB |

## Benchmarks

```bash
IMG_SIZE=640 TRIALS=50 WARMUP=10 python scripts/benchmark_model.py
python scripts/benchmark_api.py --image path/to/image.jpg --num-requests 100
```

Benchmark CSV files are written to `runs/benchmarks/` and are ignored by Git.

## Push to GitHub

Push the **`shrimp-detection` folder**, because this folder contains `.git`,
`README.md`, the application, model, scripts, and Git configuration:

```bash
cd /Users/huybui/Desktop/shrimp/shrimp-detection
git add .
git commit -m "chore: standardize project structure"
git push origin main
```

The current remote still points to `AmadasResearchGroup/Talent`. Rename that
repository to `shrimp-detection` in GitHub Settings, then update the local URL if
GitHub does not redirect it automatically:

```bash
git remote set-url origin https://github.com/AmadasResearchGroup/shrimp-detection.git
```

Do not push the parent `shrimp` folder. Dataset ZIP files, PDFs, uploads, logs,
cache files, and benchmark outputs are excluded from Git.
