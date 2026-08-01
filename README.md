# Shrimp Detection

Shrimp detection using a custom Ultralytics YOLO model, in two independent
front ends that share the same weights:

- **Web app** — a FastAPI service with image inference, uploaded-video
  streaming, and live camera inference from the browser. Documented below.
- **Android app** — Flutter, running the model on-device via TFLite with no
  server and no network. See **[mobile_app/README.md](mobile_app/README.md)**
  for its own setup and run instructions.

## Repository structure

```text
shrimp-detection/
|-- app/
|   |-- main.py                  # FastAPI application
|   `-- static/                  # Web interface and PWA assets
|-- mobile_app/                  # Flutter Android app (on-device TFLite)
|   |-- lib/                     # Dart source
|   |-- assets/models/           # Exported .tflite model
|   `-- README.md                # Setup and run guide for the mobile app
|-- docs/images/                 # Documentation images
|-- models/best.pt               # YOLO model weights
|-- scripts/
|   |-- benchmark_api.py         # End-to-end API benchmark
|   |-- benchmark_model.py       # Local model benchmark
|   |-- run_benchmark.sh         # Benchmark helper
|   |-- export_tflite.py         # best.pt -> TFLite for the mobile app
|   |-- verify_tflite.py         # Cross-check TFLite against PyTorch
|   `-- requirements-export.txt  # Pinned deps for the export pipeline
|-- training_notebooks/          # Training notebooks
|-- uploads/                     # Runtime uploads, ignored by Git
|-- runs/                        # Generated output, ignored by Git
|-- .env.example                 # Environment configuration example
`-- requirements.txt             # Project dependencies
```

## Requirements

- Python 3.10 or newer
- A compatible Ultralytics YOLO `.pt` model
- FFmpeg, optionally, for browser-compatible uploaded videos

## Training dataset export formats

When exporting labeled data from Roboflow, use the format that matches the
training framework for each model:

| Model | Roboflow export format | Notes |
| --- | --- | --- |
| YOLOv8 / YOLOv11 | `YOLOv8` or `YOLOv11` | Native format for Ultralytics YOLO training. |
| RT-DETR | `YOLOv8` or `YOLOv11` | Use this when training RT-DETR through `ultralytics.RTDETR`; it reads `data.yaml` directly. |
| Faster R-CNN | `COCO JSON` | Best fit for PyTorch detection pipelines; annotations are stored in `_annotations.coco.json`. |
| EfficientDet | `COCO JSON` | Most EfficientDet implementations expect COCO-style boxes/classes. |

Recommended exports:

```text
datasets/
|-- shrimp-yolo/                 # Export from Roboflow as YOLOv11 or YOLOv8
|   |-- data.yaml
|   |-- train/images/
|   |-- train/labels/
|   |-- valid/images/
|   |-- valid/labels/
|   |-- test/images/
|   `-- test/labels/
`-- shrimp-coco/                 # Export from Roboflow as COCO JSON
    |-- train/
    |   `-- _annotations.coco.json
    |-- valid/
    |   `-- _annotations.coco.json
    `-- test/
        `-- _annotations.coco.json
```

Use `shrimp-yolo` for YOLO and RT-DETR notebooks. Use `shrimp-coco` for
Faster R-CNN and EfficientDet notebooks.

## Training notebooks

Training is intended to run in Google Colab with a GPU runtime.

Available notebooks:

| Notebook | Models | Dataset export required |
| --- | --- | --- |
| `training_notebooks/shrimp.ipynb` | YOLOv8 / YOLOv11 comparison | `YOLOv8` or `YOLOv11` |
| `training_notebooks/train_fasterrcnn_effdet_rtdetr.ipynb` | Faster R-CNN, EfficientDet, RT-DETR | `COCO JSON` for Faster R-CNN/EfficientDet, `YOLOv8` or `YOLOv11` for RT-DETR |

Recommended Colab paths:

```text
/content/shrimp-yolo/
|-- data.yaml
|-- train/images/
|-- train/labels/
|-- valid/images/
|-- valid/labels/
|-- test/images/
`-- test/labels/

/content/shrimp-coco/
|-- train/
|   `-- _annotations.coco.json
|-- valid/
|   `-- _annotations.coco.json
`-- test/
    `-- _annotations.coco.json
```

Run order in Colab:

1. Upload or unzip the Roboflow export so the folders match the paths above.
2. Open the notebook.
3. Use `Runtime > Change runtime type > GPU`.
4. Run the cells from top to bottom.
5. Trained outputs are saved to Google Drive:
   - YOLO notebooks: `/content/drive/MyDrive/shrimp/runs`
   - Faster R-CNN / EfficientDet / RT-DETR notebook: `/content/drive/MyDrive/shrimp/runs_detection_models`

If Colab runs out of GPU memory, reduce `BATCH_SIZE` in the notebook to `2` or
`1`.

Example Colab dataset setup:

```bash
# YOLOv8/YOLOv11 export for YOLO and RT-DETR
unzip shrimp-yolo.zip -d /content/shrimp-yolo

# COCO JSON export for Faster R-CNN and EfficientDet
unzip shrimp-coco.zip -d /content/shrimp-coco
```

After unzipping, verify that these files exist before running the training
cells:

```bash
ls /content/shrimp-yolo/data.yaml
ls /content/shrimp-coco/train/_annotations.coco.json
ls /content/shrimp-coco/valid/_annotations.coco.json
ls /content/shrimp-coco/test/_annotations.coco.json
```

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

For questions, please contact huybm.ds@gmail.com.
