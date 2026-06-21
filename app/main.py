import os
import uuid
import subprocess
import shutil
import logging
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
import base64
from threading import Lock
from typing import Any, Optional

import numpy as np
import cv2
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# ========================= CONFIG (env-overridable) =========================
APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
STATIC_DIR = APP_DIR / "static"
RUNS_DIR = PROJECT_DIR / "runs"
UPLOAD_DIR = PROJECT_DIR / "uploads"

DEFAULT_MODEL = os.getenv("YOLO_MODEL", "models/best.pt")
DEFAULT_IMGSZ = int(os.getenv("YOLO_IMGSZ", "1280"))
DEFAULT_CONF = float(os.getenv("YOLO_CONF", "0.10"))
MAX_IMAGE_UPLOAD_BYTES = int(os.getenv("MAX_IMAGE_UPLOAD_MB", "20")) * 1024**2
MAX_VIDEO_UPLOAD_BYTES = int(os.getenv("MAX_VIDEO_UPLOAD_MB", "500")) * 1024**2
ALLOWED_VIDEO_SUFFIXES = {".avi", ".mkv", ".mov", ".mp4", ".webm"}
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:8000").split(",")
    if origin.strip()
]

# Inference and distance-filter configuration.

INFER_IOU       = 0.60
MAX_DET         = 300

REF_SIGN_HEIGHT_M = 0.60
REF_DISTANCE_M    = 15.0
REF_BBOX_PX       = 100.0
FOCAL_PX          = (REF_BBOX_PX * REF_DISTANCE_M) / REF_SIGN_HEIGHT_M
PREF_DIST_M          = 40.0
SIGMOID_K            = 0.08
MAX_DISTANCE_TO_KEEP = 120.0
ADJ_CONF_MIN         = 0.05

HIGH_SIGN_Y_FRAC         = 0.80
MIN_HIGH_OVERLAP_FRAC    = 0.10
CENTER_PRIOR_K           = 4.0
CENTER_LOCK_FRAC         = 0.12
MARGIN_PIX_TOP           = 8

# Lane speed mapping removed - lane-aware details are no longer used.

# ========================= APP & STATIC =========================
STATIC_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MODEL: Optional[YOLO] = None
DEVICE: Any = None
MODEL_LOCK = Lock()


def resolve_model_path() -> Path:
    model_path = Path(DEFAULT_MODEL).expanduser()
    return model_path if model_path.is_absolute() else PROJECT_DIR / model_path


@asynccontextmanager
async def lifespan(_: FastAPI):
    global MODEL, DEVICE
    model_path = resolve_model_path()
    if not model_path.is_file():
        raise RuntimeError(
            f"Model not found: {model_path}. Set YOLO_MODEL to a valid weights file."
        )
    DEVICE = 0 if torch.cuda.is_available() else "cpu"
    MODEL = YOLO(str(model_path))
    MODEL.to(DEVICE if DEVICE == "cpu" else 0)
    names = getattr(MODEL, "names", {})
    names_len = len(names) if isinstance(names, (list, dict)) else 0
    logger.info(
        f"[INIT] model={model_path.name} size={human_mb(model_path):.2f}MB "
        f"device={'cuda' if DEVICE != 'cpu' else 'cpu'} classes={names_len}"
    )
    yield
    MODEL = None


app = FastAPI(
    title="Shrimp Detection API",
    description="YOLO-based image and video inference service.",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS, allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/runs", StaticFiles(directory=RUNS_DIR), name="runs")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

@app.get("/", response_class=FileResponse)
def index():
    idx = STATIC_DIR / "index.html"
    if not idx.exists():
        raise HTTPException(500, "static/index.html not found")
    return FileResponse(idx)

def human_mb(p: Path) -> float:
    return (p.stat().st_size if p.exists() else 0) / (1024 * 1024)


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok", "model": "ready" if MODEL is not None else "unavailable"}

# ========================= UTILS: IO =========================
def pil_from_upload(f: UploadFile) -> Image.Image:
    data = f.file.read(MAX_IMAGE_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(400, "Empty file")
    if len(data) > MAX_IMAGE_UPLOAD_BYTES:
        raise HTTPException(413, "Image exceeds the configured upload limit")
    try:
        return Image.open(BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(400, "Invalid image file") from exc

def encode_png_b64(img_np_bgr: np.ndarray) -> str:
    img_rgb = Image.fromarray(img_np_bgr[..., ::-1])
    buf = BytesIO()
    img_rgb.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

def ensure_web_mp4(in_path: Path) -> Path:
    # H.264 with yuv420p provides broad browser compatibility.
    if in_path.suffix.lower() != ".mp4":
        return in_path
    if shutil.which("ffmpeg") is None:
        return in_path
    out_path = in_path.with_name(in_path.stem + "_web.mp4")
    cmd = [
        "ffmpeg", "-y", "-i", str(in_path),
        "-vcodec", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "veryfast", "-crf", "23",
        "-acodec", "aac", "-movflags", "+faststart",
        str(out_path)
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return out_path if out_path.exists() else in_path
    except Exception:
        logger.warning("FFmpeg conversion failed for %s", in_path, exc_info=True)
        return in_path

def probe_fps(path: Path) -> float | None:
    try:
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return float(fps) if fps and fps > 1e-3 else None
    except Exception:
        return None

# ========================= UTILS: Lane math =========================
def sigmoid(x: float) -> float:
    return 1/(1+np.exp(-x))

def distance_from_bbox_h(h_px: float, H: float = REF_SIGN_HEIGHT_M, f: float = FOCAL_PX) -> float:
    return (H*f)/max(float(h_px),1.0)

def distance_conf_boost(conf: float, d: float, d0: float = PREF_DIST_M, k: float = SIGMOID_K) -> float:
    return conf * sigmoid(k*(d0-d))

def draw_label(img, x1, y1, text, offset=40):
    f=cv2.FONT_HERSHEY_SIMPLEX; s=0.6; t=2
    (tw,th),_ = cv2.getTextSize(text,f,s,t)
    cv2.rectangle(img,(int(x1), int(y1)+offset),
                  (int(x1)+tw+8, int(y1)+th+offset+8),(0,0,0), -1)
    cv2.putText(img, text,(int(x1)+4, int(y1)+th+offset+4),
                f, s, (255,255,255), t, cv2.LINE_AA)

# point_in_poly removed; not used after lane detection removal



# Lane detection removed: no polygon/line-based geometry used anymore

def build_name_index(class_names):
    if isinstance(class_names, dict): return {int(k): str(v) for k,v in class_names.items()}
    return {i: str(nm) for i,nm in enumerate(class_names)}

def _draw_hud(img: np.ndarray) -> np.ndarray:
    """Simple HUD drawing: returns a copy of img; lane-specific HUD removed."""
    return img.copy()

# ========================= LANE-AWARE FILTER =========================
def _filter_detect_only(
    model: YOLO,
    img_for_yolo: np.ndarray,
    class_names,
    imgsz: int,
    conf: float,
):
    """Simple filter that runs prediction and filters only by distance/confidence.

    Returns kept, drop_log similarly shaped to previous implementation but without lane context.
    """
    name_by_id = build_name_index(class_names)

    with MODEL_LOCK:
        det = model.predict(
            img_for_yolo,
            imgsz=imgsz,
            conf=conf,
            iou=INFER_IOU,
            max_det=MAX_DET,
            verbose=False,
            device=(0 if DEVICE != "cpu" else "cpu"),
        )[0]

    kept, drop_log = [], []
    H, W = img_for_yolo.shape[:2]
    for b in det.boxes:
        xyxy = b.xyxy[0].cpu().numpy()
        conf = float(b.conf[0]); cls_id = int(b.cls[0])
        x1,y1,x2,y2 = xyxy; w_box = max(1.0,x2-x1); h_box = max(1.0,y2-y1)
        xc,yc = x1+w_box/2.0, y1+h_box/2.0

        raw_name = name_by_id.get(cls_id, f"id{cls_id}")
        dist_m = distance_from_bbox_h(h_box)
        if dist_m > MAX_DISTANCE_TO_KEEP:
            drop_log.append((xc,yc,"far")); continue
        adj_conf = distance_conf_boost(conf, dist_m)
        if adj_conf < ADJ_CONF_MIN:
            drop_log.append((xc,yc,"low_conf")); continue

        high_sign = (yc < HIGH_SIGN_Y_FRAC * H)
        kept.append({"xyxy":xyxy,"conf":adj_conf,"cls":cls_id,"dist":dist_m,
                     "lbl":raw_name,"fullname":raw_name,
                     "xc":xc,"yc":yc,"high":bool(high_sign)})

    return kept, drop_log

# ========================= API: Predict Image =========================
@app.post("/predict")
def predict(
    file: UploadFile = File(...),
    imgsz: int = Form(DEFAULT_IMGSZ),
    conf: float = Form(DEFAULT_CONF),
    save_annotated: bool = Form(True),
) -> JSONResponse:
    if MODEL is None:
        raise HTTPException(503, "Model not loaded")

    imgsz = max(32, int(imgsz))
    conf = min(1.0, max(0.0, float(conf)))

    # 1) Upload -> PIL -> numpy BGR
    img_pil = pil_from_upload(file)
    img_np = np.array(img_pil)[:, :, ::-1].copy()  # BGR

    try:
        # Draw annotations on a separate image.
        vis = _draw_hud(img_np.copy())

        # 3) YOLO on original image + detection-only filter
        kept, _ = _filter_detect_only(MODEL, img_np, MODEL.names, imgsz, conf)

        # Return normalized detection data to the UI.
        boxes = []
        for nb in kept:
            x1, y1, x2, y2 = nb["xyxy"]
            cls_id = int(nb["cls"])
            boxes.append({
                "x1": float(x1), "y1": float(y1),
                "x2": float(x2), "y2": float(y2),
                "conf": float(nb["conf"]),
                "cls": cls_id,
                "cls_name": (MODEL.names.get(cls_id, str(cls_id))
                             if isinstance(MODEL.names, dict)
                             else (MODEL.names[cls_id] if isinstance(MODEL.names, list) and 0 <= cls_id < len(MODEL.names) else str(cls_id)))
            })

        payload = {"boxes": boxes}

        # Include a base64-encoded annotated image when requested.
        if save_annotated:
            for nb in kept:
                x1,y1,x2,y2 = nb["xyxy"]
                cv2.rectangle(vis, (int(x1),int(y1)), (int(x2),int(y2)), (0,220,0), 2)
                draw_label(vis, x1, y1,
                           f"{nb['fullname']} | {nb['conf']:.2f} | {nb['dist']:.1f}m",
                           offset=40)
            payload["image"] = encode_png_b64(vis)

        return JSONResponse(payload)

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Image inference failed")
        raise HTTPException(500, "Inference failed") from exc

# ========================= API: Video Upload + MJPEG Stream =========================
VIDEO_STORE: dict[str, Path] = {}

@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    if MODEL is None:
        raise HTTPException(503, "Model not loaded")
    suffix = Path(file.filename or "").suffix or ".mp4"
    if suffix.lower() not in ALLOWED_VIDEO_SUFFIXES:
        raise HTTPException(400, "Unsupported video format")
    vid_id = uuid.uuid4().hex
    saved = UPLOAD_DIR / f"{vid_id}{suffix.lower()}"
    total_bytes = 0
    try:
        with saved.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > MAX_VIDEO_UPLOAD_BYTES:
                    raise HTTPException(413, "Video exceeds the configured upload limit")
                output.write(chunk)
    except Exception:
        saved.unlink(missing_ok=True)
        raise
    web_mp4 = ensure_web_mp4(saved)
    VIDEO_STORE[vid_id] = saved
    return {
        "id": vid_id,
        "orig_url": f"/uploads/{web_mp4.name}",
        "annot_url": f"/mjpeg/{vid_id}",
    }

@app.get("/mjpeg/{vid_id}")
def mjpeg_annot(
    vid_id: str,
    conf: float = DEFAULT_CONF,
    imgsz: int = DEFAULT_IMGSZ,
    stride: int = 1,
    start: float = 0.0,   # seconds
):
    if MODEL is None:
        raise HTTPException(503, "Model not loaded")
    if vid_id not in VIDEO_STORE:
        raise HTTPException(404, "Video id not found")

    imgsz = max(32, int(imgsz))
    conf = min(1.0, max(0.0, float(conf)))

    src_path = VIDEO_STORE[vid_id]
    fps = probe_fps(src_path) or 25.0
    stride = max(1, int(stride))

    def gen():
        try:
            cap = cv2.VideoCapture(str(src_path))
            if start > 0:
                cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)

            idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % stride != 0:
                    idx += 1
                    continue

                # HUD: no lane detection
                vis = _draw_hud(frame.copy())

                # YOLO on frame + detection-only filter
                kept, _ = _filter_detect_only(
                    MODEL, frame, MODEL.names, imgsz, conf
                )

                # Draw accepted detections.
                for nb in kept:
                    x1,y1,x2,y2 = nb["xyxy"]
                    cv2.rectangle(vis, (int(x1),int(y1)), (int(x2),int(y2)), (0,220,0), 2)
                    draw_label(vis, x1, y1,
                               f"{nb['fullname']} | {nb['conf']:.2f} | {nb['dist']:.1f}m",
                               offset=40)

                ok2, jpg = cv2.imencode(".jpg", vis)
                if not ok2:
                    idx += 1
                    continue
                data = jpg.tobytes()
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n"
                       b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n"
                       + data + b"\r\n")
                idx += 1

            cap.release()
        except Exception:
            logger.exception("Video inference failed for id=%s", vid_id)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")
