#!/usr/bin/env python3
"""Đối chiếu kết quả giữa models/best.pt và file TFLite đã export.

    .venv-export/bin/python scripts/verify_tflite.py [đường/dẫn/ảnh.jpg]

Mục đích: chứng minh chuỗi letterbox -> decode -> NMS viết tay là đúng, và
bản TFLite không lệch so với bản PyTorch. Nếu bỏ bước này, sai sót chỉ lộ ra
dưới dạng "box lệch một chút" trong app — rất khó truy ngược xem lỗi nằm ở
khâu export, khâu tiền xử lý hay khâu giải mã.

Phần hậu xử lý dưới đây cố tình viết lại y hệt bản Dart trong
mobile_app/lib/services/ chứ không gọi API của Ultralytics, để nếu bản Dart
sai thì bản Python này cũng sai theo cùng kiểu và ta phát hiện được.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
WEIGHTS = PROJECT_DIR / "models" / "best.pt"
TFLITE = PROJECT_DIR / "mobile_app" / "assets" / "models" / "shrimp_yolo11n_512.tflite"
TEST_DIR = PROJECT_DIR / "data" / "tom_benh.v1i.yolov11" / "test" / "images"

IMGSZ = 512
NUM_CLASSES = 2
CONF = 0.25
IOU = 0.60
LABELS = {0: "Tôm khỏe", 1: "Tôm bệnh"}


def letterbox(image: np.ndarray, size: int = IMGSZ, pad: int = 114):
    h, w = image.shape[:2]
    scale = min(size / w, size / h)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), pad, dtype=np.uint8)
    # Làm tròn xuống để khớp Ultralytics khi phần pad là số lẻ; xem ghi chú
    # trong mobile_app/lib/services/letterbox.dart.
    dx, dy = (size - new_w) // 2, (size - new_h) // 2
    canvas[dy : dy + new_h, dx : dx + new_w] = resized
    return canvas, scale, dx, dy


def iou_matrix(box, others):
    if len(others) == 0:
        return np.zeros(0)
    x1 = np.maximum(box[0], others[:, 0])
    y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2])
    y2 = np.minimum(box[3], others[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (box[2] - box[0]) * (box[3] - box[1])
    area_b = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0)


def run_tflite(image_bgr: np.ndarray):
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(TFLITE))
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    padded, scale, dx, dy = letterbox(image_bgr)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    interpreter.set_tensor(inp["index"], rgb[None, ...])
    interpreter.invoke()
    raw = interpreter.get_tensor(out["index"])[0]  # [6, 5376]

    scores = raw[4 : 4 + NUM_CLASSES, :]
    class_ids = scores.argmax(axis=0)
    confidences = scores.max(axis=0)
    keep = confidences >= CONF

    cx, cy, bw, bh = raw[0][keep], raw[1][keep], raw[2][keep], raw[3][keep]
    boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
    # Đưa từ toạ độ ảnh 512 đã pad về toạ độ ảnh gốc.
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dx) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dy) / scale

    # Cắt về trong khung ảnh, giống scale_boxes() của Ultralytics và giống
    # LetterboxResult.toOriginal() bên Dart. Thiếu bước này thì con tôm nằm sát
    # mép ảnh sẽ có box tràn ra ngoài, và phép so sánh báo lệch vài pixel trong
    # khi thực chất hai bên tính ra cùng một kết quả.
    src_h, src_w = image_bgr.shape[:2]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, src_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, src_h)

    confidences, class_ids = confidences[keep], class_ids[keep]

    # NMS theo từng lớp.
    final = []
    for cls in range(NUM_CLASSES):
        idx = np.where(class_ids == cls)[0]
        idx = idx[np.argsort(-confidences[idx])]
        chosen = []
        while len(idx):
            best, idx = idx[0], idx[1:]
            chosen.append(best)
            if len(idx):
                idx = idx[iou_matrix(boxes[best], boxes[idx]) <= IOU]
        final.extend(chosen)

    order = np.argsort(-confidences[final])
    final = np.array(final)[order]
    return boxes[final], confidences[final], class_ids[final]


def run_pytorch(image_path: Path):
    from ultralytics import YOLO

    # rect=False là bắt buộc để so sánh công bằng.
    #
    # Mặc định predict() dùng letterbox "auto": nó chỉ pad tới bội số của 32
    # nên ảnh 900x500 vào model dưới dạng 288x512, không phải 512x512. Model
    # TFLite thì có input cố định 512x512 nên app luôn pad vuông. Nếu để mặc
    # định, hai bên nhận hai đầu vào khác nhau và lệch vài pixel — trông hệt
    # như lỗi giải mã trong khi thực chất chỉ là so sánh sai điều kiện.
    result = YOLO(str(WEIGHTS)).predict(
        str(image_path), imgsz=IMGSZ, conf=CONF, iou=IOU, rect=False, verbose=False
    )[0]
    return (
        result.boxes.xyxy.cpu().numpy(),
        result.boxes.conf.cpu().numpy(),
        result.boxes.cls.cpu().numpy().astype(int),
    )


def compare(image_path: Path) -> bool:
    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"không đọc được ảnh {image_path}")

    pt_boxes, pt_conf, pt_cls = run_pytorch(image_path)
    tf_boxes, tf_conf, tf_cls = run_tflite(image)

    print(f"\n  {image_path.name}  ({image.shape[1]}x{image.shape[0]})")
    print(f"    PyTorch: {len(pt_boxes)} box   TFLite: {len(tf_boxes)} box")

    if len(pt_boxes) != len(tf_boxes):
        print("    ✗ SỐ LƯỢNG BOX KHÁC NHAU")
        return False
    if len(pt_boxes) == 0:
        print("    (không có box nào để so)")
        return True

    worst_shift = 0.0
    worst_conf = 0.0
    for i, box in enumerate(pt_boxes):
        ious = iou_matrix(box, tf_boxes)
        j = int(ious.argmax())
        if ious[j] < 0.9 or pt_cls[i] != tf_cls[j]:
            print(f"    ✗ box {i} không khớp (IoU tốt nhất {ious[j]:.3f})")
            return False
        worst_shift = max(worst_shift, float(np.abs(box - tf_boxes[j]).max()))
        worst_conf = max(worst_conf, abs(float(pt_conf[i] - tf_conf[j])))

    counts = {LABELS[c]: int((pt_cls == c).sum()) for c in range(NUM_CLASSES)}
    print(f"    lệch toạ độ lớn nhất: {worst_shift:.2f} px")
    print(f"    lệch điểm số lớn nhất: {worst_conf:.4f}")
    print(f"    phân loại: {counts}")
    return worst_shift < 5.0


def main() -> int:
    if not TFLITE.is_file():
        raise SystemExit(f"chưa có {TFLITE}, chạy scripts/export_tflite.py trước")

    if len(sys.argv) > 1:
        images = [Path(sys.argv[1])]
    else:
        images = sorted(TEST_DIR.glob("*.jpg"))[:5]
    if not images:
        raise SystemExit(f"không tìm thấy ảnh test trong {TEST_DIR}")

    print(f"So sánh {WEIGHTS.name} với {TFLITE.name} trên {len(images)} ảnh")
    results = [compare(p) for p in images]

    passed = sum(results)
    print(f"\n{'='*54}")
    if passed == len(results):
        print(f"✓ ĐẠT — cả {len(results)} ảnh cho kết quả trùng khớp")
        return 0
    print(f"✗ HỎNG — {len(results) - passed}/{len(results)} ảnh lệch")
    return 1


if __name__ == "__main__":
    sys.exit(main())
