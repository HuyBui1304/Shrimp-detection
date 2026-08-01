#!/usr/bin/env python3
"""Export models/best.pt sang TFLite cho app Flutter (mobile_app/).

Chạy bằng interpreter của .venv-export, không dùng python toàn cục:

    .venv-export/bin/python scripts/export_tflite.py

Vì sao không dùng thẳng `yolo export format=tflite`:
    Từ ultralytics 8.4.83, format='tflite' bị chuyển hướng sang đường
    'litert' (litert-torch). Gói đó hiện không tương thích torch 2.13 và
    gãy ở `torch.export.export` với:
        ImportError: cannot import name 'get_cuda_generator_meta_val'
    Nên script này đi đường cũ, vẫn ổn định: .pt -> ONNX -> onnx2tf -> TFLite.
"""

import shutil
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
WEIGHTS = PROJECT_DIR / "models" / "best.pt"
ONNX_PATH = PROJECT_DIR / "models" / "best.onnx"
TFLITE_DIR = PROJECT_DIR / "models" / "tflite_export"
ASSET_PATH = (
    PROJECT_DIR / "mobile_app" / "assets" / "models" / "shrimp_yolo11n_512.tflite"
)

# Phải khớp kích thước đã train (shrimp.ipynb: imgsz=512, 150 epochs).
IMGSZ = 512
NUM_CLASSES = 2


def log(msg: str) -> None:
    print(f"[export] {msg}", flush=True)


def export_onnx() -> None:
    from ultralytics import YOLO

    log(f"xuất ONNX từ {WEIGHTS.name} ở imgsz={IMGSZ}")
    YOLO(str(WEIGHTS)).export(format="onnx", imgsz=IMGSZ, opset=13)
    if not ONNX_PATH.is_file():
        raise SystemExit(f"không thấy {ONNX_PATH}")
    log(f"ONNX xong: {ONNX_PATH.name} ({ONNX_PATH.stat().st_size / 1e6:.1f} MB)")


CALIBRATION_NPY = "calibration_image_sample_data_20x128x128x3_float32.npy"


def ensure_calibration_data() -> None:
    """Đặt sẵn file mẫu mà onnx2tf cần, để nó không phải tải qua mạng.

    onnx2tf luôn gọi download_test_image_data() cho model 4 chiều và không có
    cờ nào tắt được. Hàm đó tải .npy từ GitHub với timeout đọc 5 giây; mạng
    chập là nhận về nội dung hỏng và cả tiến trình chết với
    `_pickle.UnpicklingError: could not find MARK`.

    File này chỉ dùng để chạy cùng một input qua ONNX và qua TF rồi so lệch,
    nên dữ liệu ngẫu nhiên cũng cho kết quả so sánh có ý nghĩa như nhau.
    onnx2tf đọc từ os.getcwd() nên phải đặt đúng ở thư mục gốc dự án.
    """
    target = Path.cwd() / CALIBRATION_NPY
    if target.is_file():
        log(f"dùng lại {CALIBRATION_NPY} có sẵn")
        return
    rng = np.random.default_rng(0)
    np.save(target, rng.random((20, 128, 128, 3), dtype=np.float32))
    log(f"tạo {CALIBRATION_NPY} tại chỗ (không cần tải mạng)")


def convert_tflite() -> None:
    import onnx2tf

    ensure_calibration_data()
    log("chuyển ONNX -> TFLite bằng onnx2tf")
    onnx2tf.convert(
        input_onnx_file_path=str(ONNX_PATH),
        output_folder_path=str(TFLITE_DIR),
        copy_onnx_input_output_names_to_tflite=True,
        # onnxsim không có trong venv và ONNX đã được onnxslim tinh gọn ở
        # bước trước rồi, nên bỏ qua.
        not_use_onnxsim=True,
        # Bắt buộc với YOLO11: khối attention C2PSA sinh tên OP dạng
        # '/model.10/m/m.0/attn/pe/conv/Conv/kernel', bắt đầu bằng '/' nên
        # không hợp lệ với saved_model. Bật signaturedefs thì onnx2tf đi
        # đường đặt tên khác và qua được.
        output_signaturedefs=True,
    )


def publish_asset() -> Path:
    candidates = sorted(TFLITE_DIR.glob("*_float32.tflite"))
    if not candidates:
        candidates = sorted(TFLITE_DIR.glob("*.tflite"))
    if not candidates:
        raise SystemExit(f"onnx2tf không sinh ra .tflite nào trong {TFLITE_DIR}")

    src = candidates[0]
    ASSET_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, ASSET_PATH)
    log(f"chép {src.name} -> {ASSET_PATH.relative_to(PROJECT_DIR)}")
    return ASSET_PATH


def verify(path: Path) -> None:
    """Kiểm tra tensor thật trong file.

    App Dart phải biết chính xác hai điều: trục nào là số anchor, và toạ độ
    ở dạng chuẩn hoá hay pixel. Đoán sai bất kỳ điều nào thì box vẫn vẽ ra
    nhưng lệch hoàn toàn, không có lỗi nào báo.
    """
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    log(f"input : shape={list(inp['shape'])} dtype={inp['dtype'].__name__}")
    log(f"output: shape={list(out['shape'])} dtype={out['dtype'].__name__}")

    interpreter.set_tensor(
        inp["index"],
        np.random.rand(*inp["shape"]).astype(inp["dtype"]),
    )
    interpreter.invoke()
    data = interpreter.get_tensor(out["index"])

    channels = 4 + NUM_CLASSES
    shape = list(data.shape)
    if len(shape) != 3 or channels not in shape[1:]:
        raise SystemExit(
            f"output {shape} không có chiều nào bằng {channels} -> app Dart sẽ decode sai"
        )

    channels_first = shape[1] == channels
    anchors = shape[2] if channels_first else shape[1]
    coords = data[0, :4, :] if channels_first else data[0, :, :4]
    max_coord = float(np.abs(coords).max())

    log(f"layout: {'[1, kênh, anchor]' if channels_first else '[1, anchor, kênh]'}")
    log(f"số anchor: {anchors} (mong đợi {(IMGSZ//8)**2 + (IMGSZ//16)**2 + (IMGSZ//32)**2})")
    log(f"biên độ toạ độ lớn nhất: {max_coord:.3f} -> "
        f"{'CHUẨN HOÁ [0,1], Dart phải nhân {}'.format(IMGSZ) if max_coord <= 1.5 else 'PIXEL, dùng thẳng'}")
    log(f"kích thước file: {path.stat().st_size / 1e6:.1f} MB")


def main() -> int:
    if not WEIGHTS.is_file():
        raise SystemExit(f"không thấy {WEIGHTS}")
    export_onnx()
    convert_tflite()
    verify(publish_asset())
    log("hoàn tất")
    return 0


if __name__ == "__main__":
    sys.exit(main())
