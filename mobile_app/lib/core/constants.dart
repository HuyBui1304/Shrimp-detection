import 'package:flutter/material.dart';

/// Hai lớp của model.
///
/// Tên lớp lưu trong `models/best.pt` là `{0: '0', 1: '1'}` — tên mặc định do
/// Roboflow sinh ra, không mang nghĩa gì. Ánh xạ thật nằm ở
/// `training_notebooks/shrimp.ipynb` cell 11:
///
///     model.model.names = {0: "Healthy", 1: "Diseased"}
///
/// Vì vậy phải hardcode ở đây thay vì đọc metadata từ file model.
class ShrimpClass {
  static const int healthy = 0;
  static const int diseased = 1;

  static const List<String> labels = ['Tôm khỏe', 'Tôm bệnh'];
  static const List<Color> colors = [Color(0xFF22C55E), Color(0xFFEF4444)];

  static String labelOf(int id) =>
      (id >= 0 && id < labels.length) ? labels[id] : 'Không rõ ($id)';

  static Color colorOf(int id) =>
      (id >= 0 && id < colors.length) ? colors[id] : Colors.grey;
}

class ModelConfig {
  /// Export từ `models/best.pt` với `imgsz=512` — đúng bằng kích thước đã dùng
  /// khi train (150 epochs, batch 32). Export ở kích thước khác sẽ làm giảm
  /// độ chính xác.
  static const String assetPath = 'assets/models/shrimp_yolo11n_512.tflite';

  static const int inputSize = 512;
  static const int numClasses = 2;

  /// 5376 = 64² + 32² + 16², tức feature map ở stride 8/16/32 của ảnh 512.
  /// Đã xác nhận khi export: output shape (1, 6, 5376).
  static const int numAnchors = 5376;

  /// Giữ trùng ngưỡng với server để hai đường cho kết quả nhất quán
  /// (`INFER_IOU = 0.60` tại app/main.py).
  static const double defaultConfThreshold = 0.25;
  static const double defaultIouThreshold = 0.60;

  /// Màu pad khi letterbox — quy ước của Ultralytics.
  static const int padGray = 114;
}
