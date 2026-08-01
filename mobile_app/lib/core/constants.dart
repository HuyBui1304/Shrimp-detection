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

  /// Ký hiệu đi kèm nhãn trên box. Màu sắc không bao giờ được là kênh thông tin
  /// duy nhất — xem ghi chú về mù màu ở phần màu bên dưới.
  static const List<String> marks = ['✓', '!'];

  static const List<IconData> icons = [
    Icons.check_circle_rounded,
    Icons.error_rounded,
  ];

  /// Xanh dương cho tôm khỏe thay vì xanh lá.
  ///
  /// Cặp xanh lá/đỏ quen thuộc là cái bẫy mù màu kinh điển: đo bằng OKLab ΔE
  /// với ma trận mô phỏng Machado 2009, cặp `#22C55E`/`#EF4444` chỉ đạt **7.4**
  /// ở dạng deuteranopia (ngưỡng an toàn là 8), còn cặp xanh lá/đỏ "chuẩn" của
  /// hệ thống thiết kế thậm chí chỉ 4.1. Khoảng 8% nam giới bị deuteranopia,
  /// và đây là app dùng để phân biệt tôm bệnh với tôm khỏe — nhầm lẫn ở đây là
  /// nhầm lẫn có hậu quả.
  ///
  /// Cặp xanh dương/đỏ dưới đây đạt **ΔE 23.8**, dư an toàn, mà vẫn giữ được ý
  /// nghĩa "đỏ là cảnh báo". Tương phản trên nền sáng 4.30/4.68 và nền tối
  /// 3.94/3.62, đều vượt mức 3.0.
  static const List<Color> _light = [Color(0xFF2A78D6), Color(0xFFD03B3B)];
  static const List<Color> _dark = [Color(0xFF3987E5), Color(0xFFD03B3B)];

  static String labelOf(int id) =>
      (id >= 0 && id < labels.length) ? labels[id] : 'Không rõ ($id)';

  static String markOf(int id) => (id >= 0 && id < marks.length) ? marks[id] : '?';

  static IconData iconOf(int id) =>
      (id >= 0 && id < icons.length) ? icons[id] : Icons.help_rounded;

  static Color colorOf(int id, Brightness brightness) {
    final palette = brightness == Brightness.dark ? _dark : _light;
    return (id >= 0 && id < palette.length) ? palette[id] : Colors.grey;
  }
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
