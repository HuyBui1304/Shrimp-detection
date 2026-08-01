import 'package:flutter/material.dart';

import '../core/constants.dart';
import '../models/detection.dart';

/// Vẽ box lên trên ảnh.
///
/// Toạ độ trong [Detection] theo pixel ảnh gốc, còn canvas thì đúng bằng kích
/// thước ảnh đang hiển thị trên màn hình, nên phải nhân tỉ lệ. Widget cha bọc
/// trong AspectRatio đúng tỉ lệ ảnh nên hai trục cùng hệ số.
class DetectionPainter extends CustomPainter {
  final List<Detection> detections;
  final int imageWidth;
  final int imageHeight;
  final Brightness brightness;

  const DetectionPainter({
    required this.detections,
    required this.imageWidth,
    required this.imageHeight,
    required this.brightness,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (imageWidth == 0 || imageHeight == 0) return;

    final scaleX = size.width / imageWidth;
    final scaleY = size.height / imageHeight;
    final stroke = (size.shortestSide * 0.007).clamp(2.0, 4.0);

    // Vẽ toàn bộ box trước, nhãn sau — nếu vẽ xen kẽ thì box của con sau sẽ
    // cắt ngang nhãn của con trước.
    final rects = <Rect>[];
    for (final detection in detections) {
      final color = ShrimpClass.colorOf(detection.classId, brightness);
      final rect = Rect.fromLTRB(
        detection.box.left * scaleX,
        detection.box.top * scaleY,
        detection.box.right * scaleX,
        detection.box.bottom * scaleY,
      );
      rects.add(rect);
      final rounded = RRect.fromRectAndRadius(rect, const Radius.circular(6));

      // Viền tối mảnh bao ngoài để box không chìm vào nền cùng tông — ảnh chụp
      // tôm dưới nước rất hay trùng màu với đường kẻ.
      canvas.drawRRect(
        rounded.inflate(stroke / 2),
        Paint()
          ..style = PaintingStyle.stroke
          ..strokeWidth = stroke
          ..color = Colors.black.withValues(alpha: 0.25),
      );
      canvas.drawRRect(
        rounded,
        Paint()
          ..style = PaintingStyle.stroke
          ..strokeWidth = stroke
          ..color = color,
      );
    }

    final placed = <Rect>[];
    for (var i = 0; i < detections.length; i++) {
      _paintLabel(canvas, size, rects[i], detections[i], placed);
    }
  }

  void _paintLabel(
    Canvas canvas,
    Size size,
    Rect box,
    Detection detection,
    List<Rect> placed,
  ) {
    final color = ShrimpClass.colorOf(detection.classId, brightness);
    final fontSize = (size.shortestSide * 0.034).clamp(10.0, 16.0);
    final painter = TextPainter(
      text: TextSpan(
        text: '${ShrimpClass.markOf(detection.classId)} ${detection.label}'
            '  ${(detection.score * 100).toStringAsFixed(0)}%',
        style: TextStyle(
          color: Colors.white,
          fontSize: fontSize,
          fontWeight: FontWeight.w700,
          letterSpacing: 0.2,
        ),
      ),
      textDirection: TextDirection.ltr,
    )..layout();

    const padH = 7.0;
    const padV = 3.5;
    final w = painter.width + padH * 2;
    final h = painter.height + padV * 2;

    final maxLeft = (size.width - w).clamp(0.0, size.width);
    final left = box.left.clamp(0.0, maxLeft);

    // Con tôm hay nằm sát nhau nên nhãn rất dễ đè lên nhau và che mất chữ. Thử
    // lần lượt: ngay trên box, trong lòng box, rồi đẩy dần xuống — lấy vị trí
    // trống đầu tiên. Hết cách thì vẫn vẽ ở chỗ ưu tiên, thà chồng còn hơn mất.
    final candidates = <double>[
      box.top - h - 2,
      box.top + 2,
      for (var k = 1; k <= 4; k++) box.top + 2 + k * (h + 2),
    ];

    var top = candidates.first;
    for (final candidate in candidates) {
      if (candidate < 0 || candidate + h > size.height) continue;
      final trial = Rect.fromLTWH(left, candidate, w, h);
      if (placed.every((r) => !r.overlaps(trial))) {
        top = candidate;
        break;
      }
    }
    top = top.clamp(0.0, (size.height - h).clamp(0.0, size.height));

    final rect = Rect.fromLTWH(left, top, w, h);
    placed.add(rect);

    canvas.drawRRect(
      RRect.fromRectAndRadius(rect, const Radius.circular(5)),
      Paint()..color = color,
    );
    painter.paint(canvas, Offset(left + padH, top + padV));
  }

  @override
  bool shouldRepaint(DetectionPainter old) =>
      old.detections != detections ||
      old.imageWidth != imageWidth ||
      old.imageHeight != imageHeight ||
      old.brightness != brightness;
}
