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

  const DetectionPainter({
    required this.detections,
    required this.imageWidth,
    required this.imageHeight,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (imageWidth == 0 || imageHeight == 0) return;

    final scaleX = size.width / imageWidth;
    final scaleY = size.height / imageHeight;
    final strokeWidth = (size.shortestSide * 0.006).clamp(1.5, 4.0);

    for (final detection in detections) {
      final color = ShrimpClass.colorOf(detection.classId);
      final rect = Rect.fromLTRB(
        detection.box.left * scaleX,
        detection.box.top * scaleY,
        detection.box.right * scaleX,
        detection.box.bottom * scaleY,
      );

      canvas.drawRect(
        rect,
        Paint()
          ..style = PaintingStyle.stroke
          ..strokeWidth = strokeWidth
          ..color = color,
      );

      _paintLabel(
        canvas,
        size,
        rect,
        color,
        '${detection.label} ${(detection.score * 100).toStringAsFixed(0)}%',
      );
    }
  }

  void _paintLabel(
    Canvas canvas,
    Size size,
    Rect box,
    Color color,
    String text,
  ) {
    final fontSize = (size.shortestSide * 0.032).clamp(9.0, 15.0);
    final painter = TextPainter(
      text: TextSpan(
        text: text,
        style: TextStyle(
          color: Colors.white,
          fontSize: fontSize,
          fontWeight: FontWeight.w600,
        ),
      ),
      textDirection: TextDirection.ltr,
    )..layout();

    const padH = 4.0;
    const padV = 2.0;
    final labelWidth = painter.width + padH * 2;
    final labelHeight = painter.height + padV * 2;

    // Đặt nhãn phía trên box; nếu box sát mép trên thì lật xuống dưới để
    // nhãn không bị cắt mất.
    final fitsAbove = box.top - labelHeight >= 0;
    final top = fitsAbove ? box.top - labelHeight : box.top;
    final left = box.left.clamp(0.0, (size.width - labelWidth).clamp(0.0, size.width));

    final background = Rect.fromLTWH(left, top, labelWidth, labelHeight);
    canvas.drawRect(background, Paint()..color = color);
    painter.paint(canvas, Offset(left + padH, top + padV));
  }

  @override
  bool shouldRepaint(DetectionPainter old) =>
      old.detections != detections ||
      old.imageWidth != imageWidth ||
      old.imageHeight != imageHeight;
}
