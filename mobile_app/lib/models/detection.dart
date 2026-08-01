import 'dart:ui' show Rect;

import '../core/constants.dart';

/// Một box đã qua lọc ngưỡng và NMS.
///
/// [box] luôn ở toạ độ **pixel của ảnh gốc**, không phải toạ độ 512×512 của
/// model — việc quy đổi do LetterboxResult.toOriginal đảm nhiệm.
class Detection {
  final Rect box;
  final double score;
  final int classId;

  const Detection({
    required this.box,
    required this.score,
    required this.classId,
  });

  String get label => ShrimpClass.labelOf(classId);
  bool get isDiseased => classId == ShrimpClass.diseased;

  Detection copyWith({Rect? box, double? score, int? classId}) => Detection(
        box: box ?? this.box,
        score: score ?? this.score,
        classId: classId ?? this.classId,
      );

  @override
  String toString() =>
      'Detection($label ${(score * 100).toStringAsFixed(1)}% $box)';
}

/// Kết quả một lần chạy suy luận, kèm số liệu để hiển thị lên panel.
class DetectionResult {
  final List<Detection> detections;
  final int inferenceMs;
  final int srcWidth;
  final int srcHeight;

  const DetectionResult({
    required this.detections,
    required this.inferenceMs,
    required this.srcWidth,
    required this.srcHeight,
  });

  int get healthyCount =>
      detections.where((d) => d.classId == ShrimpClass.healthy).length;

  int get diseasedCount =>
      detections.where((d) => d.classId == ShrimpClass.diseased).length;

  int get total => detections.length;

  /// Tỉ lệ tôm bệnh trên tổng số con phát hiện được, đơn vị phần trăm.
  double get diseaseRate => total == 0 ? 0.0 : diseasedCount * 100.0 / total;

  static const empty = DetectionResult(
    detections: [],
    inferenceMs: 0,
    srcWidth: 0,
    srcHeight: 0,
  );
}
