import 'dart:math' as math;
import 'dart:ui' show Rect;

import '../models/detection.dart';

/// Giải mã tensor đầu ra của YOLOv11 rồi khử box trùng.
///
/// YOLOv11 xuất `[1, 4+nc, N]` (channels-first, khác YOLOv5 vốn là `[1, N, ...]`).
/// Nhưng khi đi qua onnx2tf để ra TFLite, trục có thể bị hoán thành `[1, N, 4+nc]`,
/// và toạ độ có thể ở dạng chuẩn hoá [0,1] thay vì pixel. Hàm dưới đây tự dò cả
/// hai khác biệt đó thay vì giả định — sai một trong hai thì box sẽ lệch hoàn toàn
/// mà không có thông báo lỗi nào.
List<Detection> decodeYolo({
  required List<double> flat,
  required int dimA,
  required int dimB,
  required int numClasses,
  required double confThreshold,
  required int inputSize,
}) {
  final int channels = 4 + numClasses;

  final bool channelsFirst;
  final int numAnchors;
  if (dimA == channels) {
    channelsFirst = true;
    numAnchors = dimB;
  } else if (dimB == channels) {
    channelsFirst = false;
    numAnchors = dimA;
  } else {
    throw StateError(
      'Không nhận ra layout đầu ra [1, $dimA, $dimB]. '
      'Với $numClasses lớp thì một trong hai chiều phải bằng $channels.',
    );
  }

  double read(int anchor, int channel) => channelsFirst
      ? flat[channel * numAnchors + anchor]
      : flat[anchor * channels + channel];

  // Dò xem toạ độ đang ở dạng chuẩn hoá hay pixel: quét chiều rộng của một
  // lượng mẫu anchor, nếu không có giá trị nào vượt 1.5 thì coi là chuẩn hoá.
  double maxExtent = 0.0;
  final probe = math.min(numAnchors, 512);
  for (int i = 0; i < probe; i++) {
    maxExtent = math.max(maxExtent, read(i, 2).abs());
  }
  final double coordScale = maxExtent <= 1.5 ? inputSize.toDouble() : 1.0;

  final results = <Detection>[];
  for (int i = 0; i < numAnchors; i++) {
    int bestClass = 0;
    double bestScore = read(i, 4);
    for (int c = 1; c < numClasses; c++) {
      final s = read(i, 4 + c);
      if (s > bestScore) {
        bestScore = s;
        bestClass = c;
      }
    }
    if (bestScore < confThreshold) continue;

    final cx = read(i, 0) * coordScale;
    final cy = read(i, 1) * coordScale;
    final w = read(i, 2) * coordScale;
    final h = read(i, 3) * coordScale;

    results.add(Detection(
      box: Rect.fromLTRB(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2),
      score: bestScore,
      classId: bestClass,
    ));
  }
  return results;
}

/// NMS theo từng lớp riêng (khớp mặc định `agnostic=False` của Ultralytics —
/// một con tôm khoẻ và một con tôm bệnh chồng nhau thì giữ cả hai).
List<Detection> nonMaxSuppression(
  List<Detection> detections,
  double iouThreshold,
) {
  final byClass = <int, List<Detection>>{};
  for (final d in detections) {
    byClass.putIfAbsent(d.classId, () => <Detection>[]).add(d);
  }

  final kept = <Detection>[];
  for (final group in byClass.values) {
    group.sort((a, b) => b.score.compareTo(a.score));
    final selected = <Detection>[];
    for (final candidate in group) {
      var suppressed = false;
      for (final chosen in selected) {
        if (iou(candidate.box, chosen.box) > iouThreshold) {
          suppressed = true;
          break;
        }
      }
      if (!suppressed) selected.add(candidate);
    }
    kept.addAll(selected);
  }

  kept.sort((a, b) => b.score.compareTo(a.score));
  return kept;
}

double iou(Rect a, Rect b) {
  final left = math.max(a.left, b.left);
  final top = math.max(a.top, b.top);
  final right = math.min(a.right, b.right);
  final bottom = math.min(a.bottom, b.bottom);

  if (right <= left || bottom <= top) return 0.0;

  final intersection = (right - left) * (bottom - top);
  final union = a.width * a.height + b.width * b.height - intersection;
  return union <= 0 ? 0.0 : intersection / union;
}
