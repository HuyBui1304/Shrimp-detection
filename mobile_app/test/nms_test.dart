import 'dart:ui' show Rect;

import 'package:flutter_test/flutter_test.dart';
import 'package:shrimp_detector/models/detection.dart';
import 'package:shrimp_detector/services/nms.dart';

/// Dựng tensor giả theo đúng layout đã xác nhận lúc export:
/// [1, 6, N] channels-first, toạ độ pixel.
List<double> buildFlat(List<List<double>> anchors) {
  const channels = 6;
  final n = anchors.length;
  final flat = List<double>.filled(channels * n, 0.0);
  for (var a = 0; a < n; a++) {
    for (var c = 0; c < channels; c++) {
      flat[c * n + a] = anchors[a][c];
    }
  }
  return flat;
}

void main() {
  group('iou', () {
    test('box trùng khít bằng 1', () {
      const box = Rect.fromLTRB(0, 0, 10, 10);
      expect(iou(box, box), closeTo(1.0, 1e-9));
    });

    test('box rời nhau bằng 0', () {
      expect(
        iou(const Rect.fromLTRB(0, 0, 10, 10), const Rect.fromLTRB(20, 20, 30, 30)),
        0.0,
      );
    });

    test('chồng một nửa', () {
      // Giao = 10x20 = 200; hợp = 400 + 400 - 200 = 600
      final value = iou(
        const Rect.fromLTRB(0, 0, 20, 20),
        const Rect.fromLTRB(10, 0, 30, 20),
      );
      expect(value, closeTo(200 / 600, 1e-9));
    });

    test('chạm mép nhưng không chồng thì bằng 0', () {
      expect(
        iou(const Rect.fromLTRB(0, 0, 10, 10), const Rect.fromLTRB(10, 0, 20, 10)),
        0.0,
      );
    });
  });

  group('decodeYolo', () {
    test('đọc đúng layout [1, 6, N] và lọc theo ngưỡng', () {
      // anchor 0: điểm cao, lớp 1 (bệnh)  | anchor 1: điểm thấp, bị loại
      final flat = buildFlat([
        [100, 100, 40, 20, 0.10, 0.90],
        [200, 200, 40, 20, 0.20, 0.15],
      ]);

      final result = decodeYolo(
        flat: flat,
        dimA: 6,
        dimB: 2,
        numClasses: 2,
        confThreshold: 0.25,
        inputSize: 512,
      );

      expect(result, hasLength(1));
      expect(result.single.classId, 1);
      expect(result.single.score, closeTo(0.90, 1e-9));
      // cx=100, w=40 -> left=80, right=120
      expect(result.single.box, const Rect.fromLTRB(80, 90, 120, 110));
    });

    test('chọn lớp có điểm cao nhất', () {
      final flat = buildFlat([
        [50, 50, 10, 10, 0.80, 0.30],
      ]);
      final result = decodeYolo(
        flat: flat,
        dimA: 6,
        dimB: 1,
        numClasses: 2,
        confThreshold: 0.25,
        inputSize: 512,
      );
      expect(result.single.classId, 0);
    });

    test('cũng đọc được layout hoán trục [1, N, 6]', () {
      // Cùng dữ liệu nhưng xếp anchor-major.
      final flat = <double>[100, 100, 40, 20, 0.10, 0.90];
      final result = decodeYolo(
        flat: flat,
        dimA: 1,
        dimB: 6,
        numClasses: 2,
        confThreshold: 0.25,
        inputSize: 512,
      );
      expect(result.single.box, const Rect.fromLTRB(80, 90, 120, 110));
    });

    test('tự nhân kích thước khi toạ độ ở dạng chuẩn hoá', () {
      // Mọi giá trị <= 1 nên phải được hiểu là tỉ lệ và nhân với inputSize.
      final flat = buildFlat([
        [0.5, 0.5, 0.25, 0.25, 0.05, 0.99],
      ]);
      final result = decodeYolo(
        flat: flat,
        dimA: 6,
        dimB: 1,
        numClasses: 2,
        confThreshold: 0.25,
        inputSize: 512,
      );
      // cx = 0.5*512 = 256, w = 0.25*512 = 128 -> left = 192
      expect(result.single.box.left, closeTo(192, 1e-6));
      expect(result.single.box.right, closeTo(320, 1e-6));
    });

    test('báo lỗi khi không chiều nào khớp số kênh', () {
      expect(
        () => decodeYolo(
          flat: List.filled(21, 0.0),
          dimA: 3,
          dimB: 7,
          numClasses: 2,
          confThreshold: 0.25,
          inputSize: 512,
        ),
        throwsStateError,
      );
    });
  });

  group('nonMaxSuppression', () {
    test('giữ box điểm cao, loại box chồng cùng lớp', () {
      final input = [
        const Detection(box: Rect.fromLTRB(0, 0, 100, 100), score: 0.9, classId: 0),
        const Detection(box: Rect.fromLTRB(5, 5, 105, 105), score: 0.7, classId: 0),
        const Detection(box: Rect.fromLTRB(500, 500, 600, 600), score: 0.6, classId: 0),
      ];

      final kept = nonMaxSuppression(input, 0.6);
      expect(kept, hasLength(2));
      expect(kept.first.score, 0.9);
    });

    test('không loại chéo giữa hai lớp khác nhau', () {
      // Một con tôm khoẻ và một con tôm bệnh chồng nhau thì giữ cả hai,
      // đúng theo mặc định agnostic=False của Ultralytics.
      final input = [
        const Detection(box: Rect.fromLTRB(0, 0, 100, 100), score: 0.9, classId: 0),
        const Detection(box: Rect.fromLTRB(0, 0, 100, 100), score: 0.8, classId: 1),
      ];

      expect(nonMaxSuppression(input, 0.5), hasLength(2));
    });

    test('trả về danh sách xếp theo điểm giảm dần', () {
      final input = [
        const Detection(box: Rect.fromLTRB(0, 0, 10, 10), score: 0.3, classId: 0),
        const Detection(box: Rect.fromLTRB(50, 50, 60, 60), score: 0.9, classId: 1),
        const Detection(box: Rect.fromLTRB(90, 90, 99, 99), score: 0.6, classId: 0),
      ];

      final scores = nonMaxSuppression(input, 0.5).map((d) => d.score).toList();
      expect(scores, [0.9, 0.6, 0.3]);
    });

    test('danh sách rỗng trả về rỗng', () {
      expect(nonMaxSuppression(const [], 0.5), isEmpty);
    });
  });
}
