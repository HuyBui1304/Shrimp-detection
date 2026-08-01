import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:integration_test/integration_test.dart';
import 'package:shrimp_detector/core/constants.dart';
import 'package:shrimp_detector/services/yolo_detector.dart';

/// Chạy trên thiết bị/emulator thật:
///
///     flutter test integration_test/detector_test.dart -d emulator-5554
///
/// Các test trong test/ chỉ kiểm phần toán học tách rời. Bài này mới chứng
/// minh được toàn bộ chuỗi thật — nạp file .tflite, letterbox, chạy model,
/// giải mã, NMS — cho ra đúng kết quả như bản PyTorch gốc.
///
/// Số liệu mong đợi lấy từ scripts/verify_tflite.py trên chính ảnh này:
///     lớp 0 (Tôm khỏe) conf 0.9483  box (174.3, 219.3, 512.0, 392.9)
///     lớp 1 (Tôm bệnh) conf 0.9229  box (145.7, 231.2, 269.3, 350.5)
///     lớp 1 (Tôm bệnh) conf 0.9047  box (  0.2, 245.2,  63.6, 327.3)
void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  late YoloDetector detector;
  late img.Image sample;

  setUpAll(() async {
    detector = YoloDetector();
    await detector.load();

    final bytes = await rootBundle.load('assets/test/sample.jpg');
    sample = img.decodeImage(bytes.buffer.asUint8List())!;
  });

  tearDownAll(() => detector.dispose());

  testWidgets('model nạp đúng hình dạng tensor đã export', (_) async {
    expect(detector.isLoaded, isTrue);
    expect(detector.inputSize, ModelConfig.inputSize);
    expect(detector.outputShape, [1, 4 + ModelConfig.numClasses, ModelConfig.numAnchors]);
  });

  testWidgets('ảnh mẫu cho ra đúng 3 con: 1 khỏe, 2 bệnh', (_) async {
    final result = detector.detect(sample);

    expect(result.total, 3);
    expect(result.healthyCount, 1);
    expect(result.diseasedCount, 2);
    expect(result.diseaseRate, closeTo(66.7, 0.1));
  });

  testWidgets('toạ độ và điểm số khớp bản PyTorch', (_) async {
    final result = detector.detect(sample);
    final expected = [
      (cls: 0, conf: 0.9483, box: [174.3, 219.3, 512.0, 392.9]),
      (cls: 1, conf: 0.9229, box: [145.7, 231.2, 269.3, 350.5]),
      (cls: 1, conf: 0.9047, box: [0.2, 245.2, 63.6, 327.3]),
    ];

    // detect() trả về danh sách đã xếp theo điểm giảm dần.
    for (var i = 0; i < expected.length; i++) {
      final actual = result.detections[i];
      final want = expected[i];

      expect(actual.classId, want.cls, reason: 'sai lớp ở box $i');
      expect(actual.score, closeTo(want.conf, 0.03), reason: 'lệch điểm số ở box $i');

      // Dung sai 3 px: TFLite trên Android dùng nhân XNNPACK khác bản trên
      // máy tính nên có sai số nhỏ. Lệch quá ngưỡng này nghĩa là letterbox
      // hoặc phần ánh xạ ngược toạ độ có lỗi thật.
      expect(actual.box.left, closeTo(want.box[0], 3.0), reason: 'box $i lệch trái');
      expect(actual.box.top, closeTo(want.box[1], 3.0), reason: 'box $i lệch trên');
      expect(actual.box.right, closeTo(want.box[2], 3.0), reason: 'box $i lệch phải');
      expect(actual.box.bottom, closeTo(want.box[3], 3.0), reason: 'box $i lệch dưới');
    }
  });

  testWidgets('mọi box nằm gọn trong khung ảnh', (_) async {
    final result = detector.detect(sample);
    for (final d in result.detections) {
      expect(d.box.left, greaterThanOrEqualTo(0));
      expect(d.box.top, greaterThanOrEqualTo(0));
      expect(d.box.right, lessThanOrEqualTo(sample.width.toDouble()));
      expect(d.box.bottom, lessThanOrEqualTo(sample.height.toDouble()));
    }
  });

  testWidgets('nâng ngưỡng thì số box giảm dần, không tăng', (_) async {
    final counts = [0.10, 0.25, 0.50, 0.95]
        .map((t) => detector.detect(sample, confThreshold: t).total)
        .toList();

    for (var i = 1; i < counts.length; i++) {
      expect(counts[i], lessThanOrEqualTo(counts[i - 1]),
          reason: 'ngưỡng cao hơn mà lại ra nhiều box hơn: $counts');
    }
    expect(counts.last, 0, reason: 'ngưỡng 0.95 lẽ ra không giữ con nào');
  });
}
