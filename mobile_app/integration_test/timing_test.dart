import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:integration_test/integration_test.dart';
import 'package:shrimp_detector/services/yolo_detector.dart';

/// Đo xem 400+ ms mỗi ảnh thực sự tiêu ở đâu.
///
///     flutter test integration_test/timing_test.dart -d <thiết bị>
///
/// Câu hỏi cần trả lời trước khi làm camera realtime: phần lớn thời gian nằm ở
/// khâu chạy model hay ở khâu dựng tensor? Model chạy nhanh hơn khi đổi máy
/// mạnh hơn; còn việc dựng List lồng 4 tầng với 786.432 số double đóng hộp thì
/// là chi phí cấp phát của Dart, đổi máy gần như không giúp gì.
void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('phân rã thời gian từng khâu', (_) async {
    final detector = YoloDetector();
    await detector.load();

    final bytes = await rootBundle.load('assets/test/sample.jpg');
    final sample = img.decodeImage(bytes.buffer.asUint8List())!;

    // Bỏ 3 lần đầu cho JIT và bộ nhớ đệm ổn định.
    for (var i = 0; i < 3; i++) {
      detector.detect(sample);
    }

    const runs = 10;
    var letterbox = 0, tensor = 0, model = 0, post = 0;
    for (var i = 0; i < runs; i++) {
      final t = detector.detect(sample).timings;
      letterbox += t.letterboxUs;
      tensor += t.tensorUs;
      model += t.modelUs;
      post += t.postprocessUs;
    }

    final total = letterbox + tensor + model + post;
    String row(String name, int us) {
      final ms = us / runs / 1000;
      final pct = us * 100 / total;
      return '  ${name.padRight(16)} ${ms.toStringAsFixed(1).padLeft(7)} ms  '
          '${pct.toStringAsFixed(1).padLeft(5)}%';
    }

    // ignore: avoid_print
    print('\n=== Trung bình $runs lần chạy trên ${sample.width}x${sample.height} ===\n'
        '${row('letterbox', letterbox)}\n'
        '${row('dựng tensor', tensor)}\n'
        '${row('chạy model', model)}\n'
        '${row('giải mã + NMS', post)}\n'
        '  ${'-' * 32}\n'
        '${row('TỔNG', total)}\n');

    detector.dispose();
    expect(total, greaterThan(0));
  });
}
