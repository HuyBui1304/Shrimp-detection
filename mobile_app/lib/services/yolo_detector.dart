import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

import '../core/constants.dart';
import '../models/detection.dart';
import 'letterbox.dart';
import 'nms.dart';

/// Bọc TFLite interpreter và toàn bộ chuỗi tiền/hậu xử lý.
///
/// Hình dạng tensor đã được xác nhận lúc export (scripts/export_tflite.py):
///   input  [1, 512, 512, 3] float32, giá trị [0,1]
///   output [1, 6, 5376]     float32, toạ độ theo PIXEL của ảnh 512 (không chuẩn hoá)
/// Dù vậy các chiều vẫn được đọc lại từ file lúc chạy, để nếu sau này export
/// lại ở kích thước khác thì app không cần sửa code.
class YoloDetector {
  Interpreter? _interpreter;
  List<int> _inputShape = const [1, ModelConfig.inputSize, ModelConfig.inputSize, 3];
  List<int> _outputShape = const [1, 4 + ModelConfig.numClasses, ModelConfig.numAnchors];

  bool get isLoaded => _interpreter != null;
  int get inputSize => _inputShape[1];
  List<int> get outputShape => List.unmodifiable(_outputShape);

  Future<void> load() async {
    if (_interpreter != null) return;
    final options = InterpreterOptions()..threads = 4;
    final interpreter = await Interpreter.fromAsset(
      ModelConfig.assetPath,
      options: options,
    );
    _inputShape = interpreter.getInputTensor(0).shape;
    _outputShape = interpreter.getOutputTensor(0).shape;
    _interpreter = interpreter;
  }

  void dispose() {
    _interpreter?.close();
    _interpreter = null;
  }

  DetectionResult detect(
    img.Image source, {
    double confThreshold = ModelConfig.defaultConfThreshold,
    double iouThreshold = ModelConfig.defaultIouThreshold,
  }) {
    final interpreter = _interpreter;
    if (interpreter == null) {
      throw StateError('Chưa gọi load() trước khi nhận diện');
    }

    final stopwatch = Stopwatch()..start();

    final prepared = letterbox(
      source,
      size: inputSize,
      padColor: ModelConfig.padGray,
    );
    final input = toInputTensor(prepared.image);

    final channels = _outputShape[1];
    final anchors = _outputShape[2];
    final output = [
      List.generate(channels, (_) => List<double>.filled(anchors, 0.0)),
    ];

    interpreter.run(input, output);

    // Duỗi phẳng theo đúng thứ tự [kênh][anchor] mà decodeYolo mong đợi.
    final flat = List<double>.filled(channels * anchors, 0.0);
    for (var c = 0; c < channels; c++) {
      final row = output[0][c];
      final base = c * anchors;
      for (var a = 0; a < anchors; a++) {
        flat[base + a] = row[a];
      }
    }

    final decoded = decodeYolo(
      flat: flat,
      dimA: channels,
      dimB: anchors,
      numClasses: ModelConfig.numClasses,
      confThreshold: confThreshold,
      inputSize: inputSize,
    );

    final kept = nonMaxSuppression(decoded, iouThreshold);

    // Box đang ở toạ độ ảnh 512 đã pad; đưa về toạ độ ảnh gốc.
    final mapped = kept
        .map((d) => d.copyWith(box: prepared.toOriginal(d.box)))
        .where((d) => d.box.width > 1 && d.box.height > 1)
        .toList();

    stopwatch.stop();

    return DetectionResult(
      detections: mapped,
      inferenceMs: stopwatch.elapsedMilliseconds,
      srcWidth: source.width,
      srcHeight: source.height,
    );
  }
}
