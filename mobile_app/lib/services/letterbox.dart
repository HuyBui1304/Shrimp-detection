import 'dart:math' as math;
import 'dart:ui' show Rect;

import 'package:image/image.dart' as img;

/// Kết quả letterbox: ảnh vuông đã pad, kèm các tham số để ánh xạ ngược
/// toạ độ box từ không gian model về không gian ảnh gốc.
///
/// Đây là chỗ dễ sai nhất trong toàn bộ pipeline. Nếu box vẽ ra bị lệch
/// hoặc bị co, gần như chắc chắn lỗi nằm ở đây chứ không phải ở model.
class LetterboxResult {
  final img.Image image;
  final double scale;
  final double padX;
  final double padY;
  final int srcWidth;
  final int srcHeight;

  const LetterboxResult({
    required this.image,
    required this.scale,
    required this.padX,
    required this.padY,
    required this.srcWidth,
    required this.srcHeight,
  });

  /// Đưa một box từ toạ độ ảnh model (đã pad, vuông) về toạ độ ảnh gốc,
  /// đồng thời cắt cho nằm gọn trong khung ảnh.
  Rect toOriginal(Rect boxInModelSpace) {
    final left = (boxInModelSpace.left - padX) / scale;
    final top = (boxInModelSpace.top - padY) / scale;
    final right = (boxInModelSpace.right - padX) / scale;
    final bottom = (boxInModelSpace.bottom - padY) / scale;

    return Rect.fromLTRB(
      left.clamp(0.0, srcWidth.toDouble()),
      top.clamp(0.0, srcHeight.toDouble()),
      right.clamp(0.0, srcWidth.toDouble()),
      bottom.clamp(0.0, srcHeight.toDouble()),
    );
  }
}

/// Resize giữ nguyên tỉ lệ rồi pad cho vuông, pad được chia đều hai bên
/// (đúng quy ước `center=True` của Ultralytics).
LetterboxResult letterbox(
  img.Image src, {
  int size = 512,
  int padColor = 114,
}) {
  final scale = math.min(size / src.width, size / src.height);
  final newW = (src.width * scale).round();
  final newH = (src.height * scale).round();

  final resized = img.copyResize(
    src,
    width: newW,
    height: newH,
    interpolation: img.Interpolation.linear,
  );

  final canvas = img.Image(width: size, height: size, numChannels: 3);
  img.fill(canvas, color: img.ColorRgb8(padColor, padColor, padColor));

  // Phải làm tròn XUỐNG, không dùng round(). Khi phần cần pad là số lẻ —
  // ví dụ ảnh 777x333 thu còn 512x219 thì thừa 293 px — Ultralytics đặt 146 px
  // ở trên và 147 px ở dưới, còn round(146.5) cho ra 147 ở trên. Lệch một
  // pixel ở đây làm box dịch vài pixel sau khi ánh xạ ngược.
  final dx = (size - newW) ~/ 2;
  final dy = (size - newH) ~/ 2;
  img.compositeImage(canvas, resized, dstX: dx, dstY: dy);

  return LetterboxResult(
    image: canvas,
    scale: scale,
    padX: dx.toDouble(),
    padY: dy.toDouble(),
    srcWidth: src.width,
    srcHeight: src.height,
  );
}

/// Chuyển ảnh đã letterbox thành tensor NHWC float32 chuẩn hoá về [0, 1].
List<List<List<List<double>>>> toInputTensor(img.Image image) {
  return [
    List.generate(
      image.height,
      (y) => List.generate(image.width, (x) {
        final p = image.getPixel(x, y);
        return [p.rNormalized.toDouble(), p.gNormalized.toDouble(), p.bNormalized.toDouble()];
      }),
    ),
  ];
}
