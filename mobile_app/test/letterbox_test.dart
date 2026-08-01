import 'dart:ui' show Rect;

import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:shrimp_detector/services/letterbox.dart';

void main() {
  group('letterbox', () {
    test('ảnh ngang: pad trên/dưới, giữ nguyên tỉ lệ', () {
      final src = img.Image(width: 1000, height: 500);
      final result = letterbox(src, size: 512);

      expect(result.image.width, 512);
      expect(result.image.height, 512);
      expect(result.scale, closeTo(0.512, 1e-9));
      expect(result.padX, 0);
      // (512 - 500*0.512) / 2 = (512 - 256) / 2 = 128
      expect(result.padY, 128);
    });

    test('ảnh dọc: pad trái/phải', () {
      final src = img.Image(width: 500, height: 1000);
      final result = letterbox(src, size: 512);

      expect(result.scale, closeTo(0.512, 1e-9));
      expect(result.padX, 128);
      expect(result.padY, 0);
    });

    test('ảnh vuông: không pad', () {
      final result = letterbox(img.Image(width: 640, height: 640), size: 512);
      expect(result.padX, 0);
      expect(result.padY, 0);
      expect(result.scale, closeTo(0.8, 1e-9));
    });

    test('pad lẻ thì làm tròn xuống, khớp Ultralytics', () {
      // 777x333 thu nhỏ còn 512x219, thừa 293 px theo chiều dọc.
      // Ultralytics đặt 146 ở trên và 147 ở dưới; round(146.5) sẽ ra 147 và
      // làm box lệch vài pixel sau khi ánh xạ ngược.
      final result = letterbox(img.Image(width: 777, height: 333), size: 512);
      expect(result.padY, 146);
      expect(result.padX, 0);
    });

    test('toOriginal đảo ngược đúng phép biến đổi', () {
      final result = letterbox(img.Image(width: 1000, height: 500), size: 512);

      // Vùng ảnh thật nằm ở dải giữa: y từ 128 đến 384.
      final full = result.toOriginal(const Rect.fromLTRB(0, 128, 512, 384));
      expect(full.left, closeTo(0, 1e-6));
      expect(full.top, closeTo(0, 1e-6));
      expect(full.right, closeTo(1000, 1e-6));
      expect(full.bottom, closeTo(500, 1e-6));
    });

    test('toOriginal cắt box tràn ra ngoài khung ảnh', () {
      final result = letterbox(img.Image(width: 1000, height: 500), size: 512);

      // Box lấn vào vùng pad phía trên -> phải bị kẹp về 0, không âm.
      final clipped = result.toOriginal(const Rect.fromLTRB(-50, 0, 600, 500));
      expect(clipped.left, 0);
      expect(clipped.top, 0);
      expect(clipped.right, 1000);
      expect(clipped.bottom, 500);
    });
  });
}
