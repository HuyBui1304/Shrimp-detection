import 'package:flutter/material.dart';

import 'screens/detect_screen.dart';

void main() {
  runApp(const ShrimpDetectorApp());
}

class ShrimpDetectorApp extends StatelessWidget {
  const ShrimpDetectorApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Nhận diện tôm bệnh',
      debugShowCheckedModeBanner: false,
      // Màu chủ đạo cố tình chọn tông xanh mòng két, tách khỏi cả xanh dương
      // (tôm khỏe) lẫn đỏ (tôm bệnh) trong lib/core/constants.dart. Nếu màu
      // giao diện trùng tông với màu phân loại thì nút bấm và ô chỉ số trông
      // như đang mang nghĩa dữ liệu.
      theme: _theme(Brightness.light),
      darkTheme: _theme(Brightness.dark),
      home: const DetectScreen(),
    );
  }

  static ThemeData _theme(Brightness brightness) {
    final scheme = ColorScheme.fromSeed(
      seedColor: const Color(0xFF0F766E),
      brightness: brightness,
    );
    return ThemeData(
      colorScheme: scheme,
      useMaterial3: true,
      appBarTheme: AppBarTheme(
        backgroundColor: scheme.surface,
        surfaceTintColor: Colors.transparent,
      ),
      filledButtonTheme: FilledButtonThemeData(
        style: FilledButton.styleFrom(
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(14),
          ),
          textStyle: const TextStyle(
            fontWeight: FontWeight.w600,
            fontSize: 15,
          ),
        ),
      ),
    );
  }
}
