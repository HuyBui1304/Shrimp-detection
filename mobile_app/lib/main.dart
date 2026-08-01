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
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF0EA5E9),
          brightness: Brightness.light,
        ),
        useMaterial3: true,
      ),
      darkTheme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF0EA5E9),
          brightness: Brightness.dark,
        ),
        useMaterial3: true,
      ),
      home: const DetectScreen(),
    );
  }
}
