import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';

import '../core/constants.dart';
import '../models/detection.dart';
import '../services/yolo_detector.dart';
import '../widgets/detection_painter.dart';

class DetectScreen extends StatefulWidget {
  const DetectScreen({super.key});

  @override
  State<DetectScreen> createState() => _DetectScreenState();
}

class _DetectScreenState extends State<DetectScreen> {
  final _detector = YoloDetector();
  final _picker = ImagePicker();

  File? _imageFile;
  img.Image? _decoded;
  DetectionResult _result = DetectionResult.empty;

  double _confThreshold = ModelConfig.defaultConfThreshold;
  bool _busy = false;
  String? _error;

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  @override
  void dispose() {
    _detector.dispose();
    super.dispose();
  }

  Future<void> _loadModel() async {
    try {
      await _detector.load();
      if (mounted) setState(() {});
    } catch (e) {
      if (mounted) setState(() => _error = 'Không nạp được model: $e');
    }
  }

  Future<void> _pick(ImageSource source) async {
    final picked = await _picker.pickImage(source: source, imageQuality: 100);
    if (picked == null) return;

    setState(() {
      _busy = true;
      _error = null;
      _result = DetectionResult.empty;
      _imageFile = File(picked.path);
    });

    try {
      final bytes = await picked.readAsBytes();
      final decoded = img.decodeImage(bytes);
      if (decoded == null) {
        throw const FormatException('Không đọc được ảnh');
      }
      // Ảnh chụp từ điện thoại hay có cờ xoay trong EXIF; nếu không nắn lại
      // thì model nhận ảnh nằm ngang trong khi người dùng thấy ảnh dọc.
      _decoded = img.bakeOrientation(decoded);
      await _run();
    } catch (e) {
      if (mounted) setState(() => _error = 'Lỗi xử lý ảnh: $e');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _run() async {
    final source = _decoded;
    if (source == null || !_detector.isLoaded) return;
    final result = _detector.detect(source, confThreshold: _confThreshold);
    if (mounted) setState(() => _result = result);
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Scaffold(
      appBar: AppBar(
        title: const Text('Nhận diện tôm bệnh'),
        backgroundColor: theme.colorScheme.inversePrimary,
      ),
      body: SafeArea(
        child: Column(
          children: [
            Expanded(child: _buildPreview(theme)),
            _buildPanel(theme),
          ],
        ),
      ),
      floatingActionButton: Row(
        mainAxisAlignment: MainAxisAlignment.end,
        children: [
          FloatingActionButton.extended(
            heroTag: 'gallery',
            onPressed: _busy ? null : () => _pick(ImageSource.gallery),
            icon: const Icon(Icons.photo_library_outlined),
            label: const Text('Thư viện'),
          ),
          const SizedBox(width: 12),
          FloatingActionButton.extended(
            heroTag: 'camera',
            onPressed: _busy ? null : () => _pick(ImageSource.camera),
            icon: const Icon(Icons.photo_camera_outlined),
            label: const Text('Chụp'),
          ),
        ],
      ),
    );
  }

  Widget _buildPreview(ThemeData theme) {
    if (_error != null) {
      return _centered(
        icon: Icons.error_outline,
        color: theme.colorScheme.error,
        title: _error!,
      );
    }
    if (!_detector.isLoaded) {
      return _centered(
        icon: Icons.hourglass_empty,
        color: theme.colorScheme.primary,
        title: 'Đang nạp model...',
      );
    }
    final file = _imageFile;
    final decoded = _decoded;
    if (file == null || decoded == null) {
      return _centered(
        icon: Icons.image_search,
        color: theme.colorScheme.primary,
        title: 'Chọn một ảnh để bắt đầu',
        subtitle: 'Model yolo11n chạy ngay trên máy, không cần mạng',
      );
    }

    return Stack(
      children: [
        Center(
          child: AspectRatio(
            aspectRatio: decoded.width / decoded.height,
            child: Stack(
              fit: StackFit.expand,
              children: [
                Image.file(file, fit: BoxFit.fill),
                CustomPaint(
                  painter: DetectionPainter(
                    detections: _result.detections,
                    imageWidth: decoded.width,
                    imageHeight: decoded.height,
                  ),
                ),
              ],
            ),
          ),
        ),
        if (_busy)
          const Positioned.fill(
            child: ColoredBox(
              color: Colors.black26,
              child: Center(child: CircularProgressIndicator()),
            ),
          ),
      ],
    );
  }

  Widget _centered({
    required IconData icon,
    required Color color,
    required String title,
    String? subtitle,
  }) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(icon, size: 56, color: color),
            const SizedBox(height: 12),
            Text(title, textAlign: TextAlign.center),
            if (subtitle != null) ...[
              const SizedBox(height: 6),
              Text(
                subtitle,
                textAlign: TextAlign.center,
                style: const TextStyle(fontSize: 12, color: Colors.grey),
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildPanel(ThemeData theme) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 88),
      decoration: BoxDecoration(
        color: theme.colorScheme.surfaceContainerHighest,
        borderRadius: const BorderRadius.vertical(top: Radius.circular(16)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              _stat(
                label: ShrimpClass.labels[ShrimpClass.healthy],
                value: '${_result.healthyCount}',
                color: ShrimpClass.colors[ShrimpClass.healthy],
              ),
              _stat(
                label: ShrimpClass.labels[ShrimpClass.diseased],
                value: '${_result.diseasedCount}',
                color: ShrimpClass.colors[ShrimpClass.diseased],
              ),
              _stat(
                label: 'Tỉ lệ bệnh',
                value: '${_result.diseaseRate.toStringAsFixed(0)}%',
                color: theme.colorScheme.onSurface,
              ),
              _stat(
                label: 'Thời gian',
                value: '${_result.inferenceMs}ms',
                color: theme.colorScheme.onSurface,
              ),
            ],
          ),
          const SizedBox(height: 4),
          Row(
            children: [
              Text('Ngưỡng ${_confThreshold.toStringAsFixed(2)}'),
              Expanded(
                child: Slider(
                  value: _confThreshold,
                  min: 0.05,
                  max: 0.95,
                  divisions: 18,
                  label: _confThreshold.toStringAsFixed(2),
                  onChanged: (v) => setState(() => _confThreshold = v),
                  onChangeEnd: (_) => _run(),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _stat({
    required String label,
    required String value,
    required Color color,
  }) {
    return Expanded(
      child: Column(
        children: [
          Text(
            value,
            style: TextStyle(
              fontSize: 22,
              fontWeight: FontWeight.bold,
              color: color,
            ),
          ),
          Text(label, style: const TextStyle(fontSize: 11)),
        ],
      ),
    );
  }
}
