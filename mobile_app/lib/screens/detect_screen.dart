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
      backgroundColor: theme.colorScheme.surface,
      appBar: AppBar(
        title: const Text(
          'Nhận diện tôm bệnh',
          style: TextStyle(fontWeight: FontWeight.w600),
        ),
        centerTitle: false,
        backgroundColor: theme.colorScheme.surface,
        surfaceTintColor: Colors.transparent,
        scrolledUnderElevation: 0,
      ),
      body: SafeArea(
        top: false,
        child: Column(
          children: [
            Expanded(child: _buildPreview(theme)),
            _buildPanel(theme),
          ],
        ),
      ),
    );
  }

  // ======================= KHUNG ẢNH =======================

  Widget _buildPreview(ThemeData theme) {
    if (_error != null) {
      return _placeholder(
        theme,
        icon: Icons.error_outline_rounded,
        tint: theme.colorScheme.error,
        title: 'Đã xảy ra lỗi',
        subtitle: _error,
      );
    }
    if (!_detector.isLoaded) {
      return _placeholder(
        theme,
        icon: Icons.downloading_rounded,
        tint: theme.colorScheme.primary,
        title: 'Đang nạp model…',
      );
    }

    final file = _imageFile;
    final decoded = _decoded;
    if (file == null || decoded == null) {
      return _placeholder(
        theme,
        icon: Icons.add_photo_alternate_outlined,
        tint: theme.colorScheme.primary,
        title: 'Chọn một ảnh để bắt đầu',
        subtitle: 'yolo11n chạy ngay trên máy — không cần mạng, '
            'ảnh không rời khỏi thiết bị',
      );
    }

    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 4, 16, 16),
      child: Center(
        child: AspectRatio(
          aspectRatio: decoded.width / decoded.height,
          child: DecoratedBox(
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(20),
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withValues(alpha: 0.16),
                  blurRadius: 24,
                  offset: const Offset(0, 8),
                ),
              ],
            ),
            child: ClipRRect(
              borderRadius: BorderRadius.circular(20),
              child: Stack(
                fit: StackFit.expand,
                children: [
                  // fill chứ không phải cover: toạ độ box giả định toàn bộ ảnh
                  // phủ kín canvas. AspectRatio bên ngoài đã giữ đúng tỉ lệ nên
                  // fill không làm méo, mà lại loại hẳn khả năng bị cắt viền.
                  Image.file(file, fit: BoxFit.fill),
                  CustomPaint(
                    painter: DetectionPainter(
                      detections: _result.detections,
                      imageWidth: decoded.width,
                      imageHeight: decoded.height,
                      brightness: theme.brightness,
                    ),
                  ),
                  if (_busy)
                    ColoredBox(
                      color: Colors.black.withValues(alpha: 0.35),
                      child: const Center(
                        child: CircularProgressIndicator(color: Colors.white),
                      ),
                    ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _placeholder(
    ThemeData theme, {
    required IconData icon,
    required Color tint,
    required String title,
    String? subtitle,
  }) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 40),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Container(
              width: 96,
              height: 96,
              decoration: BoxDecoration(
                color: tint.withValues(alpha: 0.10),
                shape: BoxShape.circle,
              ),
              child: Icon(icon, size: 44, color: tint),
            ),
            const SizedBox(height: 20),
            Text(
              title,
              textAlign: TextAlign.center,
              style: theme.textTheme.titleMedium?.copyWith(
                fontWeight: FontWeight.w600,
              ),
            ),
            if (subtitle != null) ...[
              const SizedBox(height: 8),
              Text(
                subtitle,
                textAlign: TextAlign.center,
                style: theme.textTheme.bodySmall?.copyWith(
                  color: theme.colorScheme.onSurfaceVariant,
                  height: 1.45,
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }

  // ======================= BẢNG KẾT QUẢ =======================

  Widget _buildPanel(ThemeData theme) {
    final hasResult = _decoded != null;
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.fromLTRB(20, 18, 20, 20),
      decoration: BoxDecoration(
        color: theme.colorScheme.surfaceContainerLow,
        borderRadius: const BorderRadius.vertical(top: Radius.circular(28)),
        border: Border(
          top: BorderSide(color: theme.colorScheme.outlineVariant, width: 1),
        ),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              _stat(theme, classId: ShrimpClass.healthy),
              _divider(theme),
              _stat(theme, classId: ShrimpClass.diseased),
              _divider(theme),
              _totalStat(theme),
            ],
          ),
          if (hasResult) ...[
            const SizedBox(height: 18),
            _diseaseMeter(theme),
          ],
          const SizedBox(height: 18),
          _thresholdRow(theme),
          const SizedBox(height: 16),
          _actions(theme),
        ],
      ),
    );
  }

  /// Con số mặc màu chữ, biểu tượng màu bên cạnh mới mang danh tính lớp — nếu
  /// tô màu thẳng vào chữ số thì người mù màu mất luôn cả số lẫn nhãn.
  Widget _stat(ThemeData theme, {required int classId}) {
    final color = ShrimpClass.colorOf(classId, theme.brightness);
    final count = classId == ShrimpClass.healthy
        ? _result.healthyCount
        : _result.diseasedCount;

    return Expanded(
      child: Column(
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(ShrimpClass.iconOf(classId), size: 18, color: color),
              const SizedBox(width: 6),
              Text(
                '$count',
                style: theme.textTheme.headlineSmall?.copyWith(
                  fontWeight: FontWeight.w700,
                  height: 1.0,
                ),
              ),
            ],
          ),
          const SizedBox(height: 5),
          Text(
            ShrimpClass.labelOf(classId),
            style: theme.textTheme.labelMedium?.copyWith(
              color: theme.colorScheme.onSurfaceVariant,
            ),
          ),
        ],
      ),
    );
  }

  Widget _totalStat(ThemeData theme) {
    return Expanded(
      child: Column(
        children: [
          Text(
            '${_result.total}',
            style: theme.textTheme.headlineSmall?.copyWith(
              fontWeight: FontWeight.w700,
              height: 1.0,
            ),
          ),
          const SizedBox(height: 5),
          Text(
            'Tổng số con',
            style: theme.textTheme.labelMedium?.copyWith(
              color: theme.colorScheme.onSurfaceVariant,
            ),
          ),
        ],
      ),
    );
  }

  Widget _divider(ThemeData theme) => Container(
        width: 1,
        height: 34,
        color: theme.colorScheme.outlineVariant,
      );

  Widget _diseaseMeter(ThemeData theme) {
    final rate = _result.diseaseRate;
    final color = ShrimpClass.colorOf(ShrimpClass.diseased, theme.brightness);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Text(
              'Tỉ lệ tôm bệnh',
              style: theme.textTheme.labelLarge?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),
            const Spacer(),
            Text(
              '${rate.toStringAsFixed(0)}%',
              style: theme.textTheme.labelLarge?.copyWith(
                fontWeight: FontWeight.w700,
              ),
            ),
          ],
        ),
        const SizedBox(height: 7),
        ClipRRect(
          borderRadius: BorderRadius.circular(4),
          child: Stack(
            children: [
              Container(height: 8, color: theme.colorScheme.outlineVariant),
              FractionallySizedBox(
                widthFactor: (rate / 100).clamp(0.0, 1.0),
                child: Container(height: 8, color: color),
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _thresholdRow(ThemeData theme) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Text(
              'Ngưỡng tin cậy',
              style: theme.textTheme.labelLarge?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),
            const Spacer(),
            Text(
              _confThreshold.toStringAsFixed(2),
              style: theme.textTheme.labelLarge?.copyWith(
                fontWeight: FontWeight.w700,
                fontFeatures: const [FontFeature.tabularFigures()],
              ),
            ),
            if (_result.inferenceMs > 0) ...[
              Text(
                '  ·  ${_result.inferenceMs} ms',
                style: theme.textTheme.labelMedium?.copyWith(
                  color: theme.colorScheme.onSurfaceVariant,
                ),
              ),
            ],
          ],
        ),
        SliderTheme(
          data: SliderTheme.of(context).copyWith(
            trackHeight: 4,
            overlayShape: const RoundSliderOverlayShape(overlayRadius: 16),
          ),
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
    );
  }

  Widget _actions(ThemeData theme) {
    return Row(
      children: [
        Expanded(
          child: FilledButton.tonalIcon(
            onPressed: _busy ? null : () => _pick(ImageSource.gallery),
            icon: const Icon(Icons.photo_library_outlined, size: 20),
            label: const Text('Thư viện'),
            style: FilledButton.styleFrom(
              padding: const EdgeInsets.symmetric(vertical: 15),
            ),
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: FilledButton.icon(
            onPressed: _busy ? null : () => _pick(ImageSource.camera),
            icon: const Icon(Icons.photo_camera_outlined, size: 20),
            label: const Text('Chụp ảnh'),
            style: FilledButton.styleFrom(
              padding: const EdgeInsets.symmetric(vertical: 15),
            ),
          ),
        ),
      ],
    );
  }
}
