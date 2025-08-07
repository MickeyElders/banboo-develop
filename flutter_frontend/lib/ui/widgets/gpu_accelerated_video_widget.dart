import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:provider/provider.dart';
import '../../core/providers/detection_provider.dart';

/// GPU加速的视频显示组件
class GPUAcceleratedVideoWidget extends StatefulWidget {
  const GPUAcceleratedVideoWidget({super.key});

  @override
  State<GPUAcceleratedVideoWidget> createState() => _GPUAcceleratedVideoWidgetState();
}

class _GPUAcceleratedVideoWidgetState extends State<GPUAcceleratedVideoWidget>
    with AutomaticKeepAliveClientMixin {
  ui.Image? _currentImage;
  bool _isRendering = false;
  final _imageCompleter = Completer<ui.Image>();

  @override
  bool get wantKeepAlive => true;

  @override
  void initState() {
    super.initState();
    _initializeImageCompleter();
  }

  void _initializeImageCompleter() {
    _imageCompleter.complete(_createPlaceholderImage());
  }

  Future<ui.Image> _createPlaceholderImage() async {
    final recorder = ui.PictureRecorder();
    final canvas = Canvas(recorder);
    final paint = Paint()..color = Colors.grey[800]!;
    
    canvas.drawRect(
      Rect.fromLTWH(0, 0, 1920, 1080),
      paint,
    );
    
    final picture = recorder.endRecording();
    return await picture.toImage(1920, 1080);
  }

  @override
  Widget build(BuildContext context) {
    super.build(context);
    
    return Consumer<DetectionProvider>(
      builder: (context, detectionProvider, child) {
        final frameData = detectionProvider.currentVideoFrame;
        
        if (frameData != null && !_isRendering) {
          _updateImage(frameData);
        }
        
        return CustomPaint(
          painter: _VideoPainter(_currentImage),
          size: Size.infinite,
        );
      },
    );
  }

  Future<void> _updateImage(Uint8List frameData) async {
    if (_isRendering) return;
    
    setState(() {
      _isRendering = true;
    });
    
    try {
      // 使用GPU加速解码图像数据
      final codec = await ui.instantiateImageCodec(
        frameData,
        targetWidth: 1920,
        targetHeight: 1080,
      );
      
      final frameInfo = await codec.getNextFrame();
      final image = frameInfo.image;
      
      if (mounted) {
        setState(() {
          _currentImage = image;
          _isRendering = false;
        });
      }
    } catch (e) {
      print('图像更新失败: $e');
      setState(() {
        _isRendering = false;
      });
    }
  }
}

/// 视频绘制器
class _VideoPainter extends CustomPainter {
  final ui.Image? image;
  
  _VideoPainter(this.image);
  
  @override
  void paint(Canvas canvas, Size size) {
    if (image == null) {
      // 绘制占位符
      final paint = Paint()..color = Colors.grey[800]!;
      canvas.drawRect(Offset.zero & size, paint);
      return;
    }
    
    // 计算缩放比例以保持宽高比
    final imageAspectRatio = image!.width / image!.height;
    final canvasAspectRatio = size.width / size.height;
    
    Rect destRect;
    if (imageAspectRatio > canvasAspectRatio) {
      // 图像更宽，以高度为准
      final scaledWidth = size.height * imageAspectRatio;
      final x = (size.width - scaledWidth) / 2;
      destRect = Rect.fromLTWH(x, 0, scaledWidth, size.height);
    } else {
      // 图像更高，以宽度为准
      final scaledHeight = size.width / imageAspectRatio;
      final y = (size.height - scaledHeight) / 2;
      destRect = Rect.fromLTWH(0, y, size.width, scaledHeight);
    }
    
    // 使用GPU加速绘制图像
    final paint = Paint()
      ..filterQuality = FilterQuality.high
      ..isAntiAlias = true;
    
    canvas.drawImageRect(
      image!,
      Rect.fromLTWH(0, 0, image!.width.toDouble(), image!.height.toDouble()),
      destRect,
      paint,
    );
  }
  
  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) {
    return true;
  }
} 