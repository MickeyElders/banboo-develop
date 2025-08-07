import 'package:flutter/material.dart';
import 'dart:ui' as ui;
import 'package:provider/provider.dart';
import '../../core/providers/detection_provider.dart';

/// 检测结果叠加显示组件
class DetectionOverlay extends StatelessWidget {
  const DetectionOverlay({super.key});

  @override
  Widget build(BuildContext context) {
    return Consumer<DetectionProvider>(
      builder: (context, detectionProvider, child) {
        final detections = detectionProvider.currentDetections;
        
        if (detections.isEmpty) {
          return const SizedBox.shrink();
        }
        
        return RepaintBoundary(
          child: CustomPaint(
            painter: _DetectionPainter(detections),
            size: Size.infinite,
          ),
        );
      },
    );
  }
}

/// 检测结果绘制器
class _DetectionPainter extends CustomPainter {
  final List<Map<String, dynamic>> detections;
  
  _DetectionPainter(this.detections);
  
  @override
  void paint(Canvas canvas, Size size) {
    for (final detection in detections) {
      _drawDetectionBox(canvas, size, detection);
      _drawDetectionLabel(canvas, size, detection);
    }
  }
  
  void _drawDetectionBox(Canvas canvas, Size size, Map<String, dynamic> detection) {
    final x = detection['x']?.toDouble() ?? 0.0;
    final y = detection['y']?.toDouble() ?? 0.0;
    final width = detection['width']?.toDouble() ?? 0.0;
    final height = detection['height']?.toDouble() ?? 0.0;
    final confidence = detection['confidence']?.toDouble() ?? 0.0;
    
    // 计算屏幕坐标
    final screenX = x * size.width;
    final screenY = y * size.height;
    final screenWidth = width * size.width;
    final screenHeight = height * size.height;
    
    // 根据置信度选择颜色
    final color = _getConfidenceColor(confidence);
    
    // 绘制边界框
    final paint = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0
      ..isAntiAlias = true;
    
    final rect = Rect.fromLTWH(screenX, screenY, screenWidth, screenHeight);
    canvas.drawRect(rect, paint);
    
    // 绘制角标
    _drawCornerMarkers(canvas, rect, color);
  }
  
  void _drawCornerMarkers(Canvas canvas, Rect rect, Color color) {
    final paint = Paint()
      ..color = color
      ..style = PaintingStyle.fill
      ..isAntiAlias = true;
    
    final markerLength = 20.0;
    final markerThickness = 3.0;
    
    // 左上角
    canvas.drawRect(
      Rect.fromLTWH(rect.left, rect.top, markerLength, markerThickness),
      paint,
    );
    canvas.drawRect(
      Rect.fromLTWH(rect.left, rect.top, markerThickness, markerLength),
      paint,
    );
    
    // 右上角
    canvas.drawRect(
      Rect.fromLTWH(rect.right - markerLength, rect.top, markerLength, markerThickness),
      paint,
    );
    canvas.drawRect(
      Rect.fromLTWH(rect.right - markerThickness, rect.top, markerThickness, markerLength),
      paint,
    );
    
    // 左下角
    canvas.drawRect(
      Rect.fromLTWH(rect.left, rect.bottom - markerThickness, markerLength, markerThickness),
      paint,
    );
    canvas.drawRect(
      Rect.fromLTWH(rect.left, rect.bottom - markerLength, markerThickness, markerLength),
      paint,
    );
    
    // 右下角
    canvas.drawRect(
      Rect.fromLTWH(rect.right - markerLength, rect.bottom - markerThickness, markerLength, markerThickness),
      paint,
    );
    canvas.drawRect(
      Rect.fromLTWH(rect.right - markerThickness, rect.bottom - markerLength, markerThickness, markerLength),
      paint,
    );
  }
  
  void _drawDetectionLabel(Canvas canvas, Size size, Map<String, dynamic> detection) {
    final x = detection['x']?.toDouble() ?? 0.0;
    final y = detection['y']?.toDouble() ?? 0.0;
    final width = detection['width']?.toDouble() ?? 0.0;
    final confidence = detection['confidence']?.toDouble() ?? 0.0;
    final label = detection['label']?.toString() ?? 'Bamboo';
    
    // 计算屏幕坐标
    final screenX = x * size.width;
    final screenY = y * size.height;
    final screenWidth = width * size.width;
    
    // 创建文本样式
    const textStyle = TextStyle(
      color: Colors.white,
      fontSize: 14,
      fontWeight: FontWeight.bold,
      shadows: [
        Shadow(
          offset: Offset(1, 1),
          blurRadius: 2,
          color: Colors.black,
        ),
      ],
    );
    
    // 创建文本段落
    final textSpan = TextSpan(
      text: '$label ${(confidence * 100).toStringAsFixed(1)}%',
      style: textStyle,
    );
    
    final textPainter = TextPainter(
      text: textSpan,
      textDirection: TextDirection.ltr,
    );
    
    textPainter.layout();
    
    // 计算标签背景
    final labelWidth = textPainter.width + 16;
    final labelHeight = textPainter.height + 8;
    final labelX = screenX;
    final labelY = screenY - labelHeight - 5;
    
    // 绘制标签背景
    final backgroundPaint = Paint()
      ..color = _getConfidenceColor(confidence).withOpacity(0.8)
      ..style = PaintingStyle.fill
      ..isAntiAlias = true;
    
    final backgroundRect = Rect.fromLTWH(labelX, labelY, labelWidth, labelHeight);
    canvas.drawRRect(
      RRect.fromRectAndRadius(backgroundRect, const Radius.circular(4)),
      backgroundPaint,
    );
    
    // 绘制文本
    textPainter.paint(
      canvas,
      Offset(labelX + 8, labelY + 4),
    );
  }
  
  Color _getConfidenceColor(double confidence) {
    if (confidence >= 0.8) {
      return Colors.green;
    } else if (confidence >= 0.6) {
      return Colors.orange;
    } else if (confidence >= 0.4) {
      return Colors.yellow;
    } else {
      return Colors.red;
    }
  }
  
  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) {
    return true;
  }
} 