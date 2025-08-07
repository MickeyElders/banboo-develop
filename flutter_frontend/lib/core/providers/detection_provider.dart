import 'package:flutter/foundation.dart';
import 'dart:typed_data';

/// 检测结果管理Provider
class DetectionProvider extends ChangeNotifier {
  Uint8List? _currentVideoFrame;
  List<Map<String, dynamic>> _currentDetections = [];
  Map<String, dynamic> _lastDetectionResult = {};
  bool _isDetecting = false;
  double _detectionConfidence = 0.0;
  int _detectionCount = 0;
  DateTime? _lastDetectionTime;

  /// 当前视频帧数据
  Uint8List? get currentVideoFrame => _currentVideoFrame;

  /// 当前检测结果
  List<Map<String, dynamic>> get currentDetections => _currentDetections;

  /// 最后一次检测结果
  Map<String, dynamic> get lastDetectionResult => _lastDetectionResult;

  /// 是否正在检测
  bool get isDetecting => _isDetecting;

  /// 检测置信度
  double get detectionConfidence => _detectionConfidence;

  /// 检测计数
  int get detectionCount => _detectionCount;

  /// 最后检测时间
  DateTime? get lastDetectionTime => _lastDetectionTime;

  /// 是否有检测结果
  bool get hasDetections => _currentDetections.isNotEmpty;

  /// 更新视频帧
  void updateVideoFrame(Uint8List frameData) {
    _currentVideoFrame = frameData;
    notifyListeners();
  }

  /// 更新检测结果
  void updateDetectionResult(Map<String, dynamic> result) {
    _lastDetectionResult = Map<String, dynamic>.from(result);
    _lastDetectionTime = DateTime.now();
    
    // 解析检测结果
    if (result.containsKey('detections')) {
      final detections = result['detections'] as List?;
      if (detections != null) {
        _currentDetections = detections.cast<Map<String, dynamic>>();
        _detectionCount = _currentDetections.length;
        
        // 计算平均置信度
        if (_currentDetections.isNotEmpty) {
          double totalConfidence = 0.0;
          for (final detection in _currentDetections) {
            totalConfidence += detection['confidence']?.toDouble() ?? 0.0;
          }
          _detectionConfidence = totalConfidence / _currentDetections.length;
        } else {
          _detectionConfidence = 0.0;
        }
      } else {
        _currentDetections = [];
        _detectionCount = 0;
        _detectionConfidence = 0.0;
      }
    }
    
    notifyListeners();
  }

  /// 设置检测状态
  void setDetecting(bool isDetecting) {
    _isDetecting = isDetecting;
    notifyListeners();
  }

  /// 清除检测结果
  void clearDetections() {
    _currentDetections = [];
    _lastDetectionResult = {};
    _detectionCount = 0;
    _detectionConfidence = 0.0;
    notifyListeners();
  }

  /// 添加检测结果
  void addDetection(Map<String, dynamic> detection) {
    _currentDetections.add(detection);
    _detectionCount = _currentDetections.length;
    _lastDetectionTime = DateTime.now();
    
    // 更新平均置信度
    double totalConfidence = 0.0;
    for (final det in _currentDetections) {
      totalConfidence += det['confidence']?.toDouble() ?? 0.0;
    }
    _detectionConfidence = totalConfidence / _currentDetections.length;
    
    notifyListeners();
  }

  /// 移除检测结果
  void removeDetection(int index) {
    if (index >= 0 && index < _currentDetections.length) {
      _currentDetections.removeAt(index);
      _detectionCount = _currentDetections.length;
      
      // 更新平均置信度
      if (_currentDetections.isNotEmpty) {
        double totalConfidence = 0.0;
        for (final detection in _currentDetections) {
          totalConfidence += detection['confidence']?.toDouble() ?? 0.0;
        }
        _detectionConfidence = totalConfidence / _currentDetections.length;
      } else {
        _detectionConfidence = 0.0;
      }
      
      notifyListeners();
    }
  }

  /// 获取高置信度检测结果
  List<Map<String, dynamic>> getHighConfidenceDetections(double threshold) {
    return _currentDetections.where((detection) {
      final confidence = detection['confidence']?.toDouble() ?? 0.0;
      return confidence >= threshold;
    }).toList();
  }

  /// 获取检测结果统计
  Map<String, dynamic> getDetectionStats() {
    if (_currentDetections.isEmpty) {
      return {
        'count': 0,
        'average_confidence': 0.0,
        'max_confidence': 0.0,
        'min_confidence': 0.0,
        'total_area': 0.0,
        'average_area': 0.0,
      };
    }
    
    double totalConfidence = 0.0;
    double maxConfidence = 0.0;
    double minConfidence = 1.0;
    double totalArea = 0.0;
    
    for (final detection in _currentDetections) {
      final confidence = detection['confidence']?.toDouble() ?? 0.0;
      final width = detection['width']?.toDouble() ?? 0.0;
      final height = detection['height']?.toDouble() ?? 0.0;
      final area = width * height;
      
      totalConfidence += confidence;
      maxConfidence = confidence > maxConfidence ? confidence : maxConfidence;
      minConfidence = confidence < minConfidence ? confidence : minConfidence;
      totalArea += area;
    }
    
    return {
      'count': _currentDetections.length,
      'average_confidence': totalConfidence / _currentDetections.length,
      'max_confidence': maxConfidence,
      'min_confidence': minConfidence,
      'total_area': totalArea,
      'average_area': totalArea / _currentDetections.length,
    };
  }

  /// 获取检测结果摘要
  String getDetectionSummary() {
    if (_currentDetections.isEmpty) {
      return '无检测结果';
    }
    
    final stats = getDetectionStats();
    return '检测到 ${stats['count']} 个目标，平均置信度 ${(stats['average_confidence'] * 100).toStringAsFixed(1)}%';
  }

  /// 重置Provider
  void reset() {
    _currentVideoFrame = null;
    _currentDetections = [];
    _lastDetectionResult = {};
    _isDetecting = false;
    _detectionConfidence = 0.0;
    _detectionCount = 0;
    _lastDetectionTime = null;
    notifyListeners();
  }
} 