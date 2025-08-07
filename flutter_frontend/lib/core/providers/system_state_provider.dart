import 'package:flutter/foundation.dart';
import 'dart:async';

/// 系统状态管理Provider
class SystemStateProvider extends ChangeNotifier {
  Map<String, dynamic> _currentStatus = {
    'system_status': 0, // 0=停止, 1=运行, 2=错误, 3=暂停, 4=紧急停止, 5=维护模式
    'fps': 0.0,
    'cpu_usage': 0.0,
    'memory_usage': 0.0,
    'gpu_usage': 0.0,
    'plc_connected': false,
    'camera_connected': false,
    'heartbeat_count': 0,
    'timestamp': DateTime.now(),
    'emergency_stop': false,
    'error_message': '',
    'warning_message': '',
  };

  Timer? _statusTimer;
  bool _isInitialized = false;

  /// 当前系统状态
  Map<String, dynamic> get currentStatus => _currentStatus;

  /// 是否已初始化
  bool get isInitialized => _isInitialized;

  /// 是否处于紧急停止状态
  bool get isEmergencyStop => _currentStatus['emergency_stop'] ?? false;

  /// 是否有错误
  bool get hasError => _currentStatus['system_status'] == 2;

  /// 是否有警告
  bool get hasWarning => _currentStatus['warning_message']?.isNotEmpty == true;

  /// 初始化状态监控
  void initialize() {
    if (_isInitialized) return;
    
    _isInitialized = true;
    _startStatusMonitoring();
    notifyListeners();
  }

  /// 开始状态监控
  void _startStatusMonitoring() {
    _statusTimer = Timer.periodic(const Duration(milliseconds: 100), (timer) {
      _updateStatus();
    });
  }

  /// 更新状态
  void _updateStatus() {
    // 这里可以添加实际的状态更新逻辑
    // 例如从C++后端获取状态信息
    
    final newStatus = Map<String, dynamic>.from(_currentStatus);
    newStatus['timestamp'] = DateTime.now();
    
    // 模拟状态更新
    if (!isEmergencyStop) {
      newStatus['heartbeat_count'] = (newStatus['heartbeat_count'] ?? 0) + 1;
    }
    
    _currentStatus = newStatus;
    notifyListeners();
  }

  /// 更新系统状态
  void updateStatus(Map<String, dynamic> status) {
    _currentStatus = Map<String, dynamic>.from(status);
    notifyListeners();
  }

  /// 设置系统状态
  void setSystemStatus(int status) {
    _currentStatus['system_status'] = status;
    notifyListeners();
  }

  /// 设置性能指标
  void setPerformanceMetrics({
    double? fps,
    double? cpuUsage,
    double? memoryUsage,
    double? gpuUsage,
  }) {
    if (fps != null) _currentStatus['fps'] = fps;
    if (cpuUsage != null) _currentStatus['cpu_usage'] = cpuUsage;
    if (memoryUsage != null) _currentStatus['memory_usage'] = memoryUsage;
    if (gpuUsage != null) _currentStatus['gpu_usage'] = gpuUsage;
    notifyListeners();
  }

  /// 设置连接状态
  void setConnectionStatus({
    bool? plcConnected,
    bool? cameraConnected,
  }) {
    if (plcConnected != null) _currentStatus['plc_connected'] = plcConnected;
    if (cameraConnected != null) _currentStatus['camera_connected'] = cameraConnected;
    notifyListeners();
  }

  /// 设置心跳计数
  void setHeartbeatCount(int count) {
    _currentStatus['heartbeat_count'] = count;
    notifyListeners();
  }

  /// 设置错误信息
  void setError(String errorMessage) {
    _currentStatus['error_message'] = errorMessage;
    _currentStatus['system_status'] = 2; // 错误状态
    notifyListeners();
  }

  /// 清除错误信息
  void clearError() {
    _currentStatus['error_message'] = '';
    if (_currentStatus['system_status'] == 2) {
      _currentStatus['system_status'] = 0; // 恢复到停止状态
    }
    notifyListeners();
  }

  /// 设置警告信息
  void setWarning(String warningMessage) {
    _currentStatus['warning_message'] = warningMessage;
    notifyListeners();
  }

  /// 清除警告信息
  void clearWarning() {
    _currentStatus['warning_message'] = '';
    notifyListeners();
  }

  /// 触发紧急停止
  void triggerEmergencyStop() {
    _currentStatus['emergency_stop'] = true;
    _currentStatus['system_status'] = 4; // 紧急停止状态
    _currentStatus['error_message'] = '系统紧急停止';
    notifyListeners();
  }

  /// 重置紧急停止
  void resetEmergencyStop() {
    _currentStatus['emergency_stop'] = false;
    _currentStatus['system_status'] = 0; // 恢复到停止状态
    _currentStatus['error_message'] = '';
    notifyListeners();
  }

  /// 启动系统
  void startSystem() {
    if (!isEmergencyStop) {
      _currentStatus['system_status'] = 1; // 运行状态
      _currentStatus['error_message'] = '';
      _currentStatus['warning_message'] = '';
      notifyListeners();
    }
  }

  /// 停止系统
  void stopSystem() {
    _currentStatus['system_status'] = 0; // 停止状态
    notifyListeners();
  }

  /// 暂停系统
  void pauseSystem() {
    if (_currentStatus['system_status'] == 1) {
      _currentStatus['system_status'] = 3; // 暂停状态
      notifyListeners();
    }
  }

  /// 恢复系统
  void resumeSystem() {
    if (_currentStatus['system_status'] == 3) {
      _currentStatus['system_status'] = 1; // 运行状态
      notifyListeners();
    }
  }

  /// 进入维护模式
  void enterMaintenanceMode() {
    _currentStatus['system_status'] = 5; // 维护模式
    notifyListeners();
  }

  /// 退出维护模式
  void exitMaintenanceMode() {
    _currentStatus['system_status'] = 0; // 停止状态
    notifyListeners();
  }

  @override
  void dispose() {
    _statusTimer?.cancel();
    super.dispose();
  }
} 