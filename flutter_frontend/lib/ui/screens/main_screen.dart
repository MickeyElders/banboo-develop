import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'dart:async';
import 'package:provider/provider.dart';
import '../widgets/gpu_accelerated_video_widget.dart';
import '../widgets/detection_overlay.dart';
import '../widgets/control_panel.dart';
import '../widgets/status_panel.dart';
import '../widgets/emergency_button.dart';
import '../../core/ffi/cpp_bridge.dart';
import '../../core/providers/system_state_provider.dart';
import '../../core/providers/detection_provider.dart';

/// 主界面屏幕 - GPU加速的嵌入式触摸界面
class MainScreen extends StatefulWidget {
  const MainScreen({super.key});

  @override
  State<MainScreen> createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen>
    with TickerProviderStateMixin, WidgetsBindingObserver {
  late AnimationController _fadeController;
  late AnimationController _slideController;
  late Animation<double> _fadeAnimation;
  late Animation<Offset> _slideAnimation;
  
  Timer? _statusTimer;
  Timer? _detectionTimer;
  bool _isInitialized = false;
  bool _isEmergencyMode = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    
    // 初始化动画控制器
    _fadeController = AnimationController(
      duration: const Duration(milliseconds: 500),
      vsync: this,
    );
    _slideController = AnimationController(
      duration: const Duration(milliseconds: 300),
      vsync: this,
    );
    
    _fadeAnimation = Tween<double>(
      begin: 0.0,
      end: 1.0,
    ).animate(CurvedAnimation(
      parent: _fadeController,
      curve: Curves.easeInOut,
    ));
    
    _slideAnimation = Tween<Offset>(
      begin: const Offset(0, 1),
      end: Offset.zero,
    ).animate(CurvedAnimation(
      parent: _slideController,
      curve: Curves.easeOutCubic,
    ));
    
    _initializeSystem();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _fadeController.dispose();
    _slideController.dispose();
    _statusTimer?.cancel();
    _detectionTimer?.cancel();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    super.didChangeAppLifecycleState(state);
    switch (state) {
      case AppLifecycleState.resumed:
        _resumeSystem();
        break;
      case AppLifecycleState.paused:
        _pauseSystem();
        break;
      default:
        break;
    }
  }

  /// 初始化系统
  Future<void> _initializeSystem() async {
    try {
      // 初始化C++推理服务
      final result = CppBridge.initializeInferenceService();
      if (result == 0) {
        // 开始摄像头捕获
        await CppBridge.startCameraCapture();
        
        setState(() {
          _isInitialized = true;
        });
        
        // 启动状态监控
        _startStatusMonitoring();
        
        // 启动检测循环
        _startDetectionLoop();
        
        // 播放启动动画
        _fadeController.forward();
        _slideController.forward();
      } else {
        _showErrorDialog('系统初始化失败', '错误代码: $result');
      }
    } catch (e) {
      _showErrorDialog('系统初始化异常', e.toString());
    }
  }

  /// 启动状态监控
  void _startStatusMonitoring() {
    _statusTimer = Timer.periodic(const Duration(milliseconds: 100), (timer) {
      if (!_isEmergencyMode) {
        _updateSystemStatus();
      }
    });
  }

  /// 启动检测循环
  void _startDetectionLoop() {
    _detectionTimer = Timer.periodic(const Duration(milliseconds: 33), (timer) {
      if (!_isEmergencyMode) {
        _performDetection();
      }
    });
  }

  /// 更新系统状态
  Future<void> _updateSystemStatus() async {
    try {
      final status = CppBridge.getSystemStatus();
      if (status.containsKey('error')) {
        print('状态更新失败: ${status['error']}');
        return;
      }
      
      // 更新Provider状态
      final systemProvider = context.read<SystemStateProvider>();
      systemProvider.updateStatus(status);
      
      // 检查紧急状态
      if (status['emergency_stop'] == true) {
        _handleEmergencyStop();
      }
    } catch (e) {
      print('状态更新异常: $e');
    }
  }

  /// 执行检测
  Future<void> _performDetection() async {
    try {
      final frameData = CppBridge.getCameraFrame();
      if (frameData != null) {
        // 更新视频帧
        final detectionProvider = context.read<DetectionProvider>();
        detectionProvider.updateVideoFrame(frameData);
        
        // 执行AI检测
        final detectionResult = CppBridge.detectBamboo(
          frameData,
          1920, // 假设宽度
          1080, // 假设高度
        );
        
        if (!detectionResult.containsKey('error')) {
          detectionProvider.updateDetectionResult(detectionResult);
        }
      }
    } catch (e) {
      print('检测异常: $e');
    }
  }

  /// 处理紧急停止
  void _handleEmergencyStop() {
    setState(() {
      _isEmergencyMode = true;
    });
    
    // 停止所有定时器
    _statusTimer?.cancel();
    _detectionTimer?.cancel();
    
    // 调用C++紧急停止
    CppBridge.emergencyStop();
    
    // 显示紧急停止界面
    _showEmergencyStopDialog();
  }

  /// 恢复系统
  Future<void> _resumeSystem() async {
    if (_isInitialized && !_isEmergencyMode) {
      _startStatusMonitoring();
      _startDetectionLoop();
    }
  }

  /// 暂停系统
  void _pauseSystem() {
    _statusTimer?.cancel();
    _detectionTimer?.cancel();
  }

  /// 显示错误对话框
  void _showErrorDialog(String title, String message) {
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (context) => AlertDialog(
        title: Text(title),
        content: Text(message),
        actions: [
          TextButton(
            onPressed: () {
              Navigator.of(context).pop();
              SystemNavigator.pop();
            },
            child: const Text('退出'),
          ),
          TextButton(
            onPressed: () {
              Navigator.of(context).pop();
              _initializeSystem();
            },
            child: const Text('重试'),
          ),
        ],
      ),
    );
  }

  /// 显示紧急停止对话框
  void _showEmergencyStopDialog() {
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (context) => AlertDialog(
        backgroundColor: Colors.red[900],
        title: const Text(
          '紧急停止',
          style: TextStyle(color: Colors.white, fontSize: 24),
        ),
        content: const Text(
          '系统已紧急停止，请检查设备状态',
          style: TextStyle(color: Colors.white, fontSize: 18),
        ),
        actions: [
          TextButton(
            onPressed: () {
              Navigator.of(context).pop();
              SystemNavigator.pop();
            },
            child: const Text(
              '退出',
              style: TextStyle(color: Colors.white),
            ),
          ),
          TextButton(
            onPressed: () {
              Navigator.of(context).pop();
              _resetEmergencyMode();
            },
            child: const Text(
              '重置',
              style: TextStyle(color: Colors.white),
            ),
          ),
        ],
      ),
    );
  }

  /// 重置紧急模式
  void _resetEmergencyMode() {
    setState(() {
      _isEmergencyMode = false;
    });
    
    _resumeSystem();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: SafeArea(
        child: FadeTransition(
          opacity: _fadeAnimation,
          child: SlideTransition(
            position: _slideAnimation,
            child: Stack(
              children: [
                // 主视频显示区域
                Positioned.fill(
                  child: RepaintBoundary(
                    child: GPUAcceleratedVideoWidget(
                      key: const ValueKey('video_widget'),
                    ),
                  ),
                ),
                
                // 检测结果叠加层
                Positioned.fill(
                  child: RepaintBoundary(
                    child: DetectionOverlay(
                      key: const ValueKey('detection_overlay'),
                    ),
                  ),
                ),
                
                // 顶部状态栏
                Positioned(
                  top: 0,
                  left: 0,
                  right: 0,
                  child: RepaintBoundary(
                    child: StatusPanel(
                      key: const ValueKey('status_panel'),
                    ),
                  ),
                ),
                
                // 右侧控制面板
                Positioned(
                  top: 100,
                  right: 0,
                  bottom: 100,
                  child: RepaintBoundary(
                    child: ControlPanel(
                      key: const ValueKey('control_panel'),
                      onCuttingParametersChanged: (xCoordinate, bladeNumber, qualityThreshold) {
                        CppBridge.setCuttingParameters(xCoordinate, bladeNumber, qualityThreshold);
                      },
                    ),
                  ),
                ),
                
                // 紧急停止按钮
                Positioned(
                  bottom: 20,
                  right: 20,
                  child: RepaintBoundary(
                    child: EmergencyButton(
                      key: const ValueKey('emergency_button'),
                      onPressed: _handleEmergencyStop,
                    ),
                  ),
                ),
                
                // 加载指示器
                if (!_isInitialized)
                  Positioned.fill(
                    child: Container(
                      color: Colors.black87,
                      child: const Center(
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            CircularProgressIndicator(
                              valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                            ),
                            SizedBox(height: 20),
                            Text(
                              '系统初始化中...',
                              style: TextStyle(
                                color: Colors.white,
                                fontSize: 18,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ),
              ],
            ),
          ),
        ),
      ),
    );
  }
} 