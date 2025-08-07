import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';
import 'dart:ui';
import 'ui/screens/main_screen.dart';
import 'core/providers/system_state_provider.dart';
import 'core/providers/detection_provider.dart';
import 'core/ffi/cpp_bridge.dart';

void main() {
  runApp(const BambooCutApp());
}

/// 智能切竹机主应用
class BambooCutApp extends StatelessWidget {
  const BambooCutApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => SystemStateProvider()),
        ChangeNotifierProvider(create: (_) => DetectionProvider()),
      ],
      child: MaterialApp(
        title: '智能切竹机',
        debugShowCheckedModeBanner: false,
        theme: ThemeData(
          primarySwatch: Colors.blue,
          brightness: Brightness.dark,
          visualDensity: VisualDensity.adaptivePlatformDensity,
          fontFamily: 'RobotoMono',
        ),
        home: const BambooCutHomePage(),
      ),
    );
  }
}

/// 主页面
class BambooCutHomePage extends StatefulWidget {
  const BambooCutHomePage({super.key});

  @override
  State<BambooCutHomePage> createState() => _BambooCutHomePageState();
}

class _BambooCutHomePageState extends State<BambooCutHomePage>
    with WidgetsBindingObserver {
  bool _isInitialized = false;
  bool _hasError = false;
  String _errorMessage = '';

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    
    // 设置全屏模式
    SystemChrome.setEnabledSystemUIMode(SystemUiMode.immersive);
    
    // 设置屏幕方向为横屏
    SystemChrome.setPreferredOrientations([
      DeviceOrientation.landscapeLeft,
      DeviceOrientation.landscapeRight,
    ]);
    
    // 初始化系统
    _initializeSystem();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
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
      case AppLifecycleState.detached:
        _shutdownSystem();
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
        // 初始化Provider
        final systemProvider = context.read<SystemStateProvider>();
        systemProvider.initialize();
        
        setState(() {
          _isInitialized = true;
        });
      } else {
        setState(() {
          _hasError = true;
          _errorMessage = '系统初始化失败，错误代码: $result';
        });
      }
    } catch (e) {
      setState(() {
        _hasError = true;
        _errorMessage = '系统初始化异常: $e';
      });
    }
  }

  /// 恢复系统
  void _resumeSystem() {
    // 恢复系统状态
    final systemProvider = context.read<SystemStateProvider>();
    if (systemProvider.isInitialized) {
      // 可以在这里添加恢复逻辑
    }
  }

  /// 暂停系统
  void _pauseSystem() {
    // 暂停系统状态
    final systemProvider = context.read<SystemStateProvider>();
    if (systemProvider.isInitialized) {
      // 可以在这里添加暂停逻辑
    }
  }

  /// 关闭系统
  void _shutdownSystem() {
    try {
      // 关闭C++推理服务
      CppBridge.shutdownInferenceService();
    } catch (e) {
      print('系统关闭异常: $e');
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_hasError) {
      return _buildErrorScreen();
    }
    
    if (!_isInitialized) {
      return _buildLoadingScreen();
    }
    
    return const MainScreen();
  }

  /// 构建错误屏幕
  Widget _buildErrorScreen() {
    return Scaffold(
      backgroundColor: Colors.black,
      body: Center(
        child: Container(
          padding: const EdgeInsets.all(32),
          decoration: BoxDecoration(
            color: Colors.red[900],
            borderRadius: BorderRadius.circular(16),
            border: Border.all(color: Colors.red[400], width: 2),
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Icon(
                Icons.error_outline,
                color: Colors.white,
                size: 64,
              ),
              const SizedBox(height: 16),
              const Text(
                '系统错误',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 24,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 16),
              Text(
                _errorMessage,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 16,
                ),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 24),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: [
                  ElevatedButton(
                    onPressed: () {
                      setState(() {
                        _hasError = false;
                        _errorMessage = '';
                      });
                      _initializeSystem();
                    },
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.blue[700],
                      foregroundColor: Colors.white,
                    ),
                    child: const Text('重试'),
                  ),
                  ElevatedButton(
                    onPressed: () {
                      SystemNavigator.pop();
                    },
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.grey[700],
                      foregroundColor: Colors.white,
                    ),
                    child: const Text('退出'),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }

  /// 构建加载屏幕
  Widget _buildLoadingScreen() {
    return Scaffold(
      backgroundColor: Colors.black,
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            // Logo或图标
            Container(
              width: 120,
              height: 120,
              decoration: BoxDecoration(
                color: Colors.blue[900],
                shape: BoxShape.circle,
                boxShadow: [
                  BoxShadow(
                    color: Colors.blue.withOpacity(0.3),
                    blurRadius: 20,
                    spreadRadius: 5,
                  ),
                ],
              ),
              child: const Icon(
                Icons.forest,
                color: Colors.white,
                size: 60,
              ),
            ),
            const SizedBox(height: 32),
            const Text(
              '智能切竹机',
              style: TextStyle(
                color: Colors.white,
                fontSize: 32,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 16),
            const Text(
              '系统初始化中...',
              style: TextStyle(
                color: Colors.grey,
                fontSize: 16,
              ),
            ),
            const SizedBox(height: 32),
            const CircularProgressIndicator(
              valueColor: AlwaysStoppedAnimation<Color>(Colors.blue),
              strokeWidth: 3,
            ),
            const SizedBox(height: 16),
            const Text(
              '正在加载AI模型和硬件驱动...',
              style: TextStyle(
                color: Colors.grey,
                fontSize: 14,
              ),
            ),
          ],
        ),
      ),
    );
  }
} 