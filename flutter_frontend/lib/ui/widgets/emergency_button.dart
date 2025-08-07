import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'dart:ui' as ui;

/// 紧急停止按钮组件
class EmergencyButton extends StatefulWidget {
  final VoidCallback onPressed;
  
  const EmergencyButton({
    super.key,
    required this.onPressed,
  });

  @override
  State<EmergencyButton> createState() => _EmergencyButtonState();
}

class _EmergencyButtonState extends State<EmergencyButton>
    with TickerProviderStateMixin {
  late AnimationController _pulseController;
  late AnimationController _scaleController;
  late Animation<double> _pulseAnimation;
  late Animation<double> _scaleAnimation;
  
  bool _isPressed = false;
  bool _isEmergencyMode = false;

  @override
  void initState() {
    super.initState();
    
    _pulseController = AnimationController(
      duration: const Duration(milliseconds: 1000),
      vsync: this,
    );
    
    _scaleController = AnimationController(
      duration: const Duration(milliseconds: 150),
      vsync: this,
    );
    
    _pulseAnimation = Tween<double>(
      begin: 1.0,
      end: 1.1,
    ).animate(CurvedAnimation(
      parent: _pulseController,
      curve: Curves.easeInOut,
    ));
    
    _scaleAnimation = Tween<double>(
      begin: 1.0,
      end: 0.95,
    ).animate(CurvedAnimation(
      parent: _scaleController,
      curve: Curves.easeInOut,
    ));
    
    // 开始脉冲动画
    _pulseController.repeat(reverse: true);
  }

  @override
  void dispose() {
    _pulseController.dispose();
    _scaleController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return RepaintBoundary(
      child: AnimatedBuilder(
        animation: Listenable.merge([_pulseAnimation, _scaleAnimation]),
        builder: (context, child) {
          return Transform.scale(
            scale: _isEmergencyMode ? _pulseAnimation.value : _scaleAnimation.value,
            child: GestureDetector(
              onTapDown: (_) => _onTapDown(),
              onTapUp: (_) => _onTapUp(),
              onTapCancel: _onTapCancel,
              child: Container(
                width: 80,
                height: 80,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: _isEmergencyMode ? Colors.red[900] : Colors.red[700],
                  border: Border.all(
                    color: Colors.white,
                    width: 4,
                  ),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.red.withOpacity(0.5),
                      blurRadius: 20,
                      spreadRadius: 5,
                    ),
                    BoxShadow(
                      color: Colors.black.withOpacity(0.3),
                      blurRadius: 10,
                      offset: const Offset(0, 5),
                    ),
                  ],
                ),
                child: Center(
                  child: Icon(
                    Icons.emergency,
                    color: Colors.white,
                    size: 40,
                  ),
                ),
              ),
            ),
          );
        },
      ),
    );
  }

  void _onTapDown() {
    setState(() {
      _isPressed = true;
    });
    _scaleController.forward();
    
    // 触觉反馈
    HapticFeedback.heavyImpact();
  }

  void _onTapUp() {
    setState(() {
      _isPressed = false;
      _isEmergencyMode = true;
    });
    _scaleController.reverse();
    
    // 触发紧急停止
    widget.onPressed();
    
    // 开始紧急模式动画
    _pulseController.repeat(reverse: true);
    
    // 触觉反馈
    HapticFeedback.vibrate();
  }

  void _onTapCancel() {
    setState(() {
      _isPressed = false;
    });
    _scaleController.reverse();
  }

  /// 重置紧急模式
  void resetEmergencyMode() {
    setState(() {
      _isEmergencyMode = false;
    });
    _pulseController.stop();
    _pulseController.reset();
  }
} 