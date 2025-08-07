import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'dart:ui' as ui;

/// 控制面板组件
class ControlPanel extends StatefulWidget {
  final Function(double xCoordinate, int bladeNumber, double qualityThreshold) onCuttingParametersChanged;
  
  const ControlPanel({
    super.key,
    required this.onCuttingParametersChanged,
  });

  @override
  State<ControlPanel> createState() => _ControlPanelState();
}

class _ControlPanelState extends State<ControlPanel>
    with TickerProviderStateMixin {
  late AnimationController _slideController;
  late Animation<Offset> _slideAnimation;
  
  double _xCoordinate = 0.0;
  int _bladeNumber = 1;
  double _qualityThreshold = 0.7;
  bool _isAutoMode = true;
  bool _isCutting = false;

  @override
  void initState() {
    super.initState();
    
    _slideController = AnimationController(
      duration: const Duration(milliseconds: 300),
      vsync: this,
    );
    
    _slideAnimation = Tween<Offset>(
      begin: const Offset(1, 0),
      end: Offset.zero,
    ).animate(CurvedAnimation(
      parent: _slideController,
      curve: Curves.easeOutCubic,
    ));
    
    _slideController.forward();
  }

  @override
  void dispose() {
    _slideController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return SlideTransition(
      position: _slideAnimation,
      child: Container(
        width: 300,
        margin: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: Colors.black87,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: Colors.white24, width: 1),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.3),
              blurRadius: 10,
              offset: const Offset(0, 5),
            ),
          ],
        ),
        child: RepaintBoundary(
          child: Column(
            children: [
              _buildHeader(),
              Expanded(
                child: SingleChildScrollView(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    children: [
                      _buildModeSwitch(),
                      const SizedBox(height: 20),
                      _buildCoordinateControl(),
                      const SizedBox(height: 20),
                      _buildBladeControl(),
                      const SizedBox(height: 20),
                      _buildQualityControl(),
                      const SizedBox(height: 20),
                      _buildActionButtons(),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildHeader() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.blue[900],
        borderRadius: const BorderRadius.only(
          topLeft: Radius.circular(16),
          topRight: Radius.circular(16),
        ),
      ),
      child: Row(
        children: [
          Icon(
            Icons.control_camera,
            color: Colors.white,
            size: 24,
          ),
          const SizedBox(width: 12),
          const Text(
            '控制面板',
            style: TextStyle(
              color: Colors.white,
              fontSize: 18,
              fontWeight: FontWeight.bold,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildModeSwitch() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.grey[900],
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.white24),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            '运行模式',
            style: TextStyle(
              color: Colors.white,
              fontSize: 16,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 12),
          Row(
            children: [
              Expanded(
                child: _buildModeButton(
                  '自动模式',
                  Icons.auto_awesome,
                  _isAutoMode,
                  () => setState(() => _isAutoMode = true),
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: _buildModeButton(
                  '手动模式',
                  Icons.touch_app,
                  !_isAutoMode,
                  () => setState(() => _isAutoMode = false),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildModeButton(String label, IconData icon, bool isSelected, VoidCallback onTap) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 8),
        decoration: BoxDecoration(
          color: isSelected ? Colors.blue[700] : Colors.grey[800],
          borderRadius: BorderRadius.circular(8),
          border: Border.all(
            color: isSelected ? Colors.blue[400] : Colors.grey[600],
            width: 2,
          ),
        ),
        child: Column(
          children: [
            Icon(
              icon,
              color: isSelected ? Colors.white : Colors.grey[400],
              size: 20,
            ),
            const SizedBox(height: 4),
            Text(
              label,
              style: TextStyle(
                color: isSelected ? Colors.white : Colors.grey[400],
                fontSize: 12,
                fontWeight: FontWeight.w500,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildCoordinateControl() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.grey[900],
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.white24),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'X坐标控制',
            style: TextStyle(
              color: Colors.white,
              fontSize: 16,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 12),
          Row(
            children: [
              Expanded(
                child: Text(
                  '${_xCoordinate.toStringAsFixed(1)} mm',
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
              IconButton(
                onPressed: () => _adjustCoordinate(-1.0),
                icon: const Icon(Icons.remove, color: Colors.white),
                style: IconButton.styleFrom(
                  backgroundColor: Colors.red[700],
                ),
              ),
              IconButton(
                onPressed: () => _adjustCoordinate(1.0),
                icon: const Icon(Icons.add, color: Colors.white),
                style: IconButton.styleFrom(
                  backgroundColor: Colors.green[700],
                ),
              ),
            ],
          ),
          Slider(
            value: _xCoordinate,
            min: 0.0,
            max: 1000.0,
            divisions: 1000,
            activeColor: Colors.blue[400],
            inactiveColor: Colors.grey[600],
            onChanged: (value) {
              setState(() {
                _xCoordinate = value;
              });
              _updateCuttingParameters();
            },
          ),
        ],
      ),
    );
  }

  Widget _buildBladeControl() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.grey[900],
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.white24),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            '刀片选择',
            style: TextStyle(
              color: Colors.white,
              fontSize: 16,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 12),
          Row(
            children: [
              Expanded(
                child: _buildBladeButton(1, '刀片1'),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: _buildBladeButton(2, '刀片2'),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: _buildBladeButton(3, '双刀片'),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildBladeButton(int bladeNumber, String label) {
    final isSelected = _bladeNumber == bladeNumber;
    return GestureDetector(
      onTap: () {
        setState(() {
          _bladeNumber = bladeNumber;
        });
        _updateCuttingParameters();
      },
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 12),
        decoration: BoxDecoration(
          color: isSelected ? Colors.orange[700] : Colors.grey[800],
          borderRadius: BorderRadius.circular(8),
          border: Border.all(
            color: isSelected ? Colors.orange[400] : Colors.grey[600],
            width: 2,
          ),
        ),
        child: Text(
          label,
          textAlign: TextAlign.center,
          style: TextStyle(
            color: isSelected ? Colors.white : Colors.grey[400],
            fontSize: 12,
            fontWeight: FontWeight.w500,
          ),
        ),
      ),
    );
  }

  Widget _buildQualityControl() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.grey[900],
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.white24),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            '质量阈值',
            style: TextStyle(
              color: Colors.white,
              fontSize: 16,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 12),
          Row(
            children: [
              Expanded(
                child: Text(
                  '${(_qualityThreshold * 100).toStringAsFixed(0)}%',
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ],
          ),
          Slider(
            value: _qualityThreshold,
            min: 0.1,
            max: 1.0,
            divisions: 90,
            activeColor: Colors.green[400],
            inactiveColor: Colors.grey[600],
            onChanged: (value) {
              setState(() {
                _qualityThreshold = value;
              });
              _updateCuttingParameters();
            },
          ),
        ],
      ),
    );
  }

  Widget _buildActionButtons() {
    return Column(
      children: [
        SizedBox(
          width: double.infinity,
          child: ElevatedButton.icon(
            onPressed: _isCutting ? null : _startCutting,
            icon: Icon(_isCutting ? Icons.pause : Icons.play_arrow),
            label: Text(_isCutting ? '切割中...' : '开始切割'),
            style: ElevatedButton.styleFrom(
              backgroundColor: _isCutting ? Colors.grey[600] : Colors.green[700],
              foregroundColor: Colors.white,
              padding: const EdgeInsets.symmetric(vertical: 16),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(12),
              ),
            ),
          ),
        ),
        const SizedBox(height: 12),
        SizedBox(
          width: double.infinity,
          child: ElevatedButton.icon(
            onPressed: _isCutting ? _stopCutting : null,
            icon: const Icon(Icons.stop),
            label: const Text('停止切割'),
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.red[700],
              foregroundColor: Colors.white,
              padding: const EdgeInsets.symmetric(vertical: 16),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(12),
              ),
            ),
          ),
        ),
      ],
    );
  }

  void _adjustCoordinate(double delta) {
    setState(() {
      _xCoordinate = (_xCoordinate + delta).clamp(0.0, 1000.0);
    });
    _updateCuttingParameters();
  }

  void _updateCuttingParameters() {
    widget.onCuttingParametersChanged(_xCoordinate, _bladeNumber, _qualityThreshold);
  }

  void _startCutting() {
    setState(() {
      _isCutting = true;
    });
    // 这里可以添加开始切割的逻辑
  }

  void _stopCutting() {
    setState(() {
      _isCutting = false;
    });
    // 这里可以添加停止切割的逻辑
  }
} 