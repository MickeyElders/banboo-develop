import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../../core/providers/system_state_provider.dart';

/// 状态面板组件
class StatusPanel extends StatelessWidget {
  const StatusPanel({super.key});

  @override
  Widget build(BuildContext context) {
    return Consumer<SystemStateProvider>(
      builder: (context, systemProvider, child) {
        final status = systemProvider.currentStatus;
        
        return Container(
          height: 80,
          margin: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: Colors.black87,
            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: Colors.white24, width: 1),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withOpacity(0.3),
                blurRadius: 8,
                offset: const Offset(0, 2),
              ),
            ],
          ),
          child: RepaintBoundary(
            child: Row(
              children: [
                _buildSystemStatus(status),
                const VerticalDivider(color: Colors.white24, width: 1),
                _buildPerformanceMetrics(status),
                const VerticalDivider(color: Colors.white24, width: 1),
                _buildConnectionStatus(status),
                const VerticalDivider(color: Colors.white24, width: 1),
                _buildTimestamp(status),
              ],
            ),
          ),
        );
      },
    );
  }

  Widget _buildSystemStatus(Map<String, dynamic> status) {
    final systemStatus = status['system_status'] ?? 0;
    final statusText = _getSystemStatusText(systemStatus);
    final statusColor = _getSystemStatusColor(systemStatus);
    
    return Expanded(
      flex: 2,
      child: Container(
        padding: const EdgeInsets.all(12),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Container(
                  width: 12,
                  height: 12,
                  decoration: BoxDecoration(
                    color: statusColor,
                    shape: BoxShape.circle,
                  ),
                ),
                const SizedBox(width: 8),
                const Text(
                  '系统状态',
                  style: TextStyle(
                    color: Colors.grey,
                    fontSize: 12,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 4),
            Text(
              statusText,
              style: TextStyle(
                color: statusColor,
                fontSize: 16,
                fontWeight: FontWeight.bold,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildPerformanceMetrics(Map<String, dynamic> status) {
    final fps = status['fps'] ?? 0;
    final cpuUsage = status['cpu_usage'] ?? 0.0;
    final memoryUsage = status['memory_usage'] ?? 0.0;
    final gpuUsage = status['gpu_usage'] ?? 0.0;
    
    return Expanded(
      flex: 3,
      child: Container(
        padding: const EdgeInsets.all(12),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Row(
              children: [
                _buildMetricItem('FPS', '${fps.toStringAsFixed(1)}', Colors.green),
                const SizedBox(width: 16),
                _buildMetricItem('CPU', '${(cpuUsage * 100).toStringAsFixed(1)}%', Colors.blue),
                const SizedBox(width: 16),
                _buildMetricItem('内存', '${(memoryUsage * 100).toStringAsFixed(1)}%', Colors.orange),
                const SizedBox(width: 16),
                _buildMetricItem('GPU', '${(gpuUsage * 100).toStringAsFixed(1)}%', Colors.purple),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildMetricItem(String label, String value, Color color) {
    return Expanded(
      child: Column(
        children: [
          Text(
            label,
            style: const TextStyle(
              color: Colors.grey,
              fontSize: 10,
            ),
          ),
          const SizedBox(height: 2),
          Text(
            value,
            style: TextStyle(
              color: color,
              fontSize: 14,
              fontWeight: FontWeight.bold,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildConnectionStatus(Map<String, dynamic> status) {
    final plcConnected = status['plc_connected'] ?? false;
    final cameraConnected = status['camera_connected'] ?? false;
    final heartbeatCount = status['heartbeat_count'] ?? 0;
    
    return Expanded(
      flex: 2,
      child: Container(
        padding: const EdgeInsets.all(12),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              '连接状态',
              style: TextStyle(
                color: Colors.grey,
                fontSize: 12,
              ),
            ),
            const SizedBox(height: 8),
            Row(
              children: [
                _buildConnectionIndicator('PLC', plcConnected),
                const SizedBox(width: 12),
                _buildConnectionIndicator('摄像头', cameraConnected),
              ],
            ),
            const SizedBox(height: 4),
            Text(
              '心跳: $heartbeatCount',
              style: const TextStyle(
                color: Colors.white,
                fontSize: 10,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildConnectionIndicator(String label, bool connected) {
    return Row(
      children: [
        Container(
          width: 8,
          height: 8,
          decoration: BoxDecoration(
            color: connected ? Colors.green : Colors.red,
            shape: BoxShape.circle,
          ),
        ),
        const SizedBox(width: 4),
        Text(
          label,
          style: TextStyle(
            color: connected ? Colors.green : Colors.red,
            fontSize: 10,
            fontWeight: FontWeight.w500,
          ),
        ),
      ],
    );
  }

  Widget _buildTimestamp(Map<String, dynamic> status) {
    final timestamp = status['timestamp'] ?? DateTime.now();
    final formattedTime = _formatTimestamp(timestamp);
    
    return Expanded(
      flex: 2,
      child: Container(
        padding: const EdgeInsets.all(12),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          crossAxisAlignment: CrossAxisAlignment.end,
          children: [
            const Text(
              '时间',
              style: TextStyle(
                color: Colors.grey,
                fontSize: 12,
              ),
            ),
            const SizedBox(height: 4),
            Text(
              formattedTime,
              style: const TextStyle(
                color: Colors.white,
                fontSize: 14,
                fontWeight: FontWeight.w500,
              ),
            ),
          ],
        ),
      ),
    );
  }

  String _getSystemStatusText(int status) {
    switch (status) {
      case 0:
        return '停止';
      case 1:
        return '运行';
      case 2:
        return '错误';
      case 3:
        return '暂停';
      case 4:
        return '紧急停止';
      case 5:
        return '维护模式';
      default:
        return '未知';
    }
  }

  Color _getSystemStatusColor(int status) {
    switch (status) {
      case 0:
        return Colors.grey;
      case 1:
        return Colors.green;
      case 2:
        return Colors.red;
      case 3:
        return Colors.orange;
      case 4:
        return Colors.red[900]!;
      case 5:
        return Colors.blue;
      default:
        return Colors.grey;
    }
  }

  String _formatTimestamp(dynamic timestamp) {
    if (timestamp is String) {
      return timestamp;
    } else if (timestamp is DateTime) {
      return '${timestamp.hour.toString().padLeft(2, '0')}:${timestamp.minute.toString().padLeft(2, '0')}:${timestamp.second.toString().padLeft(2, '0')}';
    } else {
      return '--:--:--';
    }
  }
} 