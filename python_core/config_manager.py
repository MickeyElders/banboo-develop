"""
配置管理器 - Python层
负责系统配置的加载、验证和管理
"""

import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    """摄像头配置"""
    device_id: str = "/dev/video0"
    width: int = 1920
    height: int = 1080
    fps: int = 30
    use_stereo: bool = False
    stereo_device_id: str = "/dev/video1"
    calibration_file: str = "config/stereo_calibration.xml"
    

@dataclass  
class AIConfig:
    """AI推理配置"""
    model_path: str = "./models/best.pt"
    engine_path: str = "./models/bamboo_detection.trt"
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    max_detections: int = 10
    use_tensorrt: bool = True
    use_fp16: bool = True
    batch_size: int = 1


@dataclass
class ModbusConfig:
    """Modbus服务器配置"""
    ip: str = "0.0.0.0"
    port: int = 502
    max_connections: int = 10
    timeout: int = 30
    

@dataclass
class UIConfig:
    """UI界面配置"""
    use_gui: bool = True
    fullscreen: bool = True
    width: int = 1024
    height: int = 600
    theme: str = "dark"
    font_size: int = 14
    

@dataclass
class SystemConfig:
    """系统配置"""
    log_level: str = "INFO"
    max_log_files: int = 10
    enable_monitoring: bool = True
    auto_restart: bool = True


class ConfigManager:
    """配置管理器"""
    
    def __init__(self):
        self.camera_config = CameraConfig()
        self.ai_config = AIConfig()
        self.modbus_config = ModbusConfig()
        self.ui_config = UIConfig()
        self.system_config = SystemConfig()
        self.config_loaded = False
        
    def load_config(self, config_file: str) -> bool:
        """加载配置文件"""
        try:
            config_path = Path(config_file)
            if not config_path.exists():
                logger.warning(f"配置文件不存在: {config_file}，使用默认配置")
                return True
                
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() == '.yaml':
                    config_data = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    config_data = json.load(f)
                else:
                    logger.error(f"不支持的配置文件格式: {config_path.suffix}")
                    return False
                    
            # 解析配置
            self._parse_config(config_data)
            self.config_loaded = True
            
            logger.info(f"配置文件加载成功: {config_file}")
            return True
            
        except Exception as e:
            logger.error(f"配置文件加载失败: {e}")
            return False
            
    def _parse_config(self, config_data: Dict[str, Any]):
        """解析配置数据"""
        # 摄像头配置
        if 'camera' in config_data:
            camera_data = config_data['camera']
            self.camera_config = CameraConfig(
                device_id=camera_data.get('device_id', self.camera_config.device_id),
                width=camera_data.get('width', self.camera_config.width),
                height=camera_data.get('height', self.camera_config.height),
                fps=camera_data.get('fps', self.camera_config.fps),
                use_stereo=camera_data.get('use_stereo', self.camera_config.use_stereo),
                stereo_device_id=camera_data.get('stereo_device_id', self.camera_config.stereo_device_id),
                calibration_file=camera_data.get('calibration_file', self.camera_config.calibration_file)
            )
            
        # AI配置
        if 'ai' in config_data:
            ai_data = config_data['ai']
            self.ai_config = AIConfig(
                model_path=ai_data.get('model_path', self.ai_config.model_path),
                engine_path=ai_data.get('engine_path', self.ai_config.engine_path),
                confidence_threshold=ai_data.get('confidence_threshold', self.ai_config.confidence_threshold),
                nms_threshold=ai_data.get('nms_threshold', self.ai_config.nms_threshold),
                max_detections=ai_data.get('max_detections', self.ai_config.max_detections),
                use_tensorrt=ai_data.get('use_tensorrt', self.ai_config.use_tensorrt),
                use_fp16=ai_data.get('use_fp16', self.ai_config.use_fp16),
                batch_size=ai_data.get('batch_size', self.ai_config.batch_size)
            )
            
        # Modbus配置
        if 'modbus' in config_data:
            modbus_data = config_data['modbus']
            self.modbus_config = ModbusConfig(
                ip=modbus_data.get('ip', self.modbus_config.ip),
                port=modbus_data.get('port', self.modbus_config.port),
                max_connections=modbus_data.get('max_connections', self.modbus_config.max_connections),
                timeout=modbus_data.get('timeout', self.modbus_config.timeout)
            )
            
        # UI配置
        if 'ui' in config_data:
            ui_data = config_data['ui']
            self.ui_config = UIConfig(
                use_gui=ui_data.get('use_gui', self.ui_config.use_gui),
                fullscreen=ui_data.get('fullscreen', self.ui_config.fullscreen),
                width=ui_data.get('width', self.ui_config.width),
                height=ui_data.get('height', self.ui_config.height),
                theme=ui_data.get('theme', self.ui_config.theme),
                font_size=ui_data.get('font_size', self.ui_config.font_size)
            )
            
        # 系统配置
        if 'system' in config_data:
            system_data = config_data['system']
            self.system_config = SystemConfig(
                log_level=system_data.get('log_level', self.system_config.log_level),
                max_log_files=system_data.get('max_log_files', self.system_config.max_log_files),
                enable_monitoring=system_data.get('enable_monitoring', self.system_config.enable_monitoring),
                auto_restart=system_data.get('auto_restart', self.system_config.auto_restart)
            )
            
    def save_config(self, config_file: str) -> bool:
        """保存配置到文件"""
        try:
            config_data = {
                'camera': asdict(self.camera_config),
                'ai': asdict(self.ai_config),  
                'modbus': asdict(self.modbus_config),
                'ui': asdict(self.ui_config),
                'system': asdict(self.system_config)
            }
            
            config_path = Path(config_file)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                if config_path.suffix.lower() == '.yaml':
                    yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
                elif config_path.suffix.lower() == '.json':
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
                else:
                    logger.error(f"不支持的配置文件格式: {config_path.suffix}")
                    return False
                    
            logger.info(f"配置文件保存成功: {config_file}")
            return True
            
        except Exception as e:
            logger.error(f"配置文件保存失败: {e}")
            return False
            
    def get_camera_config(self) -> CameraConfig:
        """获取摄像头配置"""
        return self.camera_config
        
    def get_ai_config(self) -> AIConfig:
        """获取AI配置"""
        return self.ai_config
        
    def get_modbus_config(self) -> ModbusConfig:
        """获取Modbus配置"""
        return self.modbus_config
        
    def get_ui_config(self) -> UIConfig:
        """获取UI配置"""
        return self.ui_config
        
    def get_system_config(self) -> SystemConfig:
        """获取系统配置"""
        return self.system_config
        
    def is_loaded(self) -> bool:
        """检查配置是否已加载"""
        return self.config_loaded
        
    def validate_config(self) -> bool:
        """验证配置有效性"""
        try:
            # 验证摄像头配置
            if not Path(self.camera_config.device_id).exists() and not self.camera_config.device_id.startswith('rtsp://'):
                logger.warning(f"摄像头设备不存在: {self.camera_config.device_id}")
                
            # 验证AI模型文件
            if not Path(self.ai_config.model_path).exists():
                logger.warning(f"AI模型文件不存在: {self.ai_config.model_path}")
                
            # 验证端口范围
            if not (1 <= self.modbus_config.port <= 65535):
                logger.error(f"无效的Modbus端口: {self.modbus_config.port}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"配置验证失败: {e}")
            return False