"""
摄像头管理器 - Python实现
负责视频采集、图像预处理和数据管理
"""

import logging
import cv2
import numpy as np
import threading
import time
from typing import Optional, Tuple, Dict, Any, Callable
from queue import Queue, Empty
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    """摄像头配置"""
    device_id: int = 0                    # 摄像头设备ID
    width: int = 1920                     # 图像宽度
    height: int = 1080                    # 图像高度
    fps: int = 30                         # 帧率
    buffer_size: int = 2                  # 缓冲区大小
    format: str = "MJPG"                  # 图像格式
    auto_exposure: bool = True            # 自动曝光
    exposure: int = -1                    # 手动曝光值
    brightness: float = 0.5               # 亮度
    contrast: float = 1.0                 # 对比度
    saturation: float = 1.0               # 饱和度
    use_gstreamer: bool = False           # 是否使用GStreamer
    gstreamer_pipeline: str = ""          # GStreamer管道


@dataclass  
class FrameData:
    """帧数据结构"""
    image: np.ndarray                     # 图像数据
    timestamp: float                      # 时间戳
    frame_id: int                         # 帧ID
    width: int                            # 图像宽度
    height: int                           # 图像高度
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'frame_id': self.frame_id,
            'width': self.width,
            'height': self.height,
            'image_shape': self.image.shape
        }


class CameraManager:
    """摄像头管理器类"""
    
    def __init__(self, config: CameraConfig):
        self.config = config
        self.cap = None
        self.running = False
        self.capture_thread = None
        
        # 帧数据管理
        self.frame_queue = Queue(maxsize=config.buffer_size)
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # 统计信息
        self.frame_count = 0
        self.start_time = time.time()
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0.0
        
        # 回调函数
        self.frame_callback = None
        
    def initialize(self) -> bool:
        """初始化摄像头"""
        try:
            logger.info(f"初始化摄像头设备 {self.config.device_id}...")
            
            if self.config.use_gstreamer and self.config.gstreamer_pipeline:
                # 使用GStreamer管道
                self.cap = cv2.VideoCapture(self.config.gstreamer_pipeline, cv2.CAP_GSTREAMER)
            else:
                # 使用V4L2 (Linux) 或 DirectShow (Windows)
                self.cap = cv2.VideoCapture(self.config.device_id)
            
            if not self.cap.isOpened():
                logger.error("无法打开摄像头设备")
                return False
                
            # 设置摄像头参数
            self._configure_camera()
            
            # 验证配置
            if not self._verify_configuration():
                logger.error("摄像头配置验证失败")
                return False
                
            logger.info("摄像头初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"摄像头初始化异常: {e}")
            return False
            
    def _configure_camera(self):
        """配置摄像头参数"""
        try:
            # 设置分辨率
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            
            # 设置帧率
            self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)
            
            # 设置缓冲区大小
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)
            
            # 设置图像格式
            if self.config.format == "MJPG":
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            elif self.config.format == "YUYV":
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V'))
                
            # 设置曝光
            if self.config.auto_exposure:
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # 自动模式
            else:
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 手动模式
                self.cap.set(cv2.CAP_PROP_EXPOSURE, self.config.exposure)
                
            # 设置其他参数
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, self.config.brightness)
            self.cap.set(cv2.CAP_PROP_CONTRAST, self.config.contrast)
            self.cap.set(cv2.CAP_PROP_SATURATION, self.config.saturation)
            
            logger.info("摄像头参数配置完成")
            
        except Exception as e:
            logger.error(f"摄像头参数配置异常: {e}")
            
    def _verify_configuration(self) -> bool:
        """验证摄像头配置"""
        try:
            # 获取实际配置
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"实际配置: {actual_width}x{actual_height} @ {actual_fps}fps")
            
            # 测试获取一帧图像
            ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.error("无法获取测试帧")
                return False
                
            logger.info(f"测试帧形状: {frame.shape}")
            return True
            
        except Exception as e:
            logger.error(f"摄像头配置验证异常: {e}")
            return False
            
    def start_capture(self) -> bool:
        """开始视频采集"""
        try:
            if self.running:
                logger.warning("视频采集已经在运行")
                return True
                
            if not self.cap or not self.cap.isOpened():
                logger.error("摄像头未初始化")
                return False
                
            self.running = True
            self.start_time = time.time()
            self.last_fps_time = time.time()
            
            # 启动采集线程
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            
            logger.info("视频采集已启动")
            return True
            
        except Exception as e:
            logger.error(f"启动视频采集异常: {e}")
            return False
            
    def stop_capture(self):
        """停止视频采集"""
        try:
            if not self.running:
                return
                
            self.running = False
            
            if self.capture_thread and self.capture_thread.is_alive():
                self.capture_thread.join(timeout=2.0)
                
            logger.info("视频采集已停止")
            
        except Exception as e:
            logger.error(f"停止视频采集异常: {e}")
            
    def _capture_loop(self):
        """采集循环线程"""
        logger.info("采集循环线程已启动")
        
        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    logger.warning("获取帧失败")
                    time.sleep(0.01)
                    continue
                    
                # 创建帧数据
                current_time = time.time()
                frame_data = FrameData(
                    image=frame.copy(),
                    timestamp=current_time,
                    frame_id=self.frame_count,
                    width=frame.shape[1],
                    height=frame.shape[0]
                )
                
                # 更新当前帧
                with self.frame_lock:
                    self.current_frame = frame_data
                    
                # 添加到队列
                try:
                    self.frame_queue.put(frame_data, timeout=0.01)
                except:
                    # 队列满，丢弃最旧的帧
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put(frame_data, timeout=0.01)
                    except:
                        pass
                        
                # 调用回调函数
                if self.frame_callback:
                    try:
                        self.frame_callback(frame_data)
                    except Exception as e:
                        logger.error(f"帧回调异常: {e}")
                        
                # 更新统计
                self.frame_count += 1
                self.fps_counter += 1
                
                # 计算FPS
                if current_time - self.last_fps_time >= 1.0:
                    self.current_fps = self.fps_counter / (current_time - self.last_fps_time)
                    self.fps_counter = 0
                    self.last_fps_time = current_time
                    
            except Exception as e:
                logger.error(f"采集循环异常: {e}")
                time.sleep(0.1)
                
        logger.info("采集循环线程已退出")
        
    def get_latest_frame(self) -> Optional[FrameData]:
        """获取最新帧"""
        try:
            with self.frame_lock:
                return self.current_frame
        except Exception as e:
            logger.error(f"获取最新帧异常: {e}")
            return None
            
    def get_frame_from_queue(self, timeout: float = 0.1) -> Optional[FrameData]:
        """从队列获取帧"""
        try:
            return self.frame_queue.get(timeout=timeout)
        except Empty:
            return None
        except Exception as e:
            logger.error(f"从队列获取帧异常: {e}")
            return None
            
    def set_frame_callback(self, callback: Callable[[FrameData], None]):
        """设置帧回调函数"""
        self.frame_callback = callback
        
    def get_camera_stats(self) -> Dict[str, Any]:
        """获取摄像头统计信息"""
        current_time = time.time()
        runtime = current_time - self.start_time if self.start_time else 0
        
        return {
            'running': self.running,
            'frame_count': self.frame_count,
            'current_fps': self.current_fps,
            'runtime_seconds': runtime,
            'queue_size': self.frame_queue.qsize(),
            'max_queue_size': self.config.buffer_size,
            'avg_fps': self.frame_count / runtime if runtime > 0 else 0.0
        }
        
    def save_frame(self, filename: str, frame_data: Optional[FrameData] = None) -> bool:
        """保存帧到文件"""
        try:
            if frame_data is None:
                frame_data = self.get_latest_frame()
                
            if frame_data is None:
                logger.error("没有可用的帧数据")
                return False
                
            # 确保目录存在
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            
            # 保存图像
            success = cv2.imwrite(filename, frame_data.image)
            if success:
                logger.info(f"帧已保存到: {filename}")
            else:
                logger.error(f"保存帧失败: {filename}")
                
            return success
            
        except Exception as e:
            logger.error(f"保存帧异常: {e}")
            return False
            
    def cleanup(self):
        """清理资源"""
        try:
            # 停止采集
            self.stop_capture()
            
            # 释放摄像头
            if self.cap:
                self.cap.release()
                self.cap = None
                
            # 清空队列
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except:
                    break
                    
            logger.info("摄像头资源清理完成")
            
        except Exception as e:
            logger.error(f"摄像头清理异常: {e}")
            
    def is_running(self) -> bool:
        """检查是否正在运行"""
        return self.running and self.capture_thread and self.capture_thread.is_alive()
        
    def get_config(self) -> CameraConfig:
        """获取配置信息"""
        return self.config