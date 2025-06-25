#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能切竹机 - 视觉处理器基类
定义统一的视觉处理接口，支持多种算法实现
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import time
import numpy as np
import cv2
import logging

from .vision_types import (
    DetectionResult, CalibrationData, AlgorithmConfig,
    ProcessingROI, BoundingBox, Point2D
)

logger = logging.getLogger(__name__)


class VisionProcessor(ABC):
    """视觉处理器基类"""
    
    def __init__(self, config: AlgorithmConfig, calibration: Optional[CalibrationData] = None):
        """
        初始化视觉处理器
        Args:
            config: 算法配置参数
            calibration: 相机标定数据
        """
        self.config = config
        self.calibration = calibration
        self.is_initialized = False
        self.processing_stats = {
            'total_processed': 0,
            'total_time': 0.0,
            'avg_processing_time': 0.0
        }
        
        # 验证配置
        if not config.validate():
            raise ValueError("算法配置参数无效")
        
        logger.info(f"{self.__class__.__name__} 初始化完成")
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        初始化处理器
        Returns:
            初始化是否成功
        """
        pass
    
    @abstractmethod
    def process_image(self, image: np.ndarray, roi: Optional[ProcessingROI] = None) -> DetectionResult:
        """
        处理图像，检测竹节和分割竹筒
        Args:
            image: 输入图像
            roi: 感兴趣区域 (可选)
        Returns:
            检测结果
        """
        pass
    
    @abstractmethod
    def cleanup(self):
        """清理资源"""
        pass
    
    def set_config(self, config: AlgorithmConfig):
        """
        更新算法配置
        Args:
            config: 新的配置参数
        """
        if not config.validate():
            raise ValueError("算法配置参数无效")
        
        self.config = config
        logger.info("算法配置已更新")
    
    def set_calibration(self, calibration: CalibrationData):
        """
        设置相机标定数据
        Args:
            calibration: 标定数据
        """
        self.calibration = calibration
        logger.info("相机标定数据已更新")
    
    def process_with_timing(self, image: np.ndarray, roi: Optional[ProcessingROI] = None) -> DetectionResult:
        """
        带计时的图像处理
        Args:
            image: 输入图像
            roi: 感兴趣区域
        Returns:
            检测结果（包含处理时间）
        """
        start_time = time.time()
        
        try:
            result = self.process_image(image, roi)
            processing_time = time.time() - start_time
            
            # 更新统计信息
            self._update_stats(processing_time)
            
            # 设置处理时间
            result.processing_time = processing_time
            
            logger.debug(f"图像处理完成，耗时: {processing_time:.3f}s")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"图像处理失败，耗时: {processing_time:.3f}s, 错误: {e}")
            raise
    
    def _update_stats(self, processing_time: float):
        """更新处理统计信息"""
        self.processing_stats['total_processed'] += 1
        self.processing_stats['total_time'] += processing_time
        self.processing_stats['avg_processing_time'] = (
            self.processing_stats['total_time'] / self.processing_stats['total_processed']
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        获取性能统计信息
        Returns:
            性能统计数据
        """
        return self.processing_stats.copy()
    
    def reset_stats(self):
        """重置统计信息"""
        self.processing_stats = {
            'total_processed': 0,
            'total_time': 0.0,
            'avg_processing_time': 0.0
        }
        logger.info("性能统计信息已重置")
    
    def validate_image(self, image: np.ndarray) -> bool:
        """
        验证输入图像
        Args:
            image: 输入图像
        Returns:
            图像是否有效
        """
        if image is None:
            logger.error("输入图像为None")
            return False
        
        if len(image.shape) not in [2, 3]:
            logger.error(f"不支持的图像维度: {image.shape}")
            return False
        
        if image.size == 0:
            logger.error("输入图像大小为0")
            return False
        
        return True
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        图像预处理
        Args:
            image: 输入图像
        Returns:
            预处理后的图像
        """
        if not self.validate_image(image):
            raise ValueError("输入图像无效")
        
        # 转换为灰度图像（如果需要）
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 高斯滤波
        if self.config.gaussian_blur_kernel > 1:
            gray = cv2.GaussianBlur(gray, 
                                  (self.config.gaussian_blur_kernel, self.config.gaussian_blur_kernel), 
                                  0)
        
        # 双边滤波
        if self.config.bilateral_filter_d > 0:
            gray = cv2.bilateralFilter(gray, 
                                     self.config.bilateral_filter_d,
                                     self.config.bilateral_sigma_color,
                                     self.config.bilateral_sigma_space)
        
        return gray
    
    def apply_calibration_correction(self, image: np.ndarray) -> np.ndarray:
        """
        应用相机标定校正
        Args:
            image: 输入图像
        Returns:
            校正后的图像
        """
        if self.calibration is None:
            return image
        
        # 畸变校正
        if self.calibration.camera_matrix is not None and self.calibration.dist_coeffs is not None:
            h, w = image.shape[:2]
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                self.calibration.camera_matrix, 
                self.calibration.dist_coeffs, 
                (w, h), 1, (w, h)
            )
            
            corrected = cv2.undistort(image, 
                                    self.calibration.camera_matrix,
                                    self.calibration.dist_coeffs,
                                    None, 
                                    new_camera_matrix)
            return corrected
        
        return image
    
    def extract_roi(self, image: np.ndarray, roi: ProcessingROI) -> np.ndarray:
        """
        提取感兴趣区域
        Args:
            image: 输入图像
            roi: ROI定义
        Returns:
            ROI图像
        """
        return roi.extract_from_image(image)
    
    def pixel_to_mm_conversion(self, pixel_coord: Point2D) -> Point2D:
        """
        像素坐标转毫米坐标
        Args:
            pixel_coord: 像素坐标
        Returns:
            毫米坐标
        """
        if self.calibration is None:
            # 使用默认比例
            default_ratio = 0.5  # 0.5mm/pixel
            return Point2D(pixel_coord.x * default_ratio, pixel_coord.y * default_ratio)
        
        return self.calibration.pixel_to_mm(pixel_coord)
    
    def create_debug_image(self, original: np.ndarray, processed: np.ndarray, 
                          result: DetectionResult) -> np.ndarray:
        """
        创建调试图像
        Args:
            original: 原始图像
            processed: 处理后的图像
            result: 检测结果
        Returns:
            调试图像
        """
        # 创建组合图像
        h, w = original.shape[:2]
        
        # 确保processed是3通道
        if len(processed.shape) == 2:
            processed_color = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        else:
            processed_color = processed.copy()
        
        # 在原图上绘制结果
        from .vision_types import draw_detection_result
        result_image = draw_detection_result(original.copy(), result)
        
        # 水平拼接
        debug_image = np.hstack([original, processed_color, result_image])
        
        # 添加标题
        cv2.putText(debug_image, "Original", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(debug_image, "Processed", (w + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(debug_image, "Result", (2*w + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return debug_image


class VisionProcessorFactory:
    """视觉处理器工厂类"""
    
    _processors = {}
    
    @classmethod
    def register_processor(cls, name: str, processor_class):
        """
        注册处理器类
        Args:
            name: 处理器名称
            processor_class: 处理器类
        """
        cls._processors[name] = processor_class
        logger.info(f"视觉处理器 '{name}' 已注册")
    
    @classmethod
    def create_processor(cls, name: str, config: AlgorithmConfig, 
                        calibration: Optional[CalibrationData] = None) -> VisionProcessor:
        """
        创建视觉处理器实例
        Args:
            name: 处理器名称
            config: 算法配置
            calibration: 标定数据
        Returns:
            处理器实例
        """
        if name not in cls._processors:
            raise ValueError(f"未知的视觉处理器: {name}")
        
        processor_class = cls._processors[name]
        return processor_class(config, calibration)
    
    @classmethod
    def list_processors(cls) -> list:
        """
        列出所有可用的处理器
        Returns:
            处理器名称列表
        """
        return list(cls._processors.keys())


# 性能监控装饰器
def monitor_performance(func):
    """性能监控装饰器"""
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        try:
            result = func(self, *args, **kwargs)
            processing_time = time.time() - start_time
            logger.debug(f"{func.__name__} 执行时间: {processing_time:.3f}s")
            return result
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"{func.__name__} 执行失败，耗时: {processing_time:.3f}s, 错误: {e}")
            raise
    return wrapper 