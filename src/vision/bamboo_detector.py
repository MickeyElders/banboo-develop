#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能切竹机 - AI视觉识别模块
集成多种检测算法，提供统一的竹节检测接口
"""

import cv2
import numpy as np
import time
import logging
import json
from typing import List, Tuple, Dict, Optional

from .vision_processor import VisionProcessorFactory
from .vision_types import (
    DetectionResult, BambooNode, BambooSegment, CuttingPoint,
    AlgorithmConfig, CalibrationData, create_default_config,
    draw_detection_result
)

# 确保处理器被注册
try:
    from .traditional_detector import register_traditional_detector
    register_traditional_detector()
except ImportError as e:
    logger.warning(f"无法导入传统检测器: {e}")

logger = logging.getLogger(__name__)


class BambooDetector:
    """
    竹节检测器 - 主要接口类
    支持多种检测算法，提供统一的检测接口
    """
    
    def __init__(self, algorithm: str = "traditional", config: Optional[Dict] = None):
        """
        初始化检测器
        Args:
            algorithm: 算法类型 ("traditional", "deep_learning", "hybrid")
            config: 配置参数字典
        """
        self.algorithm_name = algorithm
        self.config = self._create_algorithm_config(config)
        self.calibration = None
        self.processor = None
        self.is_initialized = False
        
        logger.info(f"竹节检测器初始化完成，使用算法: {algorithm}")
    
    def _create_algorithm_config(self, config: Optional[Dict]) -> AlgorithmConfig:
        """
        创建算法配置
        Args:
            config: 用户配置字典
        Returns:
            算法配置对象
        """
        if config is None:
            return create_default_config()
        
        # 从字典创建配置对象
        algo_config = create_default_config()
        
        # 更新配置参数
        for key, value in config.items():
            if hasattr(algo_config, key):
                setattr(algo_config, key, value)
        
        return algo_config
    
    def set_calibration(self, calibration_data: Dict):
        """
        设置相机标定数据
        Args:
            calibration_data: 标定数据字典
        """
        try:
            # 从字典创建标定对象
            from .vision_types import CalibrationData, BoundingBox
            
            self.calibration = CalibrationData(
                pixel_to_mm_ratio=calibration_data.get('pixel_to_mm_ratio', 0.5),
                camera_matrix=np.array(calibration_data.get('camera_matrix', np.eye(3))),
                dist_coeffs=np.array(calibration_data.get('dist_coeffs', np.zeros(5))),
                rotation_angle=calibration_data.get('rotation_angle', 0.0),
                roi_bounds=BoundingBox(0, 0, 1920, 1080)  # 默认ROI
            )
            
            if self.processor:
                self.processor.set_calibration(self.calibration)
            
            logger.info("相机标定数据已设置")
            
        except Exception as e:
            logger.error(f"设置标定数据失败: {e}")
    
    def initialize(self) -> bool:
        """
        初始化检测器
        Returns:
            初始化是否成功
        """
        try:
            # 创建视觉处理器
            self.processor = VisionProcessorFactory.create_processor(
                self.algorithm_name, 
                self.config, 
                self.calibration
            )
            
            # 初始化处理器
            if not self.processor.initialize():
                logger.error("视觉处理器初始化失败")
                return False
            
            self.is_initialized = True
            logger.info(f"检测器初始化成功，算法: {self.algorithm_name}")
            return True
            
        except Exception as e:
            logger.error(f"检测器初始化失败: {e}")
            return False
    
    def detect_nodes(self, image: np.ndarray, roi: Optional[Dict] = None) -> DetectionResult:
        """
        检测竹节
        Args:
            image: 输入图像 (BGR格式)
            roi: 感兴趣区域 (可选)
        Returns:
            检测结果
        """
        if not self.is_initialized:
            if not self.initialize():
                raise RuntimeError("检测器未正确初始化")
        
        try:
            # 转换ROI格式
            processing_roi = None
            if roi:
                from .vision_types import ProcessingROI, BoundingBox
                processing_roi = ProcessingROI(
                    bbox=BoundingBox(
                        roi.get('x', 0),
                        roi.get('y', 0),
                        roi.get('x', 0) + roi.get('width', image.shape[1]),
                        roi.get('y', 0) + roi.get('height', image.shape[0])
                    ),
                    roi_type=roi.get('type', 'manual')
                )
            
            # 执行检测
            result = self.processor.process_with_timing(image, processing_roi)
            
            logger.info(f"检测完成: {result.total_nodes}个竹节, {result.total_segments}个竹筒段, "
                       f"{len(result.cutting_points)}个切割点, 耗时{result.processing_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"竹节检测失败: {e}")
            raise
    
    def switch_algorithm(self, algorithm: str) -> bool:
        """
        切换检测算法
        Args:
            algorithm: 新的算法名称
        Returns:
            切换是否成功
        """
        try:
            # 清理旧的处理器
            if self.processor:
                self.processor.cleanup()
            
            # 创建新的处理器
            self.algorithm_name = algorithm
            self.processor = VisionProcessorFactory.create_processor(
                algorithm, 
                self.config, 
                self.calibration
            )
            
            # 初始化新处理器
            if self.processor.initialize():
                logger.info(f"成功切换到算法: {algorithm}")
                return True
            else:
                logger.error(f"新算法初始化失败: {algorithm}")
                return False
                
        except Exception as e:
            logger.error(f"算法切换失败: {e}")
            return False
    
    def get_available_algorithms(self) -> List[str]:
        """
        获取可用的算法列表
        Returns:
            算法名称列表
        """
        return VisionProcessorFactory.list_processors()
    
    def get_performance_stats(self) -> Dict:
        """
        获取性能统计信息
        Returns:
            性能统计数据
        """
        if self.processor:
            return self.processor.get_performance_stats()
        return {}
    
    def visualize_result(self, image: np.ndarray, result: DetectionResult,
                        show_nodes: bool = True, show_segments: bool = True, 
                        show_cutting_points: bool = True) -> np.ndarray:
        """
        可视化检测结果
        Args:
            image: 原始图像
            result: 检测结果
            show_nodes: 是否显示竹节
            show_segments: 是否显示竹筒段
            show_cutting_points: 是否显示切割点
        Returns:
            可视化后的图像
        """
        try:
            # 使用内置的绘制函数
            vis_image = draw_detection_result(
                image, result, show_nodes, show_segments, show_cutting_points
            )
            
            # 添加性能统计信息
            info_text = [
                f"Algorithm: {self.algorithm_name}",
                f"Nodes: {result.total_nodes}",
                f"Segments: {result.total_segments} (Usable: {len(result.usable_segments)})",
                f"Cuts: {len(result.cutting_points)}",
                f"Total Length: {result.total_usable_length:.1f}mm",
                f"Processing Time: {result.processing_time:.3f}s"
            ]
            
            # 添加文本背景
            overlay = vis_image.copy()
            cv2.rectangle(overlay, (10, 10), (350, 200), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, vis_image, 0.3, 0, vis_image)
            
            # 绘制文本
            for i, text in enumerate(info_text):
                cv2.putText(vis_image, text, (15, 35 + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            return vis_image
            
        except Exception as e:
            logger.error(f"可视化失败: {e}")
            return image
    
    def create_debug_image(self, original_image: np.ndarray, result: DetectionResult) -> np.ndarray:
        """
        创建调试图像
        Args:
            original_image: 原始图像
            result: 检测结果
        Returns:
            调试图像
        """
        if self.processor:
            # 获取处理后的图像
            processed_image = self.processor.preprocess_image(original_image)
            return self.processor.create_debug_image(original_image, processed_image, result)
        else:
            return self.visualize_result(original_image, result)
    
    def _convert_to_json_serializable(self, obj):
        """
        将numpy数据类型转换为JSON可序列化的类型
        Args:
            obj: 要转换的对象
        Returns:
            转换后的对象
        """
        import numpy as np
        
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self._convert_to_json_serializable(item) for item in obj]
        else:
            return obj
    
    def save_result(self, result: DetectionResult, filename: str):
        """
        保存检测结果到文件
        Args:
            result: 检测结果
            filename: 文件名
        """
        try:
            # 转换为可序列化的格式
            data = {
                'algorithm': self.algorithm_name,
                'processing_time': float(result.processing_time),
                'image_shape': [int(x) for x in result.image_shape],
                'nodes': [
                    {
                        'position': {'x': float(node.position.x), 'y': float(node.position.y)},
                        'position_mm': {'x': float(node.position_mm.x), 'y': float(node.position_mm.y)},
                        'confidence': float(node.confidence),
                        'node_type': node.node_type.value,
                        'width': float(node.width),
                        'width_mm': float(node.width_mm),
                        'bbox': {
                            'x1': float(node.bbox.x1), 'y1': float(node.bbox.y1),
                            'x2': float(node.bbox.x2), 'y2': float(node.bbox.y2)
                        },
                        'features': self._convert_to_json_serializable(node.features)
                    }
                    for node in result.nodes
                ],
                'segments': [
                    {
                        'start_pos': {'x': float(seg.start_pos.x), 'y': float(seg.start_pos.y)},
                        'end_pos': {'x': float(seg.end_pos.x), 'y': float(seg.end_pos.y)},
                        'start_pos_mm': {'x': float(seg.start_pos_mm.x), 'y': float(seg.start_pos_mm.y)},
                        'end_pos_mm': {'x': float(seg.end_pos_mm.x), 'y': float(seg.end_pos_mm.y)},
                        'length_mm': float(seg.length_mm),
                        'diameter_mm': float(seg.diameter_mm),
                        'quality': seg.quality.value,
                        'quality_score': float(seg.quality_score),
                        'defects': [str(d) for d in seg.defects],
                        'is_usable': bool(seg.is_usable)
                    }
                    for seg in result.segments
                ],
                'cutting_points': [
                    {
                        'position': {'x': float(cp.position.x), 'y': float(cp.position.y)},
                        'position_mm': {'x': float(cp.position_mm.x), 'y': float(cp.position_mm.y)},
                        'cutting_type': cp.cutting_type.value,
                        'priority': int(cp.priority),
                        'confidence': float(cp.confidence),
                        'reason': str(cp.reason)
                    }
                    for cp in result.cutting_points
                ],
                'statistics': {
                    'total_nodes': int(result.total_nodes),
                    'total_segments': int(result.total_segments),
                    'usable_segments': int(len(result.usable_segments)),
                    'total_usable_length': float(result.total_usable_length),
                    'high_priority_cuts': int(len(result.get_high_priority_cuts()))
                },
                'metadata': self._convert_to_json_serializable(result.metadata)
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"检测结果已保存到: {filename}")
            
        except Exception as e:
            logger.error(f"保存结果失败: {e}")
    
    def load_result(self, filename: str) -> DetectionResult:
        """
        从文件加载检测结果
        Args:
            filename: 文件名
        Returns:
            检测结果
        """
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 重建数据结构
            from .vision_types import (
                Point2D, BoundingBox, NodeType, SegmentQuality, CuttingType
            )
            
            # 重建竹节
            nodes = []
            for node_data in data['nodes']:
                node = BambooNode(
                    position=Point2D(**node_data['position']),
                    position_mm=Point2D(**node_data['position_mm']),
                    confidence=node_data['confidence'],
                    node_type=NodeType(node_data['node_type']),
                    bbox=BoundingBox(**node_data['bbox']),
                    width=node_data['width'],
                    width_mm=node_data['width_mm'],
                    features=node_data['features']
                )
                nodes.append(node)
            
            # 重建竹筒段
            segments = []
            for seg_data in data['segments']:
                segment = BambooSegment(
                    start_pos=Point2D(**seg_data['start_pos']),
                    end_pos=Point2D(**seg_data['end_pos']),
                    start_pos_mm=Point2D(**seg_data['start_pos_mm']),
                    end_pos_mm=Point2D(**seg_data['end_pos_mm']),
                    length_mm=seg_data['length_mm'],
                    diameter_mm=seg_data['diameter_mm'],
                    quality=SegmentQuality(seg_data['quality']),
                    quality_score=seg_data['quality_score'],
                    defects=seg_data['defects'],
                    bbox=BoundingBox(0, 0, 0, 0)  # 简化
                )
                segments.append(segment)
            
            # 重建切割点
            cutting_points = []
            for cp_data in data['cutting_points']:
                cutting_point = CuttingPoint(
                    position=Point2D(**cp_data['position']),
                    position_mm=Point2D(**cp_data['position_mm']),
                    cutting_type=CuttingType(cp_data['cutting_type']),
                    priority=cp_data['priority'],
                    confidence=cp_data['confidence'],
                    reason=cp_data['reason']
                )
                cutting_points.append(cutting_point)
            
            # 创建检测结果
            result = DetectionResult(
                nodes=nodes,
                segments=segments,
                cutting_points=cutting_points,
                processing_time=data['processing_time'],
                image_shape=tuple(data['image_shape']),
                metadata=data.get('metadata', {})
            )
            
            logger.info(f"检测结果已从文件加载: {filename}")
            return result
            
        except Exception as e:
            logger.error(f"加载结果失败: {e}")
            raise
    
    def cleanup(self):
        """清理资源"""
        if self.processor:
            self.processor.cleanup()
            self.processor = None
        
        self.is_initialized = False
        logger.info("竹节检测器资源清理完成")


# 兼容性函数，保持向后兼容
def create_bamboo_detector(algorithm: str = "traditional", config: Optional[Dict] = None) -> BambooDetector:
    """
    创建竹节检测器实例
    Args:
        algorithm: 算法类型
        config: 配置参数
    Returns:
        检测器实例
    """
    return BambooDetector(algorithm, config)


# 主函数用于测试
def main():
    """测试主函数"""
    try:
        # 创建检测器
        detector = BambooDetector("traditional")
        
        # 初始化
        if not detector.initialize():
            print("检测器初始化失败")
            return
        
        # 测试图像
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 检测
        result = detector.detect_nodes(test_image)
        
        # 显示结果
        print(f"检测到 {result.total_nodes} 个竹节")
        print(f"分割出 {result.total_segments} 个竹筒段")
        print(f"推荐 {len(result.cutting_points)} 个切割点")
        
        # 可视化
        vis_image = detector.visualize_result(test_image, result)
        
        # 清理
        detector.cleanup()
        
    except Exception as e:
        logger.error(f"测试失败: {e}")


if __name__ == "__main__":
    main() 