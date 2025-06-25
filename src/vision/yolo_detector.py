"""
YOLOv8竹节检测器
结合深度学习模型进行竹节检测
"""

import os
import logging
import time
from typing import List, Optional, Dict, Tuple, Any
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .vision_processor import VisionProcessor, monitor_performance
from .vision_types import (
    Point2D, BoundingBox, BambooNode, BambooSegment, CuttingPoint,
    DetectionResult, ProcessingROI, AlgorithmConfig, CalibrationData,
    NodeType, SegmentQuality, CuttingType
)

logger = logging.getLogger(__name__)


class YOLODetector(VisionProcessor):
    """
    基于YOLOv8的竹节检测器
    """
    
    def __init__(self, config: AlgorithmConfig, calibration: Optional[CalibrationData] = None):
        super().__init__(config, calibration)
        
        # YOLO模型相关
        self.model: Optional[YOLO] = None
        self.model_path = "models/bamboo_yolo.pt"  # 默认模型路径
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 检测参数
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.45
        self.max_detections = 50
        
        # 类别映射
        self.class_mapping = {
            0: NodeType.NATURAL_NODE,
            1: NodeType.ARTIFICIAL_NODE,
            2: NodeType.BRANCH_NODE,
            3: NodeType.DAMAGE_NODE
        }
        
        # 性能统计
        self.inference_times = []
        self.preprocessing_times = []
        
        logger.info("YOLOv8检测器初始化完成")
    
    def initialize(self) -> bool:
        """
        初始化YOLO检测器
        Returns:
            是否成功初始化
        """
        try:
            # 检查模型文件
            if not self._check_model_file():
                logger.warning("未找到训练好的模型，将使用预训练模型")
                return self._initialize_pretrained_model()
            
            # 加载自定义模型
            logger.info(f"加载模型: {self.model_path}")
            self.model = YOLO(self.model_path)
            
            # 设置设备
            if torch.cuda.is_available():
                logger.info(f"使用GPU加速: {torch.cuda.get_device_name()}")
            else:
                logger.info("使用CPU推理")
            
            # 模型预热
            self._warmup_model()
            
            self.is_initialized = True
            logger.info("YOLO检测器初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"YOLO检测器初始化失败: {e}")
            return False
    
    def _check_model_file(self) -> bool:
        """检查模型文件是否存在"""
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        
        if Path(self.model_path).exists():
            return True
        
        # 检查其他可能的模型文件
        possible_models = [
            "models/bamboo_yolov8n.pt",
            "models/bamboo_yolov8s.pt", 
            "models/bamboo_yolov8m.pt",
            "models/bamboo_yolov8l.pt",
            "models/best.pt",
            "runs/detect/train/weights/best.pt"
        ]
        
        for model_path in possible_models:
            if Path(model_path).exists():
                self.model_path = model_path
                logger.info(f"找到模型文件: {model_path}")
                return True
        
        return False
    
    def _initialize_pretrained_model(self) -> bool:
        """
        初始化预训练模型（用于演示或迁移学习）
        """
        try:
            logger.info("初始化YOLOv8预训练模型用于演示")
            
            # 使用YOLOv8n作为基础模型
            self.model = YOLO('yolov8n.pt')
            
            # 设置检测阈值（较低，因为不是专门训练的）
            self.confidence_threshold = 0.3
            
            # 创建模型保存目录
            Path("models").mkdir(exist_ok=True)
            
            logger.info("预训练模型初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"预训练模型初始化失败: {e}")
            return False
    
    def _warmup_model(self):
        """模型预热"""
        try:
            logger.info("模型预热中...")
            dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
            
            # 执行几次推理预热
            for i in range(3):
                _ = self.model(dummy_image, verbose=False)
            
            logger.info("模型预热完成")
            
        except Exception as e:
            logger.warning(f"模型预热失败: {e}")
    
    @monitor_performance
    def process_image(self, image: np.ndarray, roi: Optional[ProcessingROI] = None) -> DetectionResult:
        """
        使用YOLO模型处理图像
        Args:
            image: 输入图像
            roi: 感兴趣区域（可选）
        Returns:
            检测结果
        """
        try:
            logger.debug("开始YOLO图像处理")
            
            # 预处理
            start_time = time.time()
            processed_image = self.preprocess_image(image)
            preprocess_time = time.time() - start_time
            self.preprocessing_times.append(preprocess_time)
            
            # YOLO推理
            start_time = time.time()
            nodes = self._yolo_inference(processed_image)
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            # 后处理 - 分割竹筒段
            segments = []
            if len(nodes) > 1:
                segments = self._segment_bamboo_tubes_yolo(processed_image, nodes)
            else:
                logger.warning("竹节数量不足，无法分割竹筒段")
            
            # 计算切割点
            cutting_points = self._calculate_cutting_points_yolo(nodes, segments)
            
            # 创建结果
            result = DetectionResult(
                nodes=nodes,
                segments=segments,
                cutting_points=cutting_points,
                processing_time=0.0,  # 将在装饰器中设置
                image_shape=image.shape[:2],
                metadata={
                    'algorithm': 'yolov8',
                    'model_path': self.model_path,
                    'device': self.device,
                    'confidence_threshold': self.confidence_threshold,
                    'inference_time': inference_time,
                    'preprocess_time': preprocess_time,
                    'num_detections': len(nodes)
                }
            )
            
            logger.info(f"YOLO检测完成: {len(nodes)}个竹节, {len(segments)}个竹筒段, {len(cutting_points)}个切割点")
            
            return result
            
        except Exception as e:
            logger.error(f"YOLO图像处理失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # 返回空结果
            return DetectionResult(
                nodes=[],
                segments=[],
                cutting_points=[],
                processing_time=0.0,
                image_shape=image.shape[:2],
                metadata={'error': str(e), 'algorithm': 'yolov8'}
            )
    
    def _yolo_inference(self, image: np.ndarray) -> List[BambooNode]:
        """
        YOLO模型推理
        Args:
            image: 预处理后的图像
        Returns:
            检测到的竹节列表
        """
        nodes = []
        
        try:
            # YOLO推理
            results = self.model(
                image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                max_det=self.max_detections,
                verbose=False
            )
            
            # 解析结果
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                
                # 提取检测框信息
                xyxy = boxes.xyxy.cpu().numpy()  # 边界框坐标
                conf = boxes.conf.cpu().numpy()  # 置信度
                cls = boxes.cls.cpu().numpy()   # 类别
                
                for i in range(len(xyxy)):
                    box = xyxy[i]
                    confidence = float(conf[i])
                    class_id = int(cls[i])
                    
                    # 计算中心点和尺寸
                    x1, y1, x2, y2 = box
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    width = x2 - x1
                    height = y2 - y1
                    
                    # 映射节点类型
                    node_type = self.class_mapping.get(class_id, NodeType.UNKNOWN)
                    
                    # 创建竹节
                    node = BambooNode(
                        position=Point2D(float(center_x), float(center_y)),
                        position_mm=Point2D(
                            float(center_x) * self.calibration.pixel_to_mm_ratio if self.calibration else float(center_x),
                            float(center_y) * self.calibration.pixel_to_mm_ratio if self.calibration else float(center_y)
                        ),
                        confidence=confidence,
                        node_type=node_type,
                        bbox=BoundingBox(float(x1), float(y1), float(x2), float(y2)),
                        width=float(width),
                        width_mm=float(width) * self.calibration.pixel_to_mm_ratio if self.calibration else float(width),
                        features={
                            'class_id': class_id,
                            'box_area': float(width * height),
                            'aspect_ratio': float(width / max(height, 1)),
                            'detection_method': 'yolov8'
                        }
                    )
                    
                    nodes.append(node)
                    
                    logger.debug(f"YOLO检测到竹节: 位置({center_x:.1f}, {center_y:.1f}), "
                               f"类型={node_type.value}, 置信度={confidence:.3f}")
            
            # 按x坐标排序
            nodes.sort(key=lambda n: n.position.x)
            
            logger.info(f"YOLO检测到 {len(nodes)} 个竹节")
            
        except Exception as e:
            logger.error(f"YOLO推理失败: {e}")
        
        return nodes
    
    def _segment_bamboo_tubes_yolo(self, image: np.ndarray, nodes: List[BambooNode]) -> List[BambooSegment]:
        """
        基于YOLO检测结果分割竹筒段
        """
        segments = []
        
        try:
            if len(nodes) < 2:
                return segments
            
            # 按x坐标排序的节点
            sorted_nodes = sorted(nodes, key=lambda n: n.position.x)
            
            # 创建相邻节点间的竹筒段
            for i in range(len(sorted_nodes) - 1):
                start_node = sorted_nodes[i]
                end_node = sorted_nodes[i + 1]
                
                # 计算长度
                length_pixels = abs(end_node.position.x - start_node.position.x)
                length_mm = length_pixels * self.calibration.pixel_to_mm_ratio if self.calibration else length_pixels
                
                # 估算直径（基于节点高度）
                diameter_pixels = (start_node.bbox.y2 - start_node.bbox.y1 + 
                                 end_node.bbox.y2 - end_node.bbox.y1) / 2
                diameter_mm = diameter_pixels * self.calibration.pixel_to_mm_ratio if self.calibration else diameter_pixels
                
                # 评估质量
                quality, quality_score, defects = self._assess_segment_quality_yolo(
                    start_node, end_node, image, diameter_mm
                )
                
                # 创建竹筒段
                segment = BambooSegment(
                    start_pos=start_node.position,
                    end_pos=end_node.position,
                    start_pos_mm=start_node.position_mm,
                    end_pos_mm=end_node.position_mm,
                    length_mm=length_mm,
                    diameter_mm=diameter_mm,
                    quality=quality,
                    quality_score=quality_score,
                    defects=defects,
                    bbox=BoundingBox(
                        min(start_node.bbox.x1, end_node.bbox.x1),
                        min(start_node.bbox.y1, end_node.bbox.y1),
                        max(start_node.bbox.x2, end_node.bbox.x2),
                        max(start_node.bbox.y2, end_node.bbox.y2)
                    )
                )
                
                segments.append(segment)
                
                logger.debug(f"创建竹筒段: 长度={length_mm:.1f}mm, 直径={diameter_mm:.1f}mm, 质量={quality.value}")
        
        except Exception as e:
            logger.error(f"竹筒段分割失败: {e}")
        
        return segments
    
    def _assess_segment_quality_yolo(self, start_node: BambooNode, end_node: BambooNode, 
                                   image: np.ndarray, diameter_mm: float) -> Tuple[SegmentQuality, float, List[str]]:
        """
        基于YOLO检测结果评估竹筒段质量
        """
        defects = []
        quality_score = 100.0
        
        try:
            # 基于节点置信度评估
            avg_confidence = (start_node.confidence + end_node.confidence) / 2
            if avg_confidence < 0.7:
                defects.append("节点置信度低")
                quality_score -= 20
            
            # 基于节点类型评估
            if start_node.node_type == NodeType.DAMAGE_NODE or end_node.node_type == NodeType.DAMAGE_NODE:
                defects.append("包含损坏节点")
                quality_score -= 30
            
            # 基于长度评估
            length_mm = abs(end_node.position_mm.x - start_node.position_mm.x)
            if length_mm < self.config.segment_min_length:
                defects.append("长度过短")
                quality_score -= 40
            elif length_mm > self.config.segment_max_length:
                defects.append("长度过长")
                quality_score -= 20
            
            # 基于直径评估
            if diameter_mm < 10:  # 最小直径阈值
                defects.append("直径过小")
                quality_score -= 25
            
            # 确定质量等级
            if quality_score >= 80:
                quality = SegmentQuality.HIGH
            elif quality_score >= 60:
                quality = SegmentQuality.MEDIUM
            elif quality_score >= 40:
                quality = SegmentQuality.LOW
            else:
                quality = SegmentQuality.UNUSABLE
        
        except Exception as e:
            logger.error(f"质量评估失败: {e}")
            quality = SegmentQuality.UNKNOWN
            quality_score = 0.0
            defects = ["质量评估失败"]
        
        return quality, max(0.0, quality_score), defects
    
    def _calculate_cutting_points_yolo(self, nodes: List[BambooNode], 
                                     segments: List[BambooSegment]) -> List[CuttingPoint]:
        """
        基于YOLO检测结果计算切割点
        """
        cutting_points = []
        
        try:
            # 为每个节点创建移除切割点
            for node in nodes:
                cutting_point = CuttingPoint(
                    position=Point2D(node.position.x, node.position.y),
                    position_mm=Point2D(node.position_mm.x, node.position_mm.y),
                    cutting_type=CuttingType.REMOVE_NODE,
                    priority=8,  # 高优先级
                    confidence=node.confidence,
                    reason=f"移除{node.node_type.value}节点"
                )
                cutting_points.append(cutting_point)
            
            # 为长竹筒段创建长度分割切割点
            for segment in segments:
                if segment.length_mm > 150:  # 长度阈值
                    # 在段中间创建切割点
                    mid_x = (segment.start_pos.x + segment.end_pos.x) / 2
                    mid_y = (segment.start_pos.y + segment.end_pos.y) / 2
                    mid_x_mm = (segment.start_pos_mm.x + segment.end_pos_mm.x) / 2
                    mid_y_mm = (segment.start_pos_mm.y + segment.end_pos_mm.y) / 2
                    
                    cutting_point = CuttingPoint(
                        position=Point2D(mid_x, mid_y),
                        position_mm=Point2D(mid_x_mm, mid_y_mm),
                        cutting_type=CuttingType.LENGTH_DIVISION,
                        priority=6,
                        confidence=segment.quality_score / 100.0,
                        reason="长度分割"
                    )
                    cutting_points.append(cutting_point)
        
        except Exception as e:
            logger.error(f"切割点计算失败: {e}")
        
        return cutting_points
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        """
        if not self.model:
            return {"status": "未初始化"}
        
        info = {
            "model_path": self.model_path,
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "class_mapping": {k: v.value for k, v in self.class_mapping.items()},
            "avg_inference_time": np.mean(self.inference_times) if self.inference_times else 0,
            "avg_preprocess_time": np.mean(self.preprocessing_times) if self.preprocessing_times else 0
        }
        
        return info
    
    def set_detection_parameters(self, confidence: float = None, iou: float = None, max_det: int = None):
        """
        设置检测参数
        """
        if confidence is not None:
            self.confidence_threshold = confidence
        if iou is not None:
            self.iou_threshold = iou
        if max_det is not None:
            self.max_detections = max_det
        
        logger.info(f"检测参数已更新: conf={self.confidence_threshold}, "
                   f"iou={self.iou_threshold}, max_det={self.max_detections}")
    
    def export_model(self, format: str = 'onnx', optimize: bool = True) -> str:
        """
        导出模型为其他格式（用于部署优化）
        """
        if not self.model:
            raise RuntimeError("模型未初始化")
        
        try:
            export_path = f"models/bamboo_model.{format}"
            
            if format == 'onnx':
                self.model.export(format='onnx', optimize=optimize)
            elif format == 'openvino':
                self.model.export(format='openvino', optimize=optimize)
            elif format == 'tensorrt':
                self.model.export(format='engine', optimize=optimize)
            else:
                raise ValueError(f"不支持的导出格式: {format}")
            
            logger.info(f"模型已导出为 {format} 格式: {export_path}")
            return export_path
            
        except Exception as e:
            logger.error(f"模型导出失败: {e}")
            raise
    
    def cleanup(self):
        """清理资源"""
        try:
            if self.model:
                # 清理GPU内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                self.model = None
            
            self.is_initialized = False
            logger.info("YOLO检测器资源清理完成")
            
        except Exception as e:
            logger.error(f"资源清理失败: {e}")


def register_yolo_detector():
    """注册YOLO检测器到工厂"""
    from .vision_processor import VisionProcessorFactory
    VisionProcessorFactory.register('yolo', YOLODetector)
    logger.info("YOLO检测器已注册")


# 自动注册
register_yolo_detector() 