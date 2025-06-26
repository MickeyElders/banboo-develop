"""
YOLOv8竹节检测器 - 优化版本
结合深度学习模型进行竹节检测，支持多种性能优化技术
"""

import os
import logging
import time
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Optional, Dict, Tuple, Any, Union
from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.autobatch import autobatch

from .vision_processor import VisionProcessor, monitor_performance
from .vision_types import (
    Point2D, BoundingBox, BambooNode, BambooSegment, CuttingPoint,
    DetectionResult, ProcessingROI, AlgorithmConfig, CalibrationData,
    NodeType, SegmentQuality, CuttingType
)

logger = logging.getLogger(__name__)


class ModelFormat(Enum):
    """支持的模型格式"""
    PYTORCH = "pt"
    ONNX = "onnx"
    TENSORRT = "engine"
    OPENVINO = "openvino"
    NCNN = "ncnn"
    TFLITE = "tflite"
    COREML = "coreml"


class InferenceMode(Enum):
    """推理模式"""
    SYNC = "sync"           # 同步推理
    ASYNC = "async"         # 异步推理
    BATCH = "batch"         # 批处理推理
    STREAM = "stream"       # 流式推理


@dataclass
class OptimizationConfig:
    """性能优化配置"""
    enable_half_precision: bool = True      # 启用FP16精度
    enable_tensorrt: bool = True            # 启用TensorRT加速
    enable_openvino: bool = False           # 启用OpenVINO优化
    enable_onnx: bool = False              # 启用ONNX优化
    
    # 批处理配置
    auto_batch_size: bool = True           # 自动确定批量大小
    max_batch_size: int = 16               # 最大批量大小
    
    # 线程配置
    max_workers: int = 4                   # 最大工作线程数
    enable_threading: bool = True          # 启用多线程
    
    # 内存优化
    enable_memory_pool: bool = True        # 启用内存池
    max_memory_cache: int = 1024           # 最大内存缓存(MB)
    
    # 预热配置
    warmup_iterations: int = 5             # 预热迭代次数
    enable_model_cache: bool = True        # 启用模型缓存


class OptimizedYOLODetector(VisionProcessor):
    """
    优化版本的YOLOv8竹节检测器
    支持多种性能优化技术
    """
    
    def __init__(self, config: AlgorithmConfig, calibration: Optional[CalibrationData] = None):
        super().__init__(config, calibration)
        
        # 优化配置
        self.opt_config = OptimizationConfig()
        
        # 模型相关
        self.model: Optional[YOLO] = None
        self.model_path = "models/bamboo_yolo.pt"
        self.model_format = ModelFormat.PYTORCH
        self.device = self._get_optimal_device()
        
        # 检测参数
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.45
        self.max_detections = 50
        
        # 批处理相关
        self.optimal_batch_size = 1
        self.inference_mode = InferenceMode.SYNC
        
        # 线程安全
        self._model_lock = threading.RLock()
        self._inference_lock = threading.Lock()
        self.thread_local = threading.local()
        
        # 性能统计
        self.performance_stats = {
            'total_inferences': 0,
            'total_time': 0.0,
            'preprocessing_time': 0.0,
            'inference_time': 0.0,
            'postprocessing_time': 0.0,
            'memory_usage': 0.0,
            'batch_sizes': [],
            'throughput_fps': 0.0
        }
        
        # 内存池
        self.memory_pool = {}
        
        # 类别映射
        self.class_mapping = {
            0: NodeType.NATURAL_NODE,
            1: NodeType.ARTIFICIAL_NODE,
            2: NodeType.BRANCH_NODE,
            3: NodeType.DAMAGE_NODE
        }
        
        logger.info("优化版YOLOv8检测器初始化完成")
    
    def _get_optimal_device(self) -> str:
        """获取最优设备"""
        if torch.cuda.is_available():
            # 检查GPU内存
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if gpu_memory >= 4.0:  # 4GB以上使用GPU
                return 'cuda'
            else:
                logger.warning(f"GPU内存不足({gpu_memory:.1f}GB)，使用CPU")
                return 'cpu'
        else:
            return 'cpu'
    
    def set_optimization_config(self, **kwargs):
        """设置优化配置"""
        for key, value in kwargs.items():
            if hasattr(self.opt_config, key):
                setattr(self.opt_config, key, value)
                logger.info(f"优化配置更新: {key} = {value}")
    
    def initialize(self) -> bool:
        """
        初始化优化版YOLO检测器
        """
        try:
            logger.info("初始化优化版YOLO检测器...")
            
            # 检查并加载模型
            if not self._load_optimal_model():
                return False
            
            # 自动优化批量大小
            if self.opt_config.auto_batch_size:
                self._optimize_batch_size()
            
            # 模型预热
            self._warmup_model()
            
            # 初始化内存池
            if self.opt_config.enable_memory_pool:
                self._initialize_memory_pool()
            
            self.is_initialized = True
            logger.info("优化版YOLO检测器初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"优化版YOLO检测器初始化失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _load_optimal_model(self) -> bool:
        """加载最优模型格式"""
        try:
            # 检查可用的模型格式，按性能优先级排序
            model_candidates = self._find_model_candidates()
            
            for model_path, format_type in model_candidates:
                try:
                    logger.info(f"尝试加载模型: {model_path} (格式: {format_type.value})")
                    
                    # 加载模型
                    self.model = YOLO(model_path)
                    self.model_path = str(model_path)
                    self.model_format = format_type
                    
                    # 设置设备
                    if hasattr(self.model.model, 'to'):
                        self.model.model.to(self.device)
                    
                    logger.info(f"成功加载模型: {model_path}")
                    return True
                    
                except Exception as e:
                    logger.warning(f"加载模型失败 {model_path}: {e}")
                    continue
            
            # 如果没有找到模型，尝试下载预训练模型
            return self._initialize_pretrained_model()
            
        except Exception as e:
            logger.error(f"模型加载过程失败: {e}")
            return False
    
    def _find_model_candidates(self) -> List[Tuple[Path, ModelFormat]]:
        """查找可用的模型文件，按性能优先级排序"""
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        
        # 按性能优先级排序
        priority_formats = [
            (ModelFormat.TENSORRT, ["*.engine"]),
            (ModelFormat.ONNX, ["*.onnx"]),
            (ModelFormat.OPENVINO, ["*_openvino_model"]),
            (ModelFormat.PYTORCH, ["*.pt"]),
            (ModelFormat.NCNN, ["*_ncnn_model"]),
            (ModelFormat.TFLITE, ["*.tflite"]),
        ]
        
        candidates = []
        
        for format_type, patterns in priority_formats:
            for pattern in patterns:
                for model_path in model_dir.glob(pattern):
                    if model_path.exists():
                        candidates.append((model_path, format_type))
        
        # 检查训练输出目录
        train_output = Path("runs/detect/train/weights")
        if train_output.exists():
            for model_file in ["best.pt", "last.pt"]:
                model_path = train_output / model_file
                if model_path.exists():
                    candidates.append((model_path, ModelFormat.PYTORCH))
        
        return candidates
    
    def _initialize_pretrained_model(self) -> bool:
        """初始化预训练模型"""
        try:
            logger.info("初始化YOLOv8预训练模型")
            
            # 选择合适的预训练模型
            if self.device == 'cuda':
                model_name = 'yolov8s.pt'  # GPU使用稍大模型
            else:
                model_name = 'yolov8n.pt'  # CPU使用轻量模型
            
            self.model = YOLO(model_name)
            self.model_path = model_name
            self.model_format = ModelFormat.PYTORCH
            
            # 降低阈值（因为不是专门训练的）
            self.confidence_threshold = 0.3
            
            logger.info(f"预训练模型初始化成功: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"预训练模型初始化失败: {e}")
            return False
    
    def _optimize_batch_size(self):
        """自动优化批量大小"""
        try:
            if self.device == 'cuda' and hasattr(self.model, 'model'):
                logger.info("自动优化批量大小...")
                
                # 使用ultralytics的autobatch功能
                optimal_batch = autobatch(self.model, imgsz=640, fraction=0.9)
                self.optimal_batch_size = min(optimal_batch, self.opt_config.max_batch_size)
                
                logger.info(f"最优批量大小: {self.optimal_batch_size}")
            else:
                self.optimal_batch_size = 1
                logger.info("使用默认批量大小: 1")
                
        except Exception as e:
            logger.warning(f"批量大小优化失败: {e}")
            self.optimal_batch_size = 1
    
    def _warmup_model(self):
        """增强版模型预热"""
        try:
            logger.info("模型预热中...")
            
            # 创建不同尺寸的测试图像
            test_sizes = [(640, 640), (320, 320), (1280, 1280)]
            
            for i, (h, w) in enumerate(test_sizes):
                for _ in range(self.opt_config.warmup_iterations):
                    dummy_image = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
                    
                    # 使用不同批量大小预热
                    if i == 0 and self.optimal_batch_size > 1:
                        dummy_batch = np.stack([dummy_image] * min(2, self.optimal_batch_size))
                        _ = self.model(dummy_batch, verbose=False)
                    else:
                        _ = self.model(dummy_image, verbose=False)
            
            logger.info("模型预热完成")
            
        except Exception as e:
            logger.warning(f"模型预热失败: {e}")
    
    def _initialize_memory_pool(self):
        """初始化内存池"""
        try:
            # 预分配常用尺寸的内存
            common_sizes = [(640, 640), (320, 320), (1280, 1280)]
            
            for size in common_sizes:
                key = f"image_{size[0]}x{size[1]}"
                self.memory_pool[key] = np.zeros((size[0], size[1], 3), dtype=np.uint8)
            
            logger.info("内存池初始化完成")
            
        except Exception as e:
            logger.warning(f"内存池初始化失败: {e}")
    
    def get_thread_local_model(self) -> YOLO:
        """获取线程本地模型实例"""
        if not hasattr(self.thread_local, 'model'):
            with self._model_lock:
                # 为每个线程创建独立的模型实例
                self.thread_local.model = YOLO(self.model_path)
                if hasattr(self.thread_local.model.model, 'to'):
                    self.thread_local.model.model.to(self.device)
                logger.debug(f"为线程创建独立模型实例: {threading.current_thread().name}")
        
        return self.thread_local.model
    
    @monitor_performance
    def process_image(self, image: np.ndarray, roi: Optional[ProcessingROI] = None) -> DetectionResult:
        """
        使用优化后的YOLO模型处理图像
        """
        try:
            start_time = time.time()
            
            # 选择推理模式
            if self.inference_mode == InferenceMode.ASYNC:
                return asyncio.run(self._async_inference(image, roi))
            elif self.inference_mode == InferenceMode.BATCH:
                return self._batch_inference([image], [roi])[0]
            else:
                return self._sync_inference(image, roi)
                
        except Exception as e:
            logger.error(f"图像处理失败: {e}")
            return self._create_empty_result(image.shape[:2], str(e))
    
    def _sync_inference(self, image: np.ndarray, roi: Optional[ProcessingROI] = None) -> DetectionResult:
        """同步推理"""
        start_time = time.time()
        
        # 预处理
        preprocess_start = time.time()
        processed_image = self.preprocess_image(image)
        preprocess_time = time.time() - preprocess_start
        
        # 推理
        inference_start = time.time()
        if self.opt_config.enable_threading:
            model = self.get_thread_local_model()
        else:
            model = self.model
        
        nodes = self._optimized_inference(processed_image, model)
        inference_time = time.time() - inference_start
        
        # 后处理
        postprocess_start = time.time()
        segments = []
        if len(nodes) > 1:
            segments = self._segment_bamboo_tubes_yolo(processed_image, nodes)
        
        cutting_points = self._calculate_cutting_points_yolo(nodes, segments)
        postprocess_time = time.time() - postprocess_start
        
        # 更新性能统计
        total_time = time.time() - start_time
        self._update_performance_stats(total_time, preprocess_time, inference_time, postprocess_time, 1)
        
        return DetectionResult(
            nodes=nodes,
            segments=segments,
            cutting_points=cutting_points,
            processing_time=total_time,
            image_shape=image.shape[:2],
            metadata={
                'algorithm': 'yolov8_optimized',
                'model_format': self.model_format.value,
                'device': self.device,
                'batch_size': 1,
                'inference_mode': self.inference_mode.value,
                'performance': {
                    'preprocess_time': preprocess_time,
                    'inference_time': inference_time,
                    'postprocess_time': postprocess_time,
                    'total_time': total_time
                }
            }
        )
    
    async def _async_inference(self, image: np.ndarray, roi: Optional[ProcessingROI] = None) -> DetectionResult:
        """异步推理"""
        loop = asyncio.get_event_loop()
        
        # 在线程池中执行推理
        with ThreadPoolExecutor(max_workers=self.opt_config.max_workers) as executor:
            result = await loop.run_in_executor(executor, self._sync_inference, image, roi)
        
        return result
    
    def process_images_batch(self, images: List[np.ndarray], 
                           rois: Optional[List[ProcessingROI]] = None) -> List[DetectionResult]:
        """批量处理图像"""
        if rois is None:
            rois = [None] * len(images)
        
        return self._batch_inference(images, rois)
    
    def _batch_inference(self, images: List[np.ndarray], 
                        rois: List[Optional[ProcessingROI]]) -> List[DetectionResult]:
        """批量推理"""
        if len(images) == 0:
            return []
        
        start_time = time.time()
        batch_size = min(len(images), self.optimal_batch_size)
        results = []
        
        # 分批处理
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_rois = rois[i:i + batch_size]
            
            # 预处理批次
            preprocess_start = time.time()
            processed_batch = [self.preprocess_image(img) for img in batch_images]
            preprocess_time = time.time() - preprocess_start
            
            # 批量推理
            inference_start = time.time()
            batch_nodes = self._batch_yolo_inference(processed_batch)
            inference_time = time.time() - inference_start
            
            # 后处理
            postprocess_start = time.time()
            for j, (img, nodes) in enumerate(zip(processed_batch, batch_nodes)):
                segments = []
                if len(nodes) > 1:
                    segments = self._segment_bamboo_tubes_yolo(img, nodes)
                
                cutting_points = self._calculate_cutting_points_yolo(nodes, segments)
                
                result = DetectionResult(
                    nodes=nodes,
                    segments=segments,
                    cutting_points=cutting_points,
                    processing_time=0.0,  # 单张图像时间在批处理中难以准确计算
                    image_shape=batch_images[j].shape[:2],
                    metadata={
                        'algorithm': 'yolov8_optimized',
                        'model_format': self.model_format.value,
                        'device': self.device,
                        'batch_size': len(batch_images),
                        'inference_mode': InferenceMode.BATCH.value
                    }
                )
                results.append(result)
            
            postprocess_time = time.time() - postprocess_start
            
            # 更新性能统计
            batch_time = time.time() - start_time
            self._update_performance_stats(batch_time, preprocess_time, inference_time, postprocess_time, len(batch_images))
        
        return results
    
    def _optimized_inference(self, image: np.ndarray, model: YOLO) -> List[BambooNode]:
        """优化的推理过程"""
        try:
            # 配置推理参数
            inference_kwargs = {
                'conf': self.confidence_threshold,
                'iou': self.iou_threshold,
                'max_det': self.max_detections,
                'verbose': False,
                'device': self.device
            }
            
            # 启用FP16精度
            if self.opt_config.enable_half_precision and self.device == 'cuda':
                inference_kwargs['half'] = True
            
            # 执行推理
            with self._inference_lock:
                results = model(image, **inference_kwargs)
            
            # 解析结果
            nodes = self._parse_yolo_results(results[0], image.shape)
            
            return nodes
            
        except Exception as e:
            logger.error(f"优化推理失败: {e}")
            return []
    
    def _batch_yolo_inference(self, images: List[np.ndarray]) -> List[List[BambooNode]]:
        """批量YOLO推理"""
        try:
            # 确保所有图像尺寸一致
            if len(set(img.shape for img in images)) > 1:
                # 调整到统一尺寸
                target_size = images[0].shape
                images = [cv2.resize(img, (target_size[1], target_size[0])) for img in images]
            
            # 批量推理
            inference_kwargs = {
                'conf': self.confidence_threshold,
                'iou': self.iou_threshold,
                'max_det': self.max_detections,
                'verbose': False,
                'device': self.device
            }
            
            if self.opt_config.enable_half_precision and self.device == 'cuda':
                inference_kwargs['half'] = True
            
            # 执行批量推理
            with self._inference_lock:
                batch_results = self.model(images, **inference_kwargs)
            
            # 解析所有结果
            all_nodes = []
            for i, result in enumerate(batch_results):
                nodes = self._parse_yolo_results(result, images[i].shape)
                all_nodes.append(nodes)
            
            return all_nodes
            
        except Exception as e:
            logger.error(f"批量推理失败: {e}")
            return [[] for _ in images]
    
    def _parse_yolo_results(self, result, image_shape: Tuple[int, int, int]) -> List[BambooNode]:
        """解析YOLO推理结果"""
        nodes = []
        
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            
            for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
                x1, y1, x2, y2 = box
                
                # 计算中心点
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # 转换类别
                node_type = self.class_mapping.get(int(cls), NodeType.NATURAL_NODE)
                
                # 创建竹节对象
                node = BambooNode(
                    id=f"yolo_{i}",
                    center=Point2D(center_x, center_y),
                    bounding_box=BoundingBox(x1, y1, x2 - x1, y2 - y1),
                    confidence=float(score),
                    node_type=node_type,
                    diameter_px=max(x2 - x1, y2 - y1),
                    metadata={
                        'detection_method': 'yolov8_optimized',
                        'class_id': int(cls),
                        'bbox_area': (x2 - x1) * (y2 - y1)
                    }
                )
                nodes.append(node)
        
        # 按Y坐标排序
        nodes.sort(key=lambda n: n.center.y)
        return nodes
    
    def _update_performance_stats(self, total_time: float, preprocess_time: float, 
                                 inference_time: float, postprocess_time: float, batch_size: int):
        """更新性能统计"""
        self.performance_stats['total_inferences'] += batch_size
        self.performance_stats['total_time'] += total_time
        self.performance_stats['preprocessing_time'] += preprocess_time
        self.performance_stats['inference_time'] += inference_time
        self.performance_stats['postprocessing_time'] += postprocess_time
        self.performance_stats['batch_sizes'].append(batch_size)
        
        # 计算吞吐量
        if self.performance_stats['total_time'] > 0:
            self.performance_stats['throughput_fps'] = (
                self.performance_stats['total_inferences'] / self.performance_stats['total_time']
            )
    
    def _create_empty_result(self, image_shape: Tuple[int, int], error_msg: str = "") -> DetectionResult:
        """创建空结果"""
        return DetectionResult(
            nodes=[],
            segments=[],
            cutting_points=[],
            processing_time=0.0,
            image_shape=image_shape,
            metadata={'error': error_msg, 'algorithm': 'yolov8_optimized'}
        )
    
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
    
    def export_optimized_model(self, format: str = 'onnx', **kwargs) -> str:
        """
        导出优化模型
        Args:
            format: 导出格式 ('onnx', 'engine', 'openvino', 'ncnn', 'tflite')
            **kwargs: 导出参数
        Returns:
            导出文件路径
        """
        try:
            if not self.model:
                raise ValueError("模型未初始化")
            
            logger.info(f"导出模型为{format}格式...")
            
            # 默认导出参数
            export_kwargs = {
                'imgsz': 640,
                'optimize': True,
                'half': self.opt_config.enable_half_precision and self.device == 'cuda',
                'simplify': True,
                'workspace': 4,  # GB
                'batch': self.optimal_batch_size if self.optimal_batch_size > 1 else 1
            }
            
            # 更新用户参数
            export_kwargs.update(kwargs)
            
            # 执行导出
            exported_path = self.model.export(format=format, **export_kwargs)
            
            logger.info(f"模型导出成功: {exported_path}")
            return str(exported_path)
            
        except Exception as e:
            logger.error(f"模型导出失败: {e}")
            raise
    
    def benchmark_performance(self, test_images: List[np.ndarray], 
                            num_runs: int = 10) -> Dict[str, Any]:
        """
        性能基准测试
        Args:
            test_images: 测试图像列表
            num_runs: 运行次数
        Returns:
            性能基准结果
        """
        try:
            logger.info(f"开始性能基准测试，{num_runs}次运行...")
            
            # 重置性能统计
            self.performance_stats = {
                'total_inferences': 0,
                'total_time': 0.0,
                'preprocessing_time': 0.0,
                'inference_time': 0.0,
                'postprocessing_time': 0.0,
                'memory_usage': 0.0,
                'batch_sizes': [],
                'throughput_fps': 0.0
            }
            
            # 单张图像测试
            single_times = []
            for _ in range(num_runs):
                start_time = time.time()
                for img in test_images:
                    self.process_image(img)
                single_times.append(time.time() - start_time)
            
            # 批量处理测试
            batch_times = []
            if len(test_images) > 1 and self.optimal_batch_size > 1:
                for _ in range(num_runs):
                    start_time = time.time()
                    self.process_images_batch(test_images)
                    batch_times.append(time.time() - start_time)
            
            # 计算统计结果
            benchmark_results = {
                'model_info': {
                    'path': self.model_path,
                    'format': self.model_format.value,
                    'device': self.device,
                    'optimal_batch_size': self.optimal_batch_size
                },
                'single_image': {
                    'avg_time': np.mean(single_times),
                    'min_time': np.min(single_times),
                    'max_time': np.max(single_times),
                    'std_time': np.std(single_times),
                    'throughput_fps': len(test_images) / np.mean(single_times)
                },
                'performance_breakdown': {
                    'preprocessing_pct': (self.performance_stats['preprocessing_time'] / 
                                        self.performance_stats['total_time'] * 100),
                    'inference_pct': (self.performance_stats['inference_time'] / 
                                    self.performance_stats['total_time'] * 100),
                    'postprocessing_pct': (self.performance_stats['postprocessing_time'] / 
                                         self.performance_stats['total_time'] * 100)
                }
            }
            
            if batch_times:
                benchmark_results['batch_processing'] = {
                    'avg_time': np.mean(batch_times),
                    'min_time': np.min(batch_times),
                    'max_time': np.max(batch_times),
                    'throughput_fps': len(test_images) / np.mean(batch_times),
                    'speedup_factor': np.mean(single_times) / np.mean(batch_times)
                }
            
            logger.info("性能基准测试完成")
            return benchmark_results
            
        except Exception as e:
            logger.error(f"性能基准测试失败: {e}")
            return {}
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        stats = self.performance_stats.copy()
        
        # 添加平均值
        if stats['total_inferences'] > 0:
            stats['avg_total_time'] = stats['total_time'] / stats['total_inferences']
            stats['avg_preprocessing_time'] = stats['preprocessing_time'] / stats['total_inferences']
            stats['avg_inference_time'] = stats['inference_time'] / stats['total_inferences']
            stats['avg_postprocessing_time'] = stats['postprocessing_time'] / stats['total_inferences']
        
        # 添加配置信息
        stats['optimization_config'] = {
            'model_format': self.model_format.value,
            'device': self.device,
            'optimal_batch_size': self.optimal_batch_size,
            'half_precision': self.opt_config.enable_half_precision,
            'threading_enabled': self.opt_config.enable_threading
        }
        
        return stats
    
    def set_inference_mode(self, mode: InferenceMode):
        """设置推理模式"""
        self.inference_mode = mode
        logger.info(f"推理模式已设置为: {mode.value}")
    
    @contextmanager
    def inference_context(self, **kwargs):
        """推理上下文管理器"""
        old_config = {}
        
        # 保存旧配置
        for key, value in kwargs.items():
            if hasattr(self, key):
                old_config[key] = getattr(self, key)
                setattr(self, key, value)
        
        try:
            yield self
        finally:
            # 恢复旧配置
            for key, value in old_config.items():
                setattr(self, key, value)

    def cleanup(self):
        """清理资源"""
        try:
            if self.model:
                del self.model
                self.model = None
            
            # 清理线程本地存储
            if hasattr(self.thread_local, 'model'):
                del self.thread_local.model
            
            # 清理内存池
            self.memory_pool.clear()
            
            # 清理CUDA缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("优化版YOLO检测器资源清理完成")
            
        except Exception as e:
            logger.error(f"资源清理失败: {e}")


# 为了向后兼容，保留原始类名
YOLODetector = OptimizedYOLODetector


def register_optimized_yolo_detector():
    """注册优化版YOLO检测器到工厂"""
    from .vision_processor import VisionProcessorFactory
    
    VisionProcessorFactory.register_processor("yolo_optimized", OptimizedYOLODetector)
    VisionProcessorFactory.register_processor("yolo", OptimizedYOLODetector)  # 默认使用优化版本
    logger.info("优化版YOLO检测器已注册到工厂")


# 自动注册
register_optimized_yolo_detector() 