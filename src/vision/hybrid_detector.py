"""
混合竹节检测器 - 优化版本
结合传统计算机视觉算法和优化版YOLOv8深度学习模型
"""

import logging
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum

import cv2
import numpy as np

from .vision_processor import VisionProcessor, monitor_performance
from .traditional_detector import TraditionalDetector
from .yolo_detector import OptimizedYOLODetector, InferenceMode, ModelFormat
from .vision_types import (
    Point2D, BoundingBox, BambooNode, BambooSegment, CuttingPoint,
    DetectionResult, ProcessingROI, AlgorithmConfig, CalibrationData,
    NodeType, SegmentQuality, CuttingType
)

logger = logging.getLogger(__name__)


class HybridStrategy(Enum):
    """混合检测策略"""
    YOLO_FIRST = "yolo_first"           # YOLO优先，传统算法备用
    TRADITIONAL_FIRST = "traditional_first"  # 传统算法优先，YOLO备用
    PARALLEL_FUSION = "parallel_fusion"      # 并行运行，融合结果
    ADAPTIVE = "adaptive"               # 自适应选择
    CONSENSUS = "consensus"             # 共识验证
    PERFORMANCE_OPTIMIZED = "performance_optimized"  # 性能优化模式


class OptimizedHybridDetector(VisionProcessor):
    """
    优化版混合竹节检测器
    结合传统CV算法和优化版YOLO深度学习模型
    """
    
    def __init__(self, config: AlgorithmConfig, calibration: Optional[CalibrationData] = None):
        super().__init__(config, calibration)
        
        # 子检测器
        self.traditional_detector: Optional[TraditionalDetector] = None
        self.yolo_detector: Optional[OptimizedYOLODetector] = None
        
        # 混合策略配置
        self.strategy = HybridStrategy.PERFORMANCE_OPTIMIZED
        self.fusion_weights = {
            'yolo': 0.7,        # YOLO权重
            'traditional': 0.3   # 传统算法权重
        }
        
        # 自适应阈值
        self.adaptive_thresholds = {
            'min_yolo_detections': 2,     # YOLO最小检测数
            'min_traditional_detections': 3,  # 传统算法最小检测数
            'consensus_threshold': 0.6,    # 共识阈值
            'image_quality_threshold': 0.7,  # 图像质量阈值
            'performance_threshold': 0.1,  # 性能阈值(秒)
        }
        
        # 性能优化配置
        self.enable_batch_processing = True
        self.enable_async_processing = False
        self.max_batch_size = 8
        self.performance_mode = True
        
        # 性能统计
        self.method_performance = {
            'yolo': {'count': 0, 'total_time': 0, 'avg_detections': 0, 'success_rate': 0},
            'traditional': {'count': 0, 'total_time': 0, 'avg_detections': 0, 'success_rate': 0},
            'hybrid': {'count': 0, 'total_time': 0, 'avg_detections': 0, 'success_rate': 0}
        }
        
        # 缓存机制
        self.result_cache = {}
        self.cache_enabled = True
        self.max_cache_size = 100
        
        logger.info("优化版混合检测器初始化完成")
    
    def initialize(self) -> bool:
        """
        初始化优化版混合检测器
        """
        try:
            logger.info("初始化优化版混合检测器...")
            
            # 初始化传统检测器
            self.traditional_detector = TraditionalDetector(self.config, self.calibration)
            traditional_ok = self.traditional_detector.initialize()
            
            # 初始化优化版YOLO检测器
            self.yolo_detector = OptimizedYOLODetector(self.config, self.calibration)
            yolo_ok = self.yolo_detector.initialize()
            
            # 检查初始化结果
            if not traditional_ok and not yolo_ok:
                logger.error("所有检测器初始化失败")
                return False
            elif not yolo_ok:
                logger.warning("YOLO检测器初始化失败，仅使用传统算法")
                self.strategy = HybridStrategy.TRADITIONAL_FIRST
            elif not traditional_ok:
                logger.warning("传统检测器初始化失败，仅使用YOLO")
                self.strategy = HybridStrategy.YOLO_FIRST
            
            # 配置YOLO检测器的性能优化
            if self.yolo_detector:
                self._configure_yolo_optimization()
            
            self.is_initialized = True
            logger.info(f"优化版混合检测器初始化成功，策略: {self.strategy.value}")
            return True
            
        except Exception as e:
            logger.error(f"优化版混合检测器初始化失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _configure_yolo_optimization(self):
        """配置YOLO检测器的优化设置"""
        if not self.yolo_detector:
            return
        
        try:
            # 设置优化配置
            optimization_config = {
                'enable_half_precision': True,
                'enable_threading': True,
                'auto_batch_size': True,
                'max_batch_size': self.max_batch_size,
                'enable_memory_pool': True,
                'warmup_iterations': 3
            }
            
            self.yolo_detector.set_optimization_config(**optimization_config)
            
            # 根据性能模式设置推理模式
            if self.performance_mode:
                if self.enable_async_processing:
                    self.yolo_detector.set_inference_mode(InferenceMode.ASYNC)
                elif self.enable_batch_processing:
                    self.yolo_detector.set_inference_mode(InferenceMode.BATCH)
                else:
                    self.yolo_detector.set_inference_mode(InferenceMode.SYNC)
            
            logger.info("YOLO检测器优化配置完成")
            
        except Exception as e:
            logger.warning(f"YOLO优化配置失败: {e}")
    
    def set_strategy(self, strategy: HybridStrategy, **kwargs):
        """
        设置混合检测策略
        Args:
            strategy: 检测策略
            **kwargs: 策略参数
        """
        self.strategy = strategy
        
        # 更新权重
        if 'yolo_weight' in kwargs:
            self.fusion_weights['yolo'] = kwargs['yolo_weight']
            self.fusion_weights['traditional'] = 1.0 - kwargs['yolo_weight']
        
        # 更新阈值
        if 'thresholds' in kwargs:
            self.adaptive_thresholds.update(kwargs['thresholds'])
        
        # 更新性能配置
        if 'enable_batch_processing' in kwargs:
            self.enable_batch_processing = kwargs['enable_batch_processing']
        
        if 'enable_async_processing' in kwargs:
            self.enable_async_processing = kwargs['enable_async_processing']
        
        if 'performance_mode' in kwargs:
            self.performance_mode = kwargs['performance_mode']
        
        # 重新配置YOLO优化
        if self.yolo_detector:
            self._configure_yolo_optimization()
        
        logger.info(f"混合检测策略已更新为: {strategy.value}")
    
    @monitor_performance
    def process_image(self, image: np.ndarray, roi: Optional[ProcessingROI] = None) -> DetectionResult:
        """
        使用优化版混合策略处理图像
        """
        try:
            logger.debug(f"开始优化版混合检测，策略: {self.strategy.value}")
            
            start_time = time.time()
            
            # 检查缓存
            if self.cache_enabled:
                cache_key = self._generate_cache_key(image, roi)
                if cache_key in self.result_cache:
                    logger.debug("使用缓存结果")
                    return self.result_cache[cache_key]
            
            # 根据策略选择检测方法
            if self.strategy == HybridStrategy.PERFORMANCE_OPTIMIZED:
                result = self._performance_optimized_strategy(image, roi)
            elif self.strategy == HybridStrategy.YOLO_FIRST:
                result = self._yolo_first_strategy(image, roi)
            elif self.strategy == HybridStrategy.TRADITIONAL_FIRST:
                result = self._traditional_first_strategy(image, roi)
            elif self.strategy == HybridStrategy.PARALLEL_FUSION:
                result = self._parallel_fusion_strategy(image, roi)
            elif self.strategy == HybridStrategy.ADAPTIVE:
                result = self._adaptive_strategy(image, roi)
            elif self.strategy == HybridStrategy.CONSENSUS:
                result = self._consensus_strategy(image, roi)
            else:
                raise ValueError(f"不支持的策略: {self.strategy}")
            
            # 更新性能统计
            processing_time = time.time() - start_time
            self._update_performance_stats(processing_time, len(result.nodes))
            
            # 添加混合策略元数据
            result.metadata.update({
                'hybrid_strategy': self.strategy.value,
                'fusion_weights': self.fusion_weights.copy(),
                'performance_stats': self.get_performance_stats(),
                'optimization_enabled': self.performance_mode
            })
            
            # 缓存结果
            if self.cache_enabled and cache_key:
                self._cache_result(cache_key, result)
            
            logger.info(f"优化版混合检测完成: {len(result.nodes)}个竹节, {len(result.segments)}个竹筒段, "
                       f"{len(result.cutting_points)}个切割点, 耗时{processing_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"优化版混合检测失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # 返回空结果
            return DetectionResult(
                nodes=[],
                segments=[],
                cutting_points=[],
                processing_time=0.0,
                image_shape=image.shape[:2],
                metadata={'error': str(e), 'algorithm': 'hybrid_optimized'}
            )
    
    def process_images_batch(self, images: List[np.ndarray], 
                           rois: Optional[List[ProcessingROI]] = None) -> List[DetectionResult]:
        """批量处理图像"""
        if not self.enable_batch_processing or not self.yolo_detector:
            # 逐个处理
            if rois is None:
                rois = [None] * len(images)
            return [self.process_image(img, roi) for img, roi in zip(images, rois)]
        
        try:
            logger.info(f"开始批量混合检测，{len(images)}张图像")
            start_time = time.time()
            
            if rois is None:
                rois = [None] * len(images)
            
            # 选择批量处理策略
            if self.strategy == HybridStrategy.PERFORMANCE_OPTIMIZED:
                results = self._batch_performance_optimized(images, rois)
            elif self.strategy == HybridStrategy.PARALLEL_FUSION:
                results = self._batch_parallel_fusion(images, rois)
            else:
                # 其他策略使用YOLO批量处理
                results = self._batch_yolo_processing(images, rois)
            
            processing_time = time.time() - start_time
            logger.info(f"批量混合检测完成，总耗时{processing_time:.3f}s, "
                       f"平均{processing_time/len(images):.3f}s/张")
            
            return results
            
        except Exception as e:
            logger.error(f"批量混合检测失败: {e}")
            # 降级到逐个处理
            return [self.process_image(img, roi) for img, roi in zip(images, rois)]
    
    async def process_image_async(self, image: np.ndarray, 
                                roi: Optional[ProcessingROI] = None) -> DetectionResult:
        """异步处理图像"""
        if not self.enable_async_processing or not self.yolo_detector:
            return self.process_image(image, roi)
        
        try:
            # 使用YOLO的异步推理
            self.yolo_detector.set_inference_mode(InferenceMode.ASYNC)
            result = await self.yolo_detector._async_inference(image, roi)
            
            # 如果需要，可以异步运行传统算法作为补充
            return result
            
        except Exception as e:
            logger.error(f"异步混合检测失败: {e}")
            # 降级到同步处理
            return self.process_image(image, roi)
    
    def _performance_optimized_strategy(self, image: np.ndarray, 
                                      roi: Optional[ProcessingROI] = None) -> DetectionResult:
        """性能优化策略 - 智能选择最快的检测方法"""
        try:
            # 评估图像复杂度
            image_complexity = self._assess_image_complexity(image)
            
            # 基于历史性能数据选择算法
            yolo_avg_time = self._get_avg_processing_time('yolo')
            traditional_avg_time = self._get_avg_processing_time('traditional')
            
            # 智能选择
            if self.yolo_detector and (yolo_avg_time < traditional_avg_time or image_complexity > 0.7):
                logger.debug("性能优化策略选择YOLO")
                result = self.yolo_detector.process_image(image, roi)
                
                # 如果YOLO检测结果不佳，快速回退到传统算法
                if (len(result.nodes) < self.adaptive_thresholds['min_yolo_detections'] and 
                    self.traditional_detector):
                    logger.debug("YOLO结果不佳，回退到传统算法")
                    traditional_result = self.traditional_detector.process_image(image, roi)
                    if len(traditional_result.nodes) > len(result.nodes):
                        result = traditional_result
                        result.metadata['fallback_used'] = True
                
                return result
            else:
                logger.debug("性能优化策略选择传统算法")
                return self._traditional_first_strategy(image, roi)
                
        except Exception as e:
            logger.error(f"性能优化策略失败: {e}")
            return self._yolo_first_strategy(image, roi)
    
    def _batch_performance_optimized(self, images: List[np.ndarray], 
                                   rois: List[Optional[ProcessingROI]]) -> List[DetectionResult]:
        """批量性能优化处理"""
        try:
            # 批量使用YOLO处理
            yolo_results = self.yolo_detector.process_images_batch(images, rois)
            
            # 识别需要传统算法补充的图像
            need_traditional = []
            final_results = []
            
            for i, result in enumerate(yolo_results):
                if len(result.nodes) < self.adaptive_thresholds['min_yolo_detections']:
                    need_traditional.append(i)
                final_results.append(result)
            
            # 对需要的图像使用传统算法
            if need_traditional and self.traditional_detector:
                for idx in need_traditional:
                    traditional_result = self.traditional_detector.process_image(images[idx], rois[idx])
                    if len(traditional_result.nodes) > len(final_results[idx].nodes):
                        final_results[idx] = traditional_result
                        final_results[idx].metadata['fallback_used'] = True
            
            return final_results
            
        except Exception as e:
            logger.error(f"批量性能优化处理失败: {e}")
            return [self._create_empty_result(img.shape[:2], str(e)) for img in images]
    
    def _batch_parallel_fusion(self, images: List[np.ndarray], 
                             rois: List[Optional[ProcessingROI]]) -> List[DetectionResult]:
        """批量并行融合处理"""
        try:
            # 并行执行YOLO和传统算法
            with ThreadPoolExecutor(max_workers=2) as executor:
                # 提交YOLO批量任务
                yolo_future = executor.submit(
                    self.yolo_detector.process_images_batch, images, rois
                ) if self.yolo_detector else None
                
                # 提交传统算法任务
                traditional_future = executor.submit(
                    self._batch_traditional_processing, images, rois
                ) if self.traditional_detector else None
            
            # 获取结果
            yolo_results = yolo_future.result() if yolo_future else [None] * len(images)
            traditional_results = traditional_future.result() if traditional_future else [None] * len(images)
            
            # 融合结果
            fused_results = []
            for i, (yolo_result, traditional_result) in enumerate(zip(yolo_results, traditional_results)):
                fused_result = self._fuse_results(yolo_result, traditional_result, images[i].shape[:2])
                fused_results.append(fused_result)
            
            return fused_results
            
        except Exception as e:
            logger.error(f"批量并行融合处理失败: {e}")
            return [self._create_empty_result(img.shape[:2], str(e)) for img in images]
    
    def _batch_traditional_processing(self, images: List[np.ndarray], 
                                    rois: List[Optional[ProcessingROI]]) -> List[DetectionResult]:
        """批量传统算法处理"""
        return [self.traditional_detector.process_image(img, roi) 
                for img, roi in zip(images, rois)]
    
    def _batch_yolo_processing(self, images: List[np.ndarray], 
                             rois: List[Optional[ProcessingROI]]) -> List[DetectionResult]:
        """批量YOLO处理"""
        if self.yolo_detector:
            return self.yolo_detector.process_images_batch(images, rois)
        else:
            return [self._create_empty_result(img.shape[:2], "YOLO不可用") for img in images]
    
    def _assess_image_complexity(self, image: np.ndarray) -> float:
        """评估图像复杂度"""
        try:
            # 计算图像的梯度幅值
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # 归一化复杂度分数
            complexity = np.mean(gradient_magnitude) / 255.0
            return min(complexity, 1.0)
            
        except Exception as e:
            logger.warning(f"图像复杂度评估失败: {e}")
            return 0.5  # 默认中等复杂度
    
    def _get_avg_processing_time(self, method: str) -> float:
        """获取算法的平均处理时间"""
        stats = self.method_performance.get(method, {})
        count = stats.get('count', 0)
        total_time = stats.get('total_time', 0)
        
        if count > 0:
            return total_time / count
        else:
            # 返回默认估计时间
            if method == 'yolo':
                return 0.1  # 100ms
            else:
                return 0.05  # 50ms
    
    def _update_performance_stats(self, processing_time: float, num_detections: int):
        """更新性能统计"""
        self.method_performance['hybrid']['count'] += 1
        self.method_performance['hybrid']['total_time'] += processing_time
        
        # 更新平均检测数
        old_avg = self.method_performance['hybrid']['avg_detections']
        count = self.method_performance['hybrid']['count']
        self.method_performance['hybrid']['avg_detections'] = (
            (old_avg * (count - 1) + num_detections) / count
        )
        
        # 更新成功率
        if num_detections > 0:
            success_count = self.method_performance['hybrid'].get('success_count', 0) + 1
            self.method_performance['hybrid']['success_count'] = success_count
            self.method_performance['hybrid']['success_rate'] = success_count / count
    
    def _generate_cache_key(self, image: np.ndarray, roi: Optional[ProcessingROI] = None) -> Optional[str]:
        """生成缓存键"""
        try:
            # 计算图像哈希
            image_hash = hash(image.tobytes())
            roi_hash = hash(str(roi)) if roi else 0
            return f"{image_hash}_{roi_hash}_{self.strategy.value}"
        except Exception:
            return None
    
    def _cache_result(self, cache_key: str, result: DetectionResult):
        """缓存结果"""
        if len(self.result_cache) >= self.max_cache_size:
            # 移除最旧的缓存项
            oldest_key = next(iter(self.result_cache))
            del self.result_cache[oldest_key]
        
        self.result_cache[cache_key] = result

    def _yolo_first_strategy(self, image: np.ndarray, roi: Optional[ProcessingROI] = None) -> DetectionResult:
        """YOLO优先策略"""
        try:
            if self.yolo_detector and self.yolo_detector.is_initialized:
                logger.debug("尝试YOLO检测")
                result = self.yolo_detector.process_image(image, roi)
                
                # 如果YOLO检测结果满意，直接返回
                if len(result.nodes) >= self.adaptive_thresholds['min_yolo_detections']:
                    logger.info(f"YOLO检测成功: {len(result.nodes)}个竹节")
                    return result
                
                logger.info("YOLO检测结果不满意，回退到传统算法")
            
            # 回退到传统算法
            if self.traditional_detector and self.traditional_detector.is_initialized:
                logger.debug("使用传统算法检测")
                return self.traditional_detector.process_image(image, roi)
            
            raise RuntimeError("所有检测器都不可用")
            
        except Exception as e:
            logger.error(f"YOLO优先策略失败: {e}")
            # 最后尝试传统算法
            if self.traditional_detector and self.traditional_detector.is_initialized:
                return self.traditional_detector.process_image(image, roi)
            raise
    
    def _traditional_first_strategy(self, image: np.ndarray, roi: Optional[ProcessingROI] = None) -> DetectionResult:
        """传统算法优先策略"""
        try:
            if self.traditional_detector and self.traditional_detector.is_initialized:
                logger.debug("尝试传统算法检测")
                result = self.traditional_detector.process_image(image, roi)
                
                # 如果传统算法检测结果满意，直接返回
                if len(result.nodes) >= self.adaptive_thresholds['min_traditional_detections']:
                    logger.info(f"传统算法检测成功: {len(result.nodes)}个竹节")
                    return result
                
                logger.info("传统算法检测结果不满意，尝试YOLO")
            
            # 回退到YOLO
            if self.yolo_detector and self.yolo_detector.is_initialized:
                logger.debug("使用YOLO检测")
                return self.yolo_detector.process_image(image, roi)
            
            raise RuntimeError("所有检测器都不可用")
            
        except Exception as e:
            logger.error(f"传统优先策略失败: {e}")
            # 最后尝试YOLO
            if self.yolo_detector and self.yolo_detector.is_initialized:
                return self.yolo_detector.process_image(image, roi)
            raise
    
    def _parallel_fusion_strategy(self, image: np.ndarray, roi: Optional[ProcessingROI] = None) -> DetectionResult:
        """并行融合策略"""
        try:
            yolo_result = None
            traditional_result = None
            
            # 并行运行两种检测器
            if self.yolo_detector and self.yolo_detector.is_initialized:
                logger.debug("运行YOLO检测")
                yolo_result = self.yolo_detector.process_image(image, roi)
            
            if self.traditional_detector and self.traditional_detector.is_initialized:
                logger.debug("运行传统算法检测")
                traditional_result = self.traditional_detector.process_image(image, roi)
            
            # 融合结果
            return self._fuse_results(yolo_result, traditional_result, image.shape[:2])
            
        except Exception as e:
            logger.error(f"并行融合策略失败: {e}")
            raise
    
    def _adaptive_strategy(self, image: np.ndarray, roi: Optional[ProcessingROI] = None) -> DetectionResult:
        """自适应策略"""
        try:
            # 评估图像质量
            image_quality = self._assess_image_quality(image)
            logger.debug(f"图像质量评分: {image_quality:.3f}")
            
            # 根据图像质量选择策略
            if image_quality > self.adaptive_thresholds['image_quality_threshold']:
                # 高质量图像，优先使用YOLO
                logger.debug("图像质量高，使用YOLO优先策略")
                return self._yolo_first_strategy(image, roi)
            else:
                # 低质量图像，使用传统算法或融合策略
                logger.debug("图像质量一般，使用传统算法优先策略")
                return self._traditional_first_strategy(image, roi)
            
        except Exception as e:
            logger.error(f"自适应策略失败: {e}")
            raise
    
    def _consensus_strategy(self, image: np.ndarray, roi: Optional[ProcessingROI] = None) -> DetectionResult:
        """共识验证策略"""
        try:
            # 运行两种检测器
            yolo_result = None
            traditional_result = None
            
            if self.yolo_detector and self.yolo_detector.is_initialized:
                yolo_result = self.yolo_detector.process_image(image, roi)
            
            if self.traditional_detector and self.traditional_detector.is_initialized:
                traditional_result = self.traditional_detector.process_image(image, roi)
            
            # 查找共识节点
            consensus_nodes = self._find_consensus_nodes(yolo_result, traditional_result)
            
            # 如果共识节点足够，使用共识结果
            if len(consensus_nodes) >= self.adaptive_thresholds['min_yolo_detections']:
                logger.info(f"找到{len(consensus_nodes)}个共识节点")
                
                # 基于共识节点重建结果
                return self._build_consensus_result(consensus_nodes, yolo_result, traditional_result, image.shape[:2])
            else:
                # 共识不足，使用较好的单一结果
                logger.info("共识不足，选择较好的单一检测结果")
                if yolo_result and traditional_result:
                    return yolo_result if len(yolo_result.nodes) > len(traditional_result.nodes) else traditional_result
                elif yolo_result:
                    return yolo_result
                elif traditional_result:
                    return traditional_result
                else:
                    raise RuntimeError("没有可用的检测结果")
            
        except Exception as e:
            logger.error(f"共识策略失败: {e}")
            raise
    
    def _fuse_results(self, yolo_result: Optional[DetectionResult], 
                     traditional_result: Optional[DetectionResult],
                     image_shape: Tuple[int, int]) -> DetectionResult:
        """融合两种检测结果"""
        try:
            if not yolo_result and not traditional_result:
                raise ValueError("没有检测结果可融合")
            
            if not yolo_result:
                return traditional_result
            if not traditional_result:
                return yolo_result
            
            # 融合节点
            fused_nodes = self._fuse_nodes(yolo_result.nodes, traditional_result.nodes)
            
            # 重新计算竹筒段和切割点
            fused_segments = []
            fused_cutting_points = []
            
            if len(fused_nodes) > 1:
                # 使用YOLO的分割方法（通常更准确）
                if self.yolo_detector:
                    fused_segments = self.yolo_detector._segment_bamboo_tubes_yolo(
                        np.zeros((*image_shape, 3), dtype=np.uint8), fused_nodes
                    )
                    fused_cutting_points = self.yolo_detector._calculate_cutting_points_yolo(
                        fused_nodes, fused_segments
                    )
            
            # 创建融合结果
            fused_result = DetectionResult(
                nodes=fused_nodes,
                segments=fused_segments,
                cutting_points=fused_cutting_points,
                processing_time=max(yolo_result.processing_time, traditional_result.processing_time),
                image_shape=image_shape,
                metadata={
                    'algorithm': 'hybrid_fusion',
                    'yolo_detections': len(yolo_result.nodes),
                    'traditional_detections': len(traditional_result.nodes),
                    'fused_detections': len(fused_nodes),
                    'fusion_weights': self.fusion_weights.copy()
                }
            )
            
            logger.info(f"结果融合完成: YOLO {len(yolo_result.nodes)} + 传统 {len(traditional_result.nodes)} → 融合 {len(fused_nodes)}")
            
            return fused_result
            
        except Exception as e:
            logger.error(f"结果融合失败: {e}")
            # 回退到较好的单一结果
            if yolo_result and traditional_result:
                return yolo_result if len(yolo_result.nodes) > len(traditional_result.nodes) else traditional_result
            return yolo_result or traditional_result
    
    def _fuse_nodes(self, yolo_nodes: List[BambooNode], traditional_nodes: List[BambooNode]) -> List[BambooNode]:
        """融合竹节检测结果"""
        fused_nodes = []
        distance_threshold = 30.0  # 像素距离阈值
        
        try:
            # 首先添加所有YOLO节点（权重较高）
            for yolo_node in yolo_nodes:
                # 加权调整置信度
                yolo_node.confidence *= self.fusion_weights['yolo']
                fused_nodes.append(yolo_node)
            
            # 添加不重复的传统算法节点
            for trad_node in traditional_nodes:
                # 检查是否与已有节点重复
                is_duplicate = False
                for fused_node in fused_nodes:
                    distance = np.sqrt((trad_node.position.x - fused_node.position.x)**2 + 
                                     (trad_node.position.y - fused_node.position.y)**2)
                    if distance < distance_threshold:
                        # 融合重复节点信息
                        weighted_conf = (fused_node.confidence + 
                                       trad_node.confidence * self.fusion_weights['traditional'])
                        fused_node.confidence = min(1.0, weighted_conf)
                        fused_node.features.update({
                            'fusion_source': 'both',
                            'traditional_confidence': trad_node.confidence
                        })
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    # 添加新的传统算法节点
                    trad_node.confidence *= self.fusion_weights['traditional']
                    trad_node.features['fusion_source'] = 'traditional'
                    fused_nodes.append(trad_node)
            
            # 按x坐标排序
            fused_nodes.sort(key=lambda n: n.position.x)
            
            return fused_nodes
            
        except Exception as e:
            logger.error(f"节点融合失败: {e}")
            return yolo_nodes + traditional_nodes  # 简单合并
    
    def _find_consensus_nodes(self, yolo_result: Optional[DetectionResult], 
                            traditional_result: Optional[DetectionResult]) -> List[BambooNode]:
        """查找两种方法的共识节点"""
        consensus_nodes = []
        distance_threshold = 25.0
        
        try:
            if not yolo_result or not traditional_result:
                return consensus_nodes
            
            for yolo_node in yolo_result.nodes:
                for trad_node in traditional_result.nodes:
                    distance = np.sqrt((yolo_node.position.x - trad_node.position.x)**2 + 
                                     (yolo_node.position.y - trad_node.position.y)**2)
                    
                    if distance < distance_threshold:
                        # 创建共识节点
                        consensus_confidence = (yolo_node.confidence + trad_node.confidence) / 2
                        
                        if consensus_confidence >= self.adaptive_thresholds['consensus_threshold']:
                            consensus_node = BambooNode(
                                position=Point2D(
                                    (yolo_node.position.x + trad_node.position.x) / 2,
                                    (yolo_node.position.y + trad_node.position.y) / 2
                                ),
                                position_mm=Point2D(
                                    (yolo_node.position_mm.x + trad_node.position_mm.x) / 2,
                                    (yolo_node.position_mm.y + trad_node.position_mm.y) / 2
                                ),
                                confidence=consensus_confidence,
                                node_type=yolo_node.node_type,  # 优先使用YOLO的分类
                                bbox=yolo_node.bbox,
                                width=(yolo_node.width + trad_node.width) / 2,
                                width_mm=(yolo_node.width_mm + trad_node.width_mm) / 2,
                                features={
                                    'consensus': True,
                                    'yolo_confidence': yolo_node.confidence,
                                    'traditional_confidence': trad_node.confidence,
                                    'distance': distance
                                }
                            )
                            consensus_nodes.append(consensus_node)
                        break
            
            return consensus_nodes
            
        except Exception as e:
            logger.error(f"共识节点查找失败: {e}")
            return consensus_nodes
    
    def _build_consensus_result(self, consensus_nodes: List[BambooNode],
                              yolo_result: Optional[DetectionResult],
                              traditional_result: Optional[DetectionResult],
                              image_shape: Tuple[int, int]) -> DetectionResult:
        """基于共识节点构建检测结果"""
        try:
            # 重新计算竹筒段和切割点
            segments = []
            cutting_points = []
            
            if len(consensus_nodes) > 1 and self.yolo_detector:
                segments = self.yolo_detector._segment_bamboo_tubes_yolo(
                    np.zeros((*image_shape, 3), dtype=np.uint8), consensus_nodes
                )
                cutting_points = self.yolo_detector._calculate_cutting_points_yolo(
                    consensus_nodes, segments
                )
            
            return DetectionResult(
                nodes=consensus_nodes,
                segments=segments,
                cutting_points=cutting_points,
                processing_time=max(
                    yolo_result.processing_time if yolo_result else 0,
                    traditional_result.processing_time if traditional_result else 0
                ),
                image_shape=image_shape,
                metadata={
                    'algorithm': 'hybrid_consensus',
                    'consensus_nodes': len(consensus_nodes),
                    'yolo_original': len(yolo_result.nodes) if yolo_result else 0,
                    'traditional_original': len(traditional_result.nodes) if traditional_result else 0
                }
            )
            
        except Exception as e:
            logger.error(f"共识结果构建失败: {e}")
            raise
    
    def _assess_image_quality(self, image: np.ndarray) -> float:
        """评估图像质量"""
        try:
            # 转换为灰度图
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # 计算多个质量指标
            
            # 1. 对比度 (标准差)
            contrast = np.std(gray) / 255.0
            
            # 2. 清晰度 (拉普拉斯变换的方差)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = np.var(laplacian) / 10000.0  # 归一化
            
            # 3. 亮度分布
            brightness = np.mean(gray) / 255.0
            brightness_score = 1.0 - abs(brightness - 0.5) * 2  # 中等亮度最好
            
            # 综合评分
            quality_score = (contrast * 0.4 + sharpness * 0.4 + brightness_score * 0.2)
            
            return min(1.0, quality_score)
            
        except Exception as e:
            logger.error(f"图像质量评估失败: {e}")
            return 0.5  # 默认中等质量
    
    def _create_empty_result(self, image_shape: Tuple[int, int], error_msg: str = "") -> DetectionResult:
        """创建空结果"""
        return DetectionResult(
            nodes=[],
            segments=[],
            cutting_points=[],
            processing_time=0.0,
            image_shape=image_shape,
            metadata={'error': error_msg, 'algorithm': 'hybrid_optimized'}
        )
    
    def export_optimized_models(self, formats: List[str] = None) -> Dict[str, str]:
        """
        导出优化模型
        Args:
            formats: 要导出的格式列表
        Returns:
            导出文件路径字典
        """
        if formats is None:
            formats = ['onnx', 'engine', 'openvino']
        
        exported_paths = {}
        
        if self.yolo_detector:
            try:
                for format_name in formats:
                    path = self.yolo_detector.export_optimized_model(format_name)
                    exported_paths[format_name] = path
                    logger.info(f"模型导出成功: {format_name} -> {path}")
            except Exception as e:
                logger.error(f"模型导出失败: {e}")
        
        return exported_paths
    
    def benchmark_performance(self, test_images: List[np.ndarray], 
                            strategies: List[HybridStrategy] = None,
                            num_runs: int = 5) -> Dict[str, Any]:
        """
        混合策略性能基准测试
        Args:
            test_images: 测试图像列表
            strategies: 要测试的策略列表
            num_runs: 每个策略的运行次数
        Returns:
            基准测试结果
        """
        if strategies is None:
            strategies = [
                HybridStrategy.PERFORMANCE_OPTIMIZED,
                HybridStrategy.YOLO_FIRST,
                HybridStrategy.TRADITIONAL_FIRST,
                HybridStrategy.PARALLEL_FUSION
            ]
        
        benchmark_results = {}
        original_strategy = self.strategy
        
        try:
            logger.info(f"开始混合策略基准测试，{len(strategies)}个策略，每个{num_runs}次运行")
            
            for strategy in strategies:
                logger.info(f"测试策略: {strategy.value}")
                self.set_strategy(strategy)
                
                # 单张处理测试
                single_times = []
                detection_counts = []
                
                for run in range(num_runs):
                    start_time = time.time()
                    total_detections = 0
                    
                    for img in test_images:
                        result = self.process_image(img)
                        total_detections += len(result.nodes)
                    
                    single_times.append(time.time() - start_time)
                    detection_counts.append(total_detections)
                
                # 批量处理测试
                batch_times = []
                if self.enable_batch_processing and len(test_images) > 1:
                    for run in range(num_runs):
                        start_time = time.time()
                        batch_results = self.process_images_batch(test_images)
                        batch_times.append(time.time() - start_time)
                
                # 统计结果
                strategy_results = {
                    'single_processing': {
                        'avg_time': np.mean(single_times),
                        'min_time': np.min(single_times),
                        'max_time': np.max(single_times),
                        'std_time': np.std(single_times),
                        'avg_detections': np.mean(detection_counts),
                        'throughput_fps': len(test_images) / np.mean(single_times)
                    }
                }
                
                if batch_times:
                    strategy_results['batch_processing'] = {
                        'avg_time': np.mean(batch_times),
                        'min_time': np.min(batch_times),
                        'max_time': np.max(batch_times),
                        'throughput_fps': len(test_images) / np.mean(batch_times),
                        'speedup_factor': np.mean(single_times) / np.mean(batch_times)
                    }
                
                benchmark_results[strategy.value] = strategy_results
            
            # 恢复原始策略
            self.set_strategy(original_strategy)
            
            logger.info("混合策略基准测试完成")
            return benchmark_results
            
        except Exception as e:
            logger.error(f"基准测试失败: {e}")
            self.set_strategy(original_strategy)
            return {}
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        stats = {
            'method_performance': self.method_performance.copy(),
            'current_strategy': self.strategy.value,
            'optimization_config': {
                'batch_processing_enabled': self.enable_batch_processing,
                'async_processing_enabled': self.enable_async_processing,
                'cache_enabled': self.cache_enabled,
                'performance_mode': self.performance_mode,
                'max_batch_size': self.max_batch_size
            }
        }
        
        # 添加YOLO性能统计
        if self.yolo_detector:
            stats['yolo_performance'] = self.yolo_detector.get_performance_stats()
        
        # 计算整体统计
        total_inferences = sum(method['count'] for method in self.method_performance.values())
        if total_inferences > 0:
            total_time = sum(method['total_time'] for method in self.method_performance.values())
            stats['overall'] = {
                'total_inferences': total_inferences,
                'avg_processing_time': total_time / total_inferences,
                'cache_hit_rate': len(self.result_cache) / max(total_inferences, 1)
            }
        
        return stats
    
    def optimize_for_deployment(self, target_device: str = 'auto', 
                              target_fps: float = 10.0) -> Dict[str, Any]:
        """
        针对部署优化配置
        Args:
            target_device: 目标设备 ('cpu', 'gpu', 'auto')
            target_fps: 目标帧率
        Returns:
            优化配置报告
        """
        try:
            logger.info(f"针对{target_device}设备优化配置，目标帧率{target_fps}FPS")
            
            optimization_report = {
                'target_device': target_device,
                'target_fps': target_fps,
                'optimizations_applied': []
            }
            
            # 设备特定优化
            if target_device == 'cpu' or (target_device == 'auto' and not torch.cuda.is_available()):
                # CPU优化
                self.set_strategy(HybridStrategy.TRADITIONAL_FIRST)
                self.enable_batch_processing = False
                self.enable_async_processing = False
                optimization_report['optimizations_applied'].extend([
                    'Traditional algorithm prioritized for CPU',
                    'Batch processing disabled',
                    'Async processing disabled'
                ])
                
                if self.yolo_detector:
                    self.yolo_detector.set_optimization_config(
                        enable_half_precision=False,
                        enable_threading=True,
                        auto_batch_size=False
                    )
                    optimization_report['optimizations_applied'].append('YOLO CPU optimization enabled')
            
            elif target_device in ['gpu', 'cuda'] or (target_device == 'auto' and torch.cuda.is_available()):
                # GPU优化
                self.set_strategy(HybridStrategy.PERFORMANCE_OPTIMIZED)
                self.enable_batch_processing = True
                self.enable_async_processing = target_fps < 5.0  # 低帧率使用异步
                optimization_report['optimizations_applied'].extend([
                    'Performance optimized strategy for GPU',
                    'Batch processing enabled',
                    f'Async processing {"enabled" if self.enable_async_processing else "disabled"}'
                ])
                
                if self.yolo_detector:
                    self.yolo_detector.set_optimization_config(
                        enable_half_precision=True,
                        enable_tensorrt=True,
                        auto_batch_size=True,
                        max_batch_size=min(16, int(target_fps))
                    )
                    optimization_report['optimizations_applied'].append('YOLO GPU optimization enabled')
            
            # 帧率特定优化
            if target_fps >= 30.0:
                # 高帧率优化
                self.adaptive_thresholds['performance_threshold'] = 0.033  # 33ms
                self.cache_enabled = True
                optimization_report['optimizations_applied'].extend([
                    'High FPS optimization',
                    'Result caching enabled'
                ])
            elif target_fps >= 10.0:
                # 中等帧率优化
                self.adaptive_thresholds['performance_threshold'] = 0.1  # 100ms
                optimization_report['optimizations_applied'].append('Medium FPS optimization')
            else:
                # 低帧率，追求质量
                self.set_strategy(HybridStrategy.PARALLEL_FUSION)
                optimization_report['optimizations_applied'].append('Low FPS - quality prioritized')
            
            logger.info("部署优化配置完成")
            return optimization_report
            
        except Exception as e:
            logger.error(f"部署优化失败: {e}")
            return {'error': str(e)}
    
    def cleanup(self):
        """清理资源"""
        try:
            if self.yolo_detector:
                self.yolo_detector.cleanup()
                self.yolo_detector = None
            
            if self.traditional_detector:
                self.traditional_detector.cleanup()
                self.traditional_detector = None
            
            # 清理缓存
            self.result_cache.clear()
            
            logger.info("优化版混合检测器资源清理完成")
            
        except Exception as e:
            logger.error(f"资源清理失败: {e}")


# 为了向后兼容，保留原始类名
HybridDetector = OptimizedHybridDetector


def register_optimized_hybrid_detector():
    """注册优化版混合检测器到工厂"""
    from .vision_processor import VisionProcessorFactory
    
    VisionProcessorFactory.register_processor("hybrid_optimized", OptimizedHybridDetector)
    VisionProcessorFactory.register_processor("hybrid", OptimizedHybridDetector)  # 默认使用优化版本
    logger.info("优化版混合检测器已注册到工厂")


# 自动注册
register_optimized_hybrid_detector() 