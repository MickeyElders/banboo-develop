"""
混合竹节检测器
结合传统计算机视觉算法和YOLOv8深度学习模型
"""

import logging
import time
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum

import cv2
import numpy as np

from .vision_processor import VisionProcessor, monitor_performance
from .traditional_detector import TraditionalDetector
from .yolo_detector import YOLODetector
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


class HybridDetector(VisionProcessor):
    """
    混合竹节检测器
    结合传统CV算法和YOLO深度学习模型
    """
    
    def __init__(self, config: AlgorithmConfig, calibration: Optional[CalibrationData] = None):
        super().__init__(config, calibration)
        
        # 子检测器
        self.traditional_detector: Optional[TraditionalDetector] = None
        self.yolo_detector: Optional[YOLODetector] = None
        
        # 混合策略配置
        self.strategy = HybridStrategy.YOLO_FIRST
        self.fusion_weights = {
            'yolo': 0.7,        # YOLO权重
            'traditional': 0.3   # 传统算法权重
        }
        
        # 自适应阈值
        self.adaptive_thresholds = {
            'min_yolo_detections': 2,     # YOLO最小检测数
            'min_traditional_detections': 3,  # 传统算法最小检测数
            'consensus_threshold': 0.6,    # 共识阈值
            'image_quality_threshold': 0.7  # 图像质量阈值
        }
        
        # 性能统计
        self.method_performance = {
            'yolo': {'count': 0, 'total_time': 0, 'avg_detections': 0},
            'traditional': {'count': 0, 'total_time': 0, 'avg_detections': 0},
            'hybrid': {'count': 0, 'total_time': 0, 'avg_detections': 0}
        }
        
        logger.info("混合检测器初始化完成")
    
    def initialize(self) -> bool:
        """
        初始化混合检测器
        Returns:
            是否成功初始化
        """
        try:
            logger.info("初始化混合检测器...")
            
            # 初始化传统检测器
            self.traditional_detector = TraditionalDetector(self.config, self.calibration)
            traditional_ok = self.traditional_detector.initialize()
            
            # 初始化YOLO检测器
            self.yolo_detector = YOLODetector(self.config, self.calibration)
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
            
            self.is_initialized = True
            logger.info(f"混合检测器初始化成功，策略: {self.strategy.value}")
            return True
            
        except Exception as e:
            logger.error(f"混合检测器初始化失败: {e}")
            return False
    
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
        
        logger.info(f"检测策略已更新为: {strategy.value}")
    
    @monitor_performance
    def process_image(self, image: np.ndarray, roi: Optional[ProcessingROI] = None) -> DetectionResult:
        """
        使用混合策略处理图像
        Args:
            image: 输入图像
            roi: 感兴趣区域（可选）
        Returns:
            检测结果
        """
        try:
            logger.debug(f"开始混合检测，策略: {self.strategy.value}")
            
            start_time = time.time()
            
            # 根据策略选择检测方法
            if self.strategy == HybridStrategy.YOLO_FIRST:
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
            self.method_performance['hybrid']['count'] += 1
            self.method_performance['hybrid']['total_time'] += processing_time
            self.method_performance['hybrid']['avg_detections'] = (
                (self.method_performance['hybrid']['avg_detections'] * 
                 (self.method_performance['hybrid']['count'] - 1) + len(result.nodes)) /
                self.method_performance['hybrid']['count']
            )
            
            # 添加混合策略元数据
            result.metadata.update({
                'hybrid_strategy': self.strategy.value,
                'fusion_weights': self.fusion_weights.copy(),
                'performance_stats': self.get_performance_stats()
            })
            
            logger.info(f"混合检测完成: {len(result.nodes)}个竹节, {len(result.segments)}个竹筒段, "
                       f"{len(result.cutting_points)}个切割点, 耗时{processing_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"混合检测失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # 返回空结果
            return DetectionResult(
                nodes=[],
                segments=[],
                cutting_points=[],
                processing_time=0.0,
                image_shape=image.shape[:2],
                metadata={'error': str(e), 'algorithm': 'hybrid'}
            )
    
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
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        stats = {
            'strategy': self.strategy.value,
            'method_performance': {}
        }
        
        for method, perf in self.method_performance.items():
            if perf['count'] > 0:
                stats['method_performance'][method] = {
                    'count': perf['count'],
                    'avg_time': perf['total_time'] / perf['count'],
                    'avg_detections': perf['avg_detections']
                }
        
        return stats
    
    def cleanup(self):
        """清理资源"""
        try:
            if self.traditional_detector:
                self.traditional_detector.cleanup()
                self.traditional_detector = None
            
            if self.yolo_detector:
                self.yolo_detector.cleanup()
                self.yolo_detector = None
            
            self.is_initialized = False
            logger.info("混合检测器资源清理完成")
            
        except Exception as e:
            logger.error(f"资源清理失败: {e}")


def register_hybrid_detector():
    """注册混合检测器到工厂"""
    from .vision_processor import VisionProcessorFactory
    VisionProcessorFactory.register('hybrid', HybridDetector)
    logger.info("混合检测器已注册")


# 自动注册
register_hybrid_detector() 