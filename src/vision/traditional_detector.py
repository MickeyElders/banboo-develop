#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能切竹机 - 传统计算机视觉检测器
基于传统图像处理算法的竹节检测和竹筒分割实现
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
import logging
from scipy import ndimage
from sklearn.cluster import KMeans

from .vision_processor import VisionProcessor, monitor_performance
from .vision_types import (
    DetectionResult, BambooNode, BambooSegment, CuttingPoint,
    Point2D, BoundingBox, ProcessingROI, NodeType, 
    SegmentQuality, CuttingType, AlgorithmConfig, CalibrationData
)

logger = logging.getLogger(__name__)


class TraditionalDetector(VisionProcessor):
    """传统计算机视觉检测器"""
    
    def __init__(self, config: AlgorithmConfig, calibration: Optional[CalibrationData] = None):
        """
        初始化传统检测器
        Args:
            config: 算法配置参数
            calibration: 相机标定数据
        """
        super().__init__(config, calibration)
        
        # 算法特定参数
        self.edge_image = None
        self.contours = None
        self.bamboo_centerline = None
        
        logger.info("传统计算机视觉检测器初始化完成")
    
    def initialize(self) -> bool:
        """
        初始化检测器
        Returns:
            初始化是否成功
        """
        try:
            # 验证配置参数
            if not self.config.validate():
                logger.error("配置参数验证失败")
                return False
            
            self.is_initialized = True
            logger.info("传统检测器初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"初始化失败: {e}")
            return False
    
    @monitor_performance
    def process_image(self, image: np.ndarray, roi: Optional[ProcessingROI] = None) -> DetectionResult:
        """
        处理图像并检测竹节
        Args:
            image: 输入图像
            roi: 感兴趣区域（可选）
        Returns:
            检测结果
        """
        try:
            logger.debug("开始图像处理")
            
            # 预处理
            processed_image = self.preprocess_image(image)
            
            # 使用简单检测方法
            nodes = self._simple_node_detection(processed_image)
            
            # 如果简单方法失败，尝试传统方法
            if len(nodes) == 0:
                logger.info("简单检测未找到竹节，尝试传统方法")
                
                # 检测中心线
                centerline = self._detect_bamboo_centerline(processed_image)
                
                # 边缘检测
                edges = self._detect_edges(processed_image)
                
                # 传统竹节检测
                nodes = self._detect_bamboo_nodes(processed_image, edges, centerline)
            
            # 分割竹筒段
            segments = []
            if len(nodes) > 1:
                # 需要中心线来分割竹筒段
                centerline = self._detect_bamboo_centerline(processed_image)
                segments = self._segment_bamboo_tubes(processed_image, nodes, centerline)
            else:
                logger.warning("竹节数量不足，无法分割竹筒段")
            
            # 计算切割点
            cutting_points = self._calculate_cutting_points(nodes, segments)
            
            # 创建结果
            result = DetectionResult(
                nodes=nodes,
                segments=segments,
                cutting_points=cutting_points,
                processing_time=0.0,  # 将在装饰器中设置
                image_shape=image.shape[:2],
                metadata={
                    'algorithm': 'traditional_cv',
                    'centerline_points': 1,  # 临时值
                    'edge_pixels': 0  # 临时值
                }
            )
            
            logger.info(f"检测完成: {len(nodes)}个竹节, {len(segments)}个竹筒段, {len(cutting_points)}个切割点")
            
            return result
            
        except Exception as e:
            logger.error(f"图像处理失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # 返回空结果
            return DetectionResult(
                nodes=[],
                segments=[],
                cutting_points=[],
                processing_time=0.0,
                image_shape=image.shape[:2],
                metadata={'error': str(e)}
            )
    
    def _simple_node_detection(self, image: np.ndarray) -> List[BambooNode]:
        """
        简单直接的竹节检测方法
        基于边缘投影和颜色差异
        Args:
            image: 输入图像
        Returns:
            检测到的竹节列表
        """
        nodes = []
        
        try:
            # 转换为灰度图
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            logger.debug(f"图像尺寸: {gray.shape}, 像素值范围: {gray.min()}-{gray.max()}")
            
            # 边缘检测
            edges = cv2.Canny(gray, 30, 100)
            edge_count = np.sum(edges > 0)
            logger.debug(f"边缘像素数: {edge_count}")
            
            if edge_count < 100:  # 如果边缘太少，降低阈值
                edges = cv2.Canny(gray, 20, 80)
                logger.debug("使用更低的边缘检测阈值")
            
            # 水平投影 - 统计每列的边缘像素数
            h_projection = np.sum(edges, axis=0)
            
            # 平滑投影信号
            from scipy.ndimage import gaussian_filter1d
            h_projection_smooth = gaussian_filter1d(h_projection.astype(float), sigma=2)
            
            # 寻找峰值（潜在的竹节位置）
            from scipy.signal import find_peaks
            
            # 动态调整阈值
            mean_proj = np.mean(h_projection_smooth)
            std_proj = np.std(h_projection_smooth)
            threshold = max(mean_proj + 0.5 * std_proj, 20)  # 最小阈值20
            
            logger.debug(f"投影统计: 均值={mean_proj:.2f}, 标准差={std_proj:.2f}, 阈值={threshold:.2f}")
            
            peaks, properties = find_peaks(h_projection_smooth, 
                                         height=threshold, 
                                         distance=30)  # 竹节间最小距离30像素
            
            logger.debug(f"找到 {len(peaks)} 个候选峰值")
            
            # 验证和创建竹节
            for i, peak_x in enumerate(peaks):
                peak_height = h_projection_smooth[peak_x]
                
                # 估算竹节的y坐标（图像中心区域）
                y_center = gray.shape[0] // 2
                
                # 在竹节位置搜索确切的y坐标
                search_region = edges[max(0, y_center-50):min(gray.shape[0], y_center+50), 
                                    max(0, peak_x-10):min(gray.shape[1], peak_x+10)]
                
                if search_region.size > 0:
                    # 找到边缘密度最高的y位置
                    y_projection = np.sum(search_region, axis=1)
                    if len(y_projection) > 0:
                        local_y = np.argmax(y_projection)
                        actual_y = max(0, y_center-50) + local_y
                    else:
                        actual_y = y_center
                else:
                    actual_y = y_center
                
                # 估算节点宽度
                node_width = self._estimate_simple_node_width(edges, int(peak_x), int(actual_y))
                
                # 计算置信度
                confidence = min(1.0, peak_height / (mean_proj + 2 * std_proj))
                
                # 创建竹节
                if (self.config.node_min_width <= node_width <= self.config.node_max_width and
                    confidence >= self.config.node_confidence_threshold):
                    
                    node = BambooNode(
                        position=Point2D(float(peak_x), float(actual_y)),
                        position_mm=Point2D(
                            float(peak_x) * self.calibration.pixel_to_mm_ratio if self.calibration else float(peak_x),
                            float(actual_y) * self.calibration.pixel_to_mm_ratio if self.calibration else float(actual_y)
                        ),
                        confidence=confidence,
                        node_type=NodeType.NATURAL_NODE,
                        bbox=BoundingBox(
                            max(0, peak_x - node_width//2),
                            max(0, actual_y - 20),
                            min(gray.shape[1], peak_x + node_width//2),
                            min(gray.shape[0], actual_y + 20)
                        ),
                        width=node_width,
                        width_mm=node_width * self.calibration.pixel_to_mm_ratio if self.calibration else node_width,
                        features={'edge_strength': float(peak_height), 'method': 'simple_projection'}
                    )
                    nodes.append(node)
                    
                    logger.debug(f"创建竹节: x={peak_x}, y={actual_y}, 宽度={node_width:.1f}, 置信度={confidence:.3f}")
            
            logger.info(f"简单检测方法找到 {len(nodes)} 个竹节")
            
        except Exception as e:
            logger.error(f"简单竹节检测失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        return nodes
    
    def _estimate_simple_node_width(self, edges: np.ndarray, x: int, y: int) -> float:
        """
        简单的节点宽度估算
        Args:
            edges: 边缘图像
            x, y: 节点中心位置
        Returns:
            估算的宽度
        """
        try:
            # 在y坐标附近搜索水平边缘
            search_height = 10
            y_start = max(0, y - search_height)
            y_end = min(edges.shape[0], y + search_height)
            
            # 获取水平线段
            horizontal_line = edges[y_start:y_end, :]
            
            if horizontal_line.size == 0:
                return 20.0  # 默认宽度
            
            # 在x附近寻找边缘的左右边界
            center_region = np.sum(horizontal_line, axis=0)
            
            # 从中心向左右扩展寻找边界
            left_bound = x
            right_bound = x
            
            # 向左寻找
            for i in range(x, max(0, x-50), -1):
                if i < len(center_region) and center_region[i] > 0:
                    left_bound = i
                else:
                    break
            
            # 向右寻找
            for i in range(x, min(len(center_region), x+50)):
                if center_region[i] > 0:
                    right_bound = i
                else:
                    break
            
            width = right_bound - left_bound
            return max(5.0, min(50.0, float(width)))  # 限制在合理范围内
            
        except Exception as e:
            logger.error(f"宽度估算失败: {e}")
            return 20.0
    
    def _detect_bamboo_centerline(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        检测竹子中心线
        Args:
            image: 预处理后的图像
        Returns:
            中心线点集
        """
        try:
            # 使用形态学操作提取主体结构
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
            
            # 二值化
            _, binary = cv2.threshold(opened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 骨架化
            skeleton = self._skeletonize(binary)
            
            # 提取最长的连续线段作为中心线
            contours, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                logger.warning("未找到中心线")
                return None
            
            # 选择最长的轮廓
            longest_contour = max(contours, key=cv2.contourArea)
            
            # 拟合中心线
            centerline = self._fit_centerline(longest_contour)
            
            logger.debug(f"检测到中心线，包含{len(centerline)}个点")
            return centerline
            
        except Exception as e:
            logger.error(f"中心线检测失败: {e}")
            return None
    
    def _skeletonize(self, binary_image: np.ndarray) -> np.ndarray:
        """
        图像骨架化
        Args:
            binary_image: 二值图像
        Returns:
            骨架图像
        """
        # 使用形态学细化
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        skeleton = np.zeros_like(binary_image)
        
        while True:
            eroded = cv2.erode(binary_image, kernel)
            opened = cv2.dilate(eroded, kernel)
            skeleton = cv2.bitwise_or(skeleton, cv2.subtract(binary_image, opened))
            binary_image = eroded.copy()
            
            if cv2.countNonZero(binary_image) == 0:
                break
        
        return skeleton
    
    def _fit_centerline(self, contour: np.ndarray) -> np.ndarray:
        """
        拟合中心线
        Args:
            contour: 轮廓点
        Returns:
            拟合的中心线点
        """
        # 将轮廓点重塑为2D数组
        points = contour.reshape(-1, 2)
        
        # 按x坐标排序
        sorted_points = points[np.argsort(points[:, 0])]
        
        # 多项式拟合
        if len(sorted_points) > 3:
            z = np.polyfit(sorted_points[:, 0], sorted_points[:, 1], 3)
            p = np.poly1d(z)
            
            # 生成拟合点
            x_new = np.linspace(sorted_points[0, 0], sorted_points[-1, 0], 100)
            y_new = p(x_new)
            
            centerline = np.column_stack([x_new, y_new])
        else:
            centerline = sorted_points
        
        return centerline
    
    def _detect_edges(self, image: np.ndarray) -> np.ndarray:
        """
        边缘检测
        Args:
            image: 预处理后的图像
        Returns:
            边缘图像
        """
        # Canny边缘检测
        edges = cv2.Canny(image, 
                         self.config.canny_low_threshold,
                         self.config.canny_high_threshold,
                         apertureSize=self.config.canny_aperture_size)
        
        # 形态学闭运算连接断裂的边缘
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        self.edge_image = edges
        return edges
    
    def _detect_bamboo_nodes(self, image: np.ndarray, edges: np.ndarray, 
                           centerline: Optional[np.ndarray]) -> List[BambooNode]:
        """
        检测竹节
        Args:
            image: 预处理后的图像
            edges: 边缘图像
            centerline: 中心线
        Returns:
            检测到的竹节列表
        """
        nodes = []
        
        try:
            # 方法1: 基于边缘密度的检测
            nodes_method1 = self._detect_nodes_by_edge_density(edges, centerline)
            
            # 方法2: 基于纹理变化的检测
            nodes_method2 = self._detect_nodes_by_texture(image, centerline)
            
            # 方法3: 基于轮廓分析的检测
            nodes_method3 = self._detect_nodes_by_contour(edges)
            
            # 融合多种方法的结果
            all_candidates = nodes_method1 + nodes_method2 + nodes_method3
            
            # 非极大值抑制和验证
            nodes = self._filter_and_validate_nodes(all_candidates, image)
            
            logger.debug(f"检测到{len(nodes)}个竹节")
            
        except Exception as e:
            logger.error(f"竹节检测失败: {e}")
        
        return nodes
    
    def _detect_nodes_by_edge_density(self, edges: np.ndarray, 
                                    centerline: Optional[np.ndarray]) -> List[BambooNode]:
        """
        基于边缘密度检测竹节
        Args:
            edges: 边缘图像
            centerline: 中心线
        Returns:
            候选竹节列表
        """
        candidates = []
        
        if centerline is None:
            return candidates
        
        try:
            # 沿中心线计算边缘密度
            window_size = 20  # 窗口大小
            
            for i in range(len(centerline) - window_size):
                # 当前窗口中心点
                center_point = centerline[i + window_size // 2]
                
                # 计算窗口内的边缘密度
                x, y = int(center_point[0]), int(center_point[1])
                
                # 边界检查
                if (y - window_size//2 < 0 or y + window_size//2 >= edges.shape[0] or
                    x - window_size//2 < 0 or x + window_size//2 >= edges.shape[1]):
                    continue
                
                window = edges[y-window_size//2:y+window_size//2, 
                              x-window_size//2:x+window_size//2]
                
                edge_density = np.sum(window > 0) / window.size
                
                # 如果边缘密度超过阈值，认为是竹节候选
                if edge_density > 0.3:  # 阈值可调
                    # 计算节点宽度
                    node_width = self._estimate_node_width(edges, x, y)
                    
                    if (self.config.node_min_width <= node_width <= self.config.node_max_width):
                        # 创建候选节点
                        node = self._create_candidate_node(
                            Point2D(x, y), node_width, edge_density, NodeType.UNKNOWN
                        )
                        candidates.append(node)
            
        except Exception as e:
            logger.error(f"边缘密度检测失败: {e}")
        
        return candidates
    
    def _detect_nodes_by_texture(self, image: np.ndarray, 
                                centerline: Optional[np.ndarray]) -> List[BambooNode]:
        """
        基于纹理变化检测竹节
        Args:
            image: 预处理后的图像
            centerline: 中心线
        Returns:
            候选竹节列表
        """
        candidates = []
        
        if centerline is None:
            return candidates
        
        try:
            # 计算局部二值模式(LBP)或使用梯度幅值
            # 这里使用简化的方法：局部标准差
            
            kernel_size = 15
            kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
            
            # 计算局部均值和标准差
            mean_img = cv2.filter2D(image.astype(np.float32), -1, kernel)
            sqr_img = cv2.filter2D((image.astype(np.float32))**2, -1, kernel)
            std_img = np.sqrt(sqr_img - mean_img**2)
            
            # 在中心线附近寻找纹理变化
            for i in range(len(centerline)):
                x, y = int(centerline[i][0]), int(centerline[i][1])
                
                # 边界检查
                if (y < kernel_size//2 or y >= std_img.shape[0] - kernel_size//2 or
                    x < kernel_size//2 or x >= std_img.shape[1] - kernel_size//2):
                    continue
                
                # 获取当前点的纹理强度
                texture_strength = std_img[y, x]
                
                # 纹理强度阈值
                if texture_strength > np.mean(std_img) + 2 * np.std(std_img):
                    node_width = self._estimate_node_width_texture(std_img, x, y)
                    
                    if (self.config.node_min_width <= node_width <= self.config.node_max_width):
                        confidence = min(1.0, texture_strength / 100.0)
                        node = self._create_candidate_node(
                            Point2D(x, y), node_width, confidence, NodeType.NATURAL_NODE
                        )
                        candidates.append(node)
            
        except Exception as e:
            logger.error(f"纹理检测失败: {e}")
        
        return candidates
    
    def _detect_nodes_by_contour(self, edges: np.ndarray) -> List[BambooNode]:
        """
        基于轮廓分析检测竹节
        Args:
            edges: 边缘图像
        Returns:
            候选竹节列表
        """
        candidates = []
        
        try:
            # 查找轮廓
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # 过滤小轮廓
                area = cv2.contourArea(contour)
                if area < self.config.min_contour_area or area > self.config.max_contour_area:
                    continue
                
                # 计算轮廓特征
                bbox = cv2.boundingRect(contour)
                aspect_ratio = bbox[2] / bbox[3] if bbox[3] > 0 else 0
                
                # 过滤长宽比
                if not (self.config.min_aspect_ratio <= aspect_ratio <= self.config.max_aspect_ratio):
                    continue
                
                # 计算轮廓中心
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # 估算节点宽度
                    node_width = max(bbox[2], bbox[3])
                    
                    if (self.config.node_min_width <= node_width <= self.config.node_max_width):
                        # 基于轮廓面积和周长计算置信度
                        perimeter = cv2.arcLength(contour, True)
                        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                        confidence = min(1.0, circularity * 2)  # 圆形度越高置信度越高
                        
                        node = self._create_candidate_node(
                            Point2D(cx, cy), node_width, confidence, NodeType.UNKNOWN
                        )
                        candidates.append(node)
            
        except Exception as e:
            logger.error(f"轮廓检测失败: {e}")
        
        return candidates
    
    def _create_candidate_node(self, position: Point2D, width: float, 
                             confidence: float, node_type: NodeType) -> BambooNode:
        """
        创建候选竹节
        Args:
            position: 节点位置
            width: 节点宽度
            confidence: 置信度
            node_type: 节点类型
        Returns:
            竹节对象
        """
        # 转换坐标到毫米
        position_mm = self.pixel_to_mm_conversion(position)
        width_mm = width * (self.calibration.pixel_to_mm_ratio if self.calibration else 0.5)
        
        # 创建边界框
        half_width = width / 2
        bbox = BoundingBox(
            position.x - half_width,
            position.y - half_width,
            position.x + half_width,
            position.y + half_width
        )
        
        # 特征信息
        features = {
            'detection_method': 'traditional_cv',
            'width_pixels': width,
            'width_mm': width_mm
        }
        
        return BambooNode(
            position=position,
            position_mm=position_mm,
            confidence=confidence,
            node_type=node_type,
            bbox=bbox,
            width=width,
            width_mm=width_mm,
            features=features
        )
    
    def _estimate_node_width(self, edges: np.ndarray, x: int, y: int) -> float:
        """
        估算节点宽度
        Args:
            edges: 边缘图像
            x, y: 节点中心坐标
        Returns:
            节点宽度（像素）
        """
        # 在水平方向搜索边缘
        row = edges[y, :]
        left_edge = x
        right_edge = x
        
        # 向左搜索
        for i in range(x, -1, -1):
            if row[i] > 0:
                left_edge = i
                break
        
        # 向右搜索
        for i in range(x, len(row)):
            if row[i] > 0:
                right_edge = i
                break
        
        return abs(right_edge - left_edge)
    
    def _estimate_node_width_texture(self, texture_img: np.ndarray, x: int, y: int) -> float:
        """
        基于纹理估算节点宽度
        Args:
            texture_img: 纹理强度图像
            x, y: 节点中心坐标
        Returns:
            节点宽度（像素）
        """
        # 简化方法：在纹理图像中搜索高强度区域的宽度
        row = texture_img[y, :]
        threshold = texture_img[y, x] * 0.5  # 当前点强度的一半作为阈值
        
        left_bound = x
        right_bound = x
        
        # 向左搜索
        for i in range(x, max(0, x-50), -1):
            if row[i] < threshold:
                left_bound = i
                break
        
        # 向右搜索
        for i in range(x, min(len(row), x+50)):
            if row[i] < threshold:
                right_bound = i
                break
        
        return abs(right_bound - left_bound)
    
    def _filter_and_validate_nodes(self, candidates: List[BambooNode], 
                                 image: np.ndarray) -> List[BambooNode]:
        """
        过滤和验证候选节点
        Args:
            candidates: 候选节点列表
            image: 原始图像
        Returns:
            验证后的节点列表
        """
        if not candidates:
            return []
        
        # 非极大值抑制
        filtered_nodes = self._non_maximum_suppression(candidates)
        
        # 验证节点特征
        validated_nodes = []
        for node in filtered_nodes:
            if self._validate_node(node, image):
                validated_nodes.append(node)
        
        # 按置信度排序
        validated_nodes.sort(key=lambda n: n.confidence, reverse=True)
        
        return validated_nodes
    
    def _non_maximum_suppression(self, candidates: List[BambooNode], 
                                distance_threshold: float = 30.0) -> List[BambooNode]:
        """
        非极大值抑制
        Args:
            candidates: 候选节点列表
            distance_threshold: 距离阈值
        Returns:
            抑制后的节点列表
        """
        if not candidates:
            return []
        
        # 按置信度排序
        sorted_candidates = sorted(candidates, key=lambda n: n.confidence, reverse=True)
        
        filtered = []
        for candidate in sorted_candidates:
            # 检查是否与已选择的节点距离过近
            too_close = False
            for selected in filtered:
                distance = candidate.position.distance_to(selected.position)
                if distance < distance_threshold:
                    too_close = True
                    break
            
            if not too_close:
                filtered.append(candidate)
        
        return filtered
    
    def _validate_node(self, node: BambooNode, image: np.ndarray) -> bool:
        """
        验证节点有效性
        Args:
            node: 待验证的节点
            image: 原始图像
        Returns:
            节点是否有效
        """
        # 置信度阈值检查
        if node.confidence < self.config.node_confidence_threshold:
            return False
        
        # 位置有效性检查
        x, y = int(node.position.x), int(node.position.y)
        if x < 0 or x >= image.shape[1] or y < 0 or y >= image.shape[0]:
            return False
        
        # 宽度合理性检查
        if not (self.config.node_min_width <= node.width <= self.config.node_max_width):
            return False
        
        return True
    
    def _segment_bamboo_tubes(self, image: np.ndarray, nodes: List[BambooNode], 
                            centerline: Optional[np.ndarray]) -> List[BambooSegment]:
        """
        分割竹筒段
        Args:
            image: 预处理后的图像
            nodes: 检测到的竹节
            centerline: 中心线
        Returns:
            竹筒段列表
        """
        segments = []
        
        if not nodes or len(nodes) < 2:
            logger.warning("竹节数量不足，无法分割竹筒段")
            return segments
        
        try:
            # 按位置排序节点
            sorted_nodes = sorted(nodes, key=lambda n: n.position.x)
            
            # 在相邻节点间创建竹筒段
            for i in range(len(sorted_nodes) - 1):
                start_node = sorted_nodes[i]
                end_node = sorted_nodes[i + 1]
                
                # 计算段的特征
                segment = self._create_bamboo_segment(start_node, end_node, image)
                
                if segment and segment.length_mm >= self.config.segment_min_length:
                    segments.append(segment)
            
            logger.debug(f"分割出{len(segments)}个竹筒段")
            
        except Exception as e:
            logger.error(f"竹筒分割失败: {e}")
        
        return segments
    
    def _create_bamboo_segment(self, start_node: BambooNode, end_node: BambooNode, 
                             image: np.ndarray) -> Optional[BambooSegment]:
        """
        创建竹筒段
        Args:
            start_node: 起始节点
            end_node: 结束节点
            image: 图像
        Returns:
            竹筒段对象
        """
        try:
            # 计算段的起始和结束位置（避开节点区域）
            direction_x = end_node.position.x - start_node.position.x
            direction_y = end_node.position.y - start_node.position.y
            length = np.sqrt(direction_x**2 + direction_y**2)
            
            if length == 0:
                return None
            
            # 单位方向向量
            unit_x = direction_x / length
            unit_y = direction_y / length
            
            # 起始位置：从起始节点边缘开始
            offset = start_node.width / 2
            start_pos = Point2D(
                start_node.position.x + unit_x * offset,
                start_node.position.y + unit_y * offset
            )
            
            # 结束位置：到结束节点边缘
            end_pos = Point2D(
                end_node.position.x - unit_x * offset,
                end_node.position.y - unit_y * offset
            )
            
            # 转换为毫米坐标
            start_pos_mm = self.pixel_to_mm_conversion(start_pos)
            end_pos_mm = self.pixel_to_mm_conversion(end_pos)
            
            # 计算长度
            segment_length_mm = start_pos_mm.distance_to(end_pos_mm)
            
            # 估算直径
            diameter_mm = self._estimate_segment_diameter(start_pos, end_pos, image)
            
            # 质量评估
            quality, quality_score, defects = self._assess_segment_quality(
                start_pos, end_pos, image, diameter_mm
            )
            
            # 创建边界框
            bbox = BoundingBox(
                min(start_pos.x, end_pos.x) - 10,
                min(start_pos.y, end_pos.y) - 10,
                max(start_pos.x, end_pos.x) + 10,
                max(start_pos.y, end_pos.y) + 10
            )
            
            return BambooSegment(
                start_pos=start_pos,
                end_pos=end_pos,
                start_pos_mm=start_pos_mm,
                end_pos_mm=end_pos_mm,
                length_mm=segment_length_mm,
                diameter_mm=diameter_mm,
                quality=quality,
                quality_score=quality_score,
                defects=defects,
                bbox=bbox
            )
            
        except Exception as e:
            logger.error(f"创建竹筒段失败: {e}")
            return None
    
    def _estimate_segment_diameter(self, start_pos: Point2D, end_pos: Point2D, 
                                 image: np.ndarray) -> float:
        """
        估算竹筒段直径
        Args:
            start_pos: 起始位置
            end_pos: 结束位置
            image: 图像
        Returns:
            直径（毫米）
        """
        # 简化方法：在中点处测量垂直方向的宽度
        mid_x = (start_pos.x + end_pos.x) / 2
        mid_y = (start_pos.y + end_pos.y) / 2
        
        # 计算垂直方向
        dx = end_pos.x - start_pos.x
        dy = end_pos.y - start_pos.y
        length = np.sqrt(dx**2 + dy**2)
        
        if length == 0:
            return 20.0  # 默认直径
        
        # 垂直单位向量
        perp_x = -dy / length
        perp_y = dx / length
        
        # 在垂直方向搜索边缘
        max_search = 50  # 最大搜索距离
        
        # 向上搜索
        top_edge = mid_y
        for i in range(1, max_search):
            y = int(mid_y + perp_y * i)
            x = int(mid_x + perp_x * i)
            
            if (0 <= y < image.shape[0] and 0 <= x < image.shape[1]):
                # 检查是否到达边缘（亮度变化）
                if i > 5 and abs(int(image[y, x]) - int(image[int(mid_y), int(mid_x)])) > 30:
                    top_edge = y
                    break
            else:
                break
        
        # 向下搜索
        bottom_edge = mid_y
        for i in range(1, max_search):
            y = int(mid_y - perp_y * i)
            x = int(mid_x - perp_x * i)
            
            if (0 <= y < image.shape[0] and 0 <= x < image.shape[1]):
                if i > 5 and abs(int(image[y, x]) - int(image[int(mid_y), int(mid_x)])) > 30:
                    bottom_edge = y
                    break
            else:
                break
        
        # 计算直径（像素）
        diameter_pixels = abs(top_edge - bottom_edge)
        
        # 转换为毫米
        pixel_to_mm = self.calibration.pixel_to_mm_ratio if self.calibration else 0.5
        diameter_mm = diameter_pixels * pixel_to_mm
        
        return max(5.0, min(50.0, diameter_mm))  # 限制在合理范围内
    
    def _assess_segment_quality(self, start_pos: Point2D, end_pos: Point2D, 
                              image: np.ndarray, diameter_mm: float) -> Tuple[SegmentQuality, float, List[str]]:
        """
        评估竹筒段质量
        Args:
            start_pos: 起始位置
            end_pos: 结束位置
            image: 图像
            diameter_mm: 直径
        Returns:
            质量等级, 质量评分, 缺陷列表
        """
        defects = []
        quality_factors = []
        
        try:
            # 长度评分
            length_mm = self.pixel_to_mm_conversion(start_pos).distance_to(
                self.pixel_to_mm_conversion(end_pos)
            )
            
            if length_mm < 100:
                quality_factors.append(0.6)
                defects.append("长度过短")
            elif length_mm > 300:
                quality_factors.append(0.9)
            else:
                quality_factors.append(0.8)
            
            # 直径评分
            if 10 <= diameter_mm <= 30:
                quality_factors.append(0.9)
            elif 5 <= diameter_mm <= 40:
                quality_factors.append(0.7)
            else:
                quality_factors.append(0.5)
                defects.append("直径异常")
            
            # 表面质量评分（基于亮度方差）
            surface_score = self._assess_surface_quality(start_pos, end_pos, image)
            quality_factors.append(surface_score)
            
            if surface_score < 0.6:
                defects.append("表面质量差")
            
            # 计算总体质量评分
            quality_score = np.mean(quality_factors)
            
            # 确定质量等级
            if quality_score >= 0.9:
                quality = SegmentQuality.EXCELLENT
            elif quality_score >= 0.8:
                quality = SegmentQuality.GOOD
            elif quality_score >= 0.6:
                quality = SegmentQuality.ACCEPTABLE
            elif quality_score >= 0.4:
                quality = SegmentQuality.POOR
            else:
                quality = SegmentQuality.UNUSABLE
            
        except Exception as e:
            logger.error(f"质量评估失败: {e}")
            quality = SegmentQuality.POOR
            quality_score = 0.5
            defects = ["评估失败"]
        
        return quality, quality_score, defects
    
    def _assess_surface_quality(self, start_pos: Point2D, end_pos: Point2D, 
                              image: np.ndarray) -> float:
        """
        评估表面质量
        Args:
            start_pos: 起始位置
            end_pos: 结束位置
            image: 图像
        Returns:
            表面质量评分 [0-1]
        """
        try:
            # 提取竹筒段区域
            x1, y1 = int(start_pos.x), int(start_pos.y)
            x2, y2 = int(end_pos.x), int(end_pos.y)
            
            # 创建掩码
            mask = np.zeros(image.shape, dtype=np.uint8)
            cv2.line(mask, (x1, y1), (x2, y2), 255, thickness=20)
            
            # 提取区域像素值
            region_pixels = image[mask > 0]
            
            if len(region_pixels) == 0:
                return 0.5
            
            # 计算亮度统计
            mean_brightness = np.mean(region_pixels)
            std_brightness = np.std(region_pixels)
            
            # 基于亮度均匀性评分
            uniformity_score = max(0, 1 - std_brightness / 100.0)
            
            # 基于平均亮度评分（避免过暗或过亮）
            brightness_score = 1 - abs(mean_brightness - 128) / 128.0
            
            # 综合评分
            surface_score = (uniformity_score + brightness_score) / 2
            
            return max(0.0, min(1.0, surface_score))
            
        except Exception as e:
            logger.error(f"表面质量评估失败: {e}")
            return 0.5
    
    def _calculate_cutting_points(self, nodes: List[BambooNode], 
                                segments: List[BambooSegment]) -> List[CuttingPoint]:
        """
        计算切割点
        Args:
            nodes: 检测到的竹节
            segments: 竹筒段
        Returns:
            切割点列表
        """
        cutting_points = []
        
        try:
            # 为每个竹节创建切割点
            for i, node in enumerate(nodes):
                # 节点移除切割点
                cut_point = CuttingPoint(
                    position=node.position,
                    position_mm=node.position_mm,
                    cutting_type=CuttingType.REMOVE_NODE,
                    priority=8,  # 高优先级
                    confidence=node.confidence,
                    reason=f"移除{node.node_type.value}节点",
                    related_node=node
                )
                cutting_points.append(cut_point)
            
            # 为长竹筒段创建分段切割点
            for segment in segments:
                if segment.length_mm > self.config.segment_max_length:
                    # 计算需要分割的数量
                    num_cuts = int(segment.length_mm / self.config.segment_max_length)
                    
                    for j in range(1, num_cuts + 1):
                        # 计算切割位置
                        ratio = j / (num_cuts + 1)
                        cut_x = segment.start_pos.x + ratio * (segment.end_pos.x - segment.start_pos.x)
                        cut_y = segment.start_pos.y + ratio * (segment.end_pos.y - segment.start_pos.y)
                        
                        cut_pos = Point2D(cut_x, cut_y)
                        cut_pos_mm = self.pixel_to_mm_conversion(cut_pos)
                        
                        cut_point = CuttingPoint(
                            position=cut_pos,
                            position_mm=cut_pos_mm,
                            cutting_type=CuttingType.SEGMENT_CUT,
                            priority=6,  # 中优先级
                            confidence=0.8,
                            reason=f"分割长竹筒段 ({segment.length_mm:.1f}mm)",
                            related_segment=segment
                        )
                        cutting_points.append(cut_point)
            
            # 为低质量竹筒段创建质量切割点
            for segment in segments:
                if segment.quality in [SegmentQuality.POOR, SegmentQuality.UNUSABLE]:
                    # 在段的中点创建切割点
                    mid_x = (segment.start_pos.x + segment.end_pos.x) / 2
                    mid_y = (segment.start_pos.y + segment.end_pos.y) / 2
                    
                    cut_pos = Point2D(mid_x, mid_y)
                    cut_pos_mm = self.pixel_to_mm_conversion(cut_pos)
                    
                    cut_point = CuttingPoint(
                        position=cut_pos,
                        position_mm=cut_pos_mm,
                        cutting_type=CuttingType.QUALITY_CUT,
                        priority=4,  # 低优先级
                        confidence=segment.quality_score,
                        reason=f"移除低质量段 ({segment.quality.value})",
                        related_segment=segment
                    )
                    cutting_points.append(cut_point)
            
            logger.debug(f"计算出{len(cutting_points)}个切割点")
            
        except Exception as e:
            logger.error(f"切割点计算失败: {e}")
        
        return cutting_points
    
    def cleanup(self):
        """清理资源"""
        self.edge_image = None
        self.contours = None
        self.bamboo_centerline = None
        logger.info("传统检测器资源清理完成")


# 注册函数，在需要时调用
def register_traditional_detector():
    """注册传统检测器到工厂"""
    from .vision_processor import VisionProcessorFactory
    VisionProcessorFactory.register_processor("traditional", TraditionalDetector)