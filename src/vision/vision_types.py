#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能切竹机 - 视觉系统数据结构定义
定义竹节检测、竹筒分割等核心数据类型
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
import numpy as np
import cv2


class NodeType(Enum):
    """竹节类型枚举"""
    UNKNOWN = "unknown"          # 未知类型
    NATURAL_NODE = "natural"     # 天然竹节
    DAMAGE_NODE = "damage"       # 损伤节点
    JUNCTION = "junction"        # 连接点


class SegmentQuality(Enum):
    """竹筒质量等级"""
    EXCELLENT = "excellent"      # 优质
    GOOD = "good"               # 良好
    ACCEPTABLE = "acceptable"    # 可接受
    POOR = "poor"               # 较差
    UNUSABLE = "unusable"       # 不可用


class CuttingType(Enum):
    """切割类型"""
    REMOVE_NODE = "remove_node"  # 移除竹节
    SEGMENT_CUT = "segment_cut"  # 分段切割
    QUALITY_CUT = "quality_cut"  # 质量切割


@dataclass
class Point2D:
    """二维点坐标"""
    x: float
    y: float
    
    def to_tuple(self) -> Tuple[float, float]:
        """转换为元组格式"""
        return (self.x, self.y)
    
    def to_int_tuple(self) -> Tuple[int, int]:
        """转换为整数元组格式"""
        return (int(self.x), int(self.y))
    
    def distance_to(self, other: 'Point2D') -> float:
        """计算到另一点的距离"""
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


@dataclass
class BoundingBox:
    """边界框"""
    x1: float
    y1: float
    x2: float
    y2: float
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        return self.y2 - self.y1
    
    @property
    def center(self) -> Point2D:
        return Point2D((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    def to_cv2_rect(self) -> Tuple[int, int, int, int]:
        """转换为OpenCV矩形格式 (x, y, w, h)"""
        return (int(self.x1), int(self.y1), int(self.width), int(self.height))


@dataclass
class BambooNode:
    """竹节信息"""
    position: Point2D           # 节点中心位置 (像素坐标)
    position_mm: Point2D        # 节点位置 (毫米坐标)
    confidence: float           # 检测置信度 [0-1]
    node_type: NodeType         # 节点类型
    bbox: BoundingBox          # 边界框
    width: float               # 节点宽度 (像素)
    width_mm: float            # 节点宽度 (毫米)
    features: Dict[str, Any]   # 特征信息
    
    def __post_init__(self):
        """后处理：验证数据有效性"""
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"置信度必须在[0,1]范围内: {self.confidence}")
        if self.width <= 0 or self.width_mm <= 0:
            raise ValueError("节点宽度必须为正数")


@dataclass
class BambooSegment:
    """竹筒段信息"""
    start_pos: Point2D          # 起始位置 (像素)
    end_pos: Point2D            # 结束位置 (像素)
    start_pos_mm: Point2D       # 起始位置 (毫米)
    end_pos_mm: Point2D         # 结束位置 (毫米)
    length_mm: float            # 长度 (毫米)
    diameter_mm: float          # 直径 (毫米)
    quality: SegmentQuality     # 质量等级
    quality_score: float        # 质量评分 [0-1]
    defects: List[str]          # 缺陷列表
    bbox: BoundingBox          # 边界框
    
    @property
    def is_usable(self) -> bool:
        """判断是否可用"""
        return self.quality != SegmentQuality.UNUSABLE
    
    @property
    def length_pixels(self) -> float:
        """像素长度"""
        return self.start_pos.distance_to(self.end_pos)


@dataclass
class CuttingPoint:
    """切割点信息"""
    position: Point2D           # 切割位置 (像素)
    position_mm: Point2D        # 切割位置 (毫米)
    cutting_type: CuttingType   # 切割类型
    priority: int               # 优先级 (1-10, 数字越大优先级越高)
    confidence: float           # 置信度 [0-1]
    reason: str                 # 切割原因说明
    related_node: Optional[BambooNode] = None    # 相关竹节
    related_segment: Optional[BambooSegment] = None  # 相关竹筒段
    
    def __post_init__(self):
        """后处理：验证数据"""
        if not 1 <= self.priority <= 10:
            raise ValueError(f"优先级必须在[1,10]范围内: {self.priority}")
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"置信度必须在[0,1]范围内: {self.confidence}")


@dataclass
class ProcessingROI:
    """处理感兴趣区域"""
    bbox: BoundingBox          # 区域范围
    roi_type: str              # 区域类型
    mask: Optional[np.ndarray] = None  # 掩码 (可选)
    
    def extract_from_image(self, image: np.ndarray) -> np.ndarray:
        """从图像中提取ROI"""
        x, y, w, h = self.bbox.to_cv2_rect()
        roi = image[y:y+h, x:x+w]
        
        if self.mask is not None:
            # 应用掩码
            mask_roi = self.mask[y:y+h, x:x+w]
            roi = cv2.bitwise_and(roi, roi, mask=mask_roi)
        
        return roi


@dataclass
class DetectionResult:
    """检测结果汇总"""
    nodes: List[BambooNode]     # 检测到的竹节
    segments: List[BambooSegment]  # 分割的竹筒段
    cutting_points: List[CuttingPoint]  # 推荐切割点
    processing_time: float      # 处理时间 (秒)
    image_shape: Tuple[int, int]  # 原始图像尺寸 (height, width)
    roi: Optional[ProcessingROI] = None  # 处理区域
    metadata: Dict[str, Any] = None  # 元数据
    
    def __post_init__(self):
        """后处理：设置默认值"""
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def total_nodes(self) -> int:
        """竹节总数"""
        return len(self.nodes)
    
    @property
    def total_segments(self) -> int:
        """竹筒段总数"""
        return len(self.segments)
    
    @property
    def usable_segments(self) -> List[BambooSegment]:
        """可用竹筒段"""
        return [seg for seg in self.segments if seg.is_usable]
    
    @property
    def total_usable_length(self) -> float:
        """总可用长度 (毫米)"""
        return sum(seg.length_mm for seg in self.usable_segments)
    
    def get_high_priority_cuts(self, min_priority: int = 7) -> List[CuttingPoint]:
        """获取高优先级切割点"""
        return [cut for cut in self.cutting_points if cut.priority >= min_priority]
    
    def sort_cutting_points_by_position(self) -> List[CuttingPoint]:
        """按位置排序切割点"""
        return sorted(self.cutting_points, key=lambda cp: cp.position.x)


@dataclass
class CalibrationData:
    """标定数据"""
    pixel_to_mm_ratio: float    # 像素到毫米转换比例
    camera_matrix: np.ndarray   # 相机内参矩阵
    dist_coeffs: np.ndarray     # 畸变系数
    rotation_angle: float       # 旋转角度 (度)
    roi_bounds: BoundingBox     # 有效区域边界
    
    def pixel_to_mm(self, pixel_coord: Point2D) -> Point2D:
        """像素坐标转毫米坐标"""
        return Point2D(
            pixel_coord.x * self.pixel_to_mm_ratio,
            pixel_coord.y * self.pixel_to_mm_ratio
        )
    
    def mm_to_pixel(self, mm_coord: Point2D) -> Point2D:
        """毫米坐标转像素坐标"""
        return Point2D(
            mm_coord.x / self.pixel_to_mm_ratio,
            mm_coord.y / self.pixel_to_mm_ratio
        )


@dataclass
class AlgorithmConfig:
    """算法配置参数"""
    # 图像预处理参数
    gaussian_blur_kernel: int = 5
    bilateral_filter_d: int = 9
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0
    
    # 边缘检测参数
    canny_low_threshold: int = 50
    canny_high_threshold: int = 150
    canny_aperture_size: int = 3
    
    # 轮廓过滤参数
    min_contour_area: float = 100.0
    max_contour_area: float = 50000.0
    min_aspect_ratio: float = 0.1
    max_aspect_ratio: float = 10.0
    
    # 竹节检测参数
    node_min_width: float = 10.0    # 像素
    node_max_width: float = 200.0   # 像素
    node_confidence_threshold: float = 0.5
    
    # 竹筒分割参数
    segment_min_length: float = 50.0  # 毫米
    segment_max_length: float = 500.0 # 毫米
    quality_threshold: float = 0.6
    
    def validate(self) -> bool:
        """验证配置参数有效性"""
        try:
            assert self.gaussian_blur_kernel > 0 and self.gaussian_blur_kernel % 2 == 1
            assert 0 < self.canny_low_threshold < self.canny_high_threshold < 255
            assert self.min_contour_area > 0
            assert self.min_aspect_ratio > 0
            assert self.node_min_width > 0
            assert self.segment_min_length > 0
            assert 0 <= self.node_confidence_threshold <= 1
            assert 0 <= self.quality_threshold <= 1
            return True
        except AssertionError:
            return False


# 工具函数
def create_default_config() -> AlgorithmConfig:
    """创建默认算法配置"""
    return AlgorithmConfig()


def draw_detection_result(image: np.ndarray, 
                         result: DetectionResult,
                         show_nodes: bool = True,
                         show_segments: bool = True,
                         show_cutting_points: bool = True) -> np.ndarray:
    """
    在图像上绘制检测结果
    Args:
        image: 输入图像
        result: 检测结果
        show_nodes: 是否显示竹节
        show_segments: 是否显示竹筒段
        show_cutting_points: 是否显示切割点
    Returns:
        绘制了结果的图像
    """
    vis_image = image.copy()
    
    # 绘制竹节
    if show_nodes:
        for i, node in enumerate(result.nodes):
            center = node.position.to_int_tuple()
            bbox = node.bbox
            
            # 绘制边界框
            cv2.rectangle(vis_image, 
                         (int(bbox.x1), int(bbox.y1)),
                         (int(bbox.x2), int(bbox.y2)),
                         (0, 255, 0), 2)
            
            # 绘制中心点
            cv2.circle(vis_image, center, 5, (0, 255, 0), -1)
            
            # 添加标签
            label = f"Node{i+1}: {node.confidence:.2f}"
            cv2.putText(vis_image, label, 
                       (center[0] + 10, center[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # 绘制竹筒段
    if show_segments:
        for i, segment in enumerate(result.segments):
            start = segment.start_pos.to_int_tuple()
            end = segment.end_pos.to_int_tuple()
            
            # 颜色根据质量确定
            color_map = {
                SegmentQuality.EXCELLENT: (0, 255, 0),   # 绿色
                SegmentQuality.GOOD: (0, 200, 255),      # 橙色
                SegmentQuality.ACCEPTABLE: (0, 255, 255), # 黄色
                SegmentQuality.POOR: (0, 100, 255),      # 深橙色
                SegmentQuality.UNUSABLE: (0, 0, 255)     # 红色
            }
            color = color_map.get(segment.quality, (128, 128, 128))
            
            # 绘制线段
            cv2.line(vis_image, start, end, color, 3)
            
            # 添加长度标签
            mid_point = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
            label = f"S{i+1}: {segment.length_mm:.1f}mm"
            cv2.putText(vis_image, label, mid_point,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # 绘制切割点
    if show_cutting_points:
        for i, cut_point in enumerate(result.cutting_points):
            pos = cut_point.position.to_int_tuple()
            
            # 颜色根据优先级确定
            if cut_point.priority >= 8:
                color = (0, 0, 255)      # 红色 - 高优先级
            elif cut_point.priority >= 6:
                color = (0, 165, 255)    # 橙色 - 中优先级
            else:
                color = (0, 255, 255)    # 黄色 - 低优先级
            
            # 绘制切割标记
            cv2.drawMarker(vis_image, pos, color, cv2.MARKER_CROSS, 
                          markerSize=15, thickness=2)
            
            # 添加标签
            label = f"Cut{i+1}: P{cut_point.priority}"
            cv2.putText(vis_image, label, 
                       (pos[0] + 15, pos[1] + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    return vis_image 