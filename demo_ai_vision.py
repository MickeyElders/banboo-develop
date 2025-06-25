#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能切竹机 - AI视觉识别算法演示程序
展示传统计算机视觉和深度学习算法的检测效果
"""

import cv2
import numpy as np
import time
import logging
from pathlib import Path
from scipy.signal import find_peaks

from src.vision.bamboo_detector import BambooDetector
from src.vision.vision_types import create_default_config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)


def create_demo_bamboo_image() -> np.ndarray:
    """
    创建演示用的竹子图像（改进版，更明显的竹节特征）
    Returns:
        演示图像
    """
    # 创建图像 (800x600)
    image = np.ones((600, 800, 3), dtype=np.uint8) * 240
    
    # 绘制竹子主体 - 水平放置，更像真实情况
    bamboo_color = (180, 220, 140)  # 浅绿色
    cv2.rectangle(image, (50, 200), (750, 400), bamboo_color, -1)
    
    # 添加明显的竹节 - 使用更强烈的对比
    node_color = (80, 120, 60)  # 更深的绿色
    node_positions = [120, 220, 320, 420, 520, 620, 720]  # 每100像素一个节点
    
    for x_pos in node_positions:
        if x_pos < 750:
            # 竹节主体 - 更宽更明显
            cv2.rectangle(image, (x_pos - 20, 190), (x_pos + 20, 410), node_color, -1)
            
            # 竹节的突出部分（上下边缘）
            cv2.rectangle(image, (x_pos - 25, 185), (x_pos + 25, 195), (60, 90, 40), -1)
            cv2.rectangle(image, (x_pos - 25, 405), (x_pos + 25, 415), (60, 90, 40), -1)
            
            # 添加更明显的纹理线条
            for y in range(200, 400, 10):
                cv2.line(image, (x_pos - 18, y), (x_pos + 18, y), (40, 70, 30), 2)
    
    # 在竹筒段添加纵向纹理（模拟竹子的纹理）
    for x in range(60, 740, 15):
        # 避开节点区域
        is_node_area = any(abs(x - pos) < 25 for pos in node_positions)
        if not is_node_area:
            for y in range(205, 395, 5):
                cv2.line(image, (x, y), (x + 8, y), (160, 200, 120), 1)
    
    # 添加边缘线条增强对比
    cv2.line(image, (50, 200), (750, 200), (120, 160, 80), 3)  # 上边缘
    cv2.line(image, (50, 400), (750, 400), (120, 160, 80), 3)  # 下边缘
    
    # 轻微模糊以模拟真实图像
    image = cv2.GaussianBlur(image, (3, 3), 0)
    
    return image


def debug_image_features(image: np.ndarray) -> None:
    """
    调试图像特征，检查预处理步骤
    Args:
        image: 输入图像
    """
    print("\n=== 图像特征调试 ===")
    
    # 基本信息
    print(f"图像尺寸: {image.shape}")
    print(f"图像类型: {image.dtype}")
    print(f"像素值范围: {image.min()} - {image.max()}")
    
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    print(f"灰度图像素值范围: {gray.min()} - {gray.max()}")
    
    # 边缘检测测试
    edges = cv2.Canny(gray, 30, 100)
    edge_pixels = np.sum(edges > 0)
    print(f"边缘像素数: {edge_pixels} ({edge_pixels/edges.size*100:.2f}%)")
    
    # 保存调试图像
    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)
    
    cv2.imwrite("demo_output/debug_gray.jpg", gray)
    cv2.imwrite("demo_output/debug_edges.jpg", edges)
    
    # 简单的竹节候选检测
    print("\n=== 简单竹节检测测试 ===")
    
    # 水平投影 - 统计每列的边缘像素数
    h_projection = np.sum(edges, axis=0)
    
    # 寻找峰值
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(h_projection, height=50, distance=50)
    
    print(f"检测到 {len(peaks)} 个潜在竹节位置")
    for i, peak in enumerate(peaks):
        print(f"  竹节 {i+1}: x={peak}, 强度={h_projection[peak]}")
    
    # 创建可视化图像
    vis_image = image.copy()
    for peak in peaks:
        cv2.line(vis_image, (peak, 0), (peak, vis_image.shape[0]), (0, 0, 255), 2)
        cv2.putText(vis_image, f"Node", (peak-20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    cv2.imwrite("demo_output/debug_simple_detection.jpg", vis_image)
    print("调试图像已保存到 demo_output/ 目录")


def run_detection_demo():
    """运行检测演示"""
    print("智能切竹机 - AI视觉识别算法演示")
    print("=" * 40)
    
    # 创建演示图像
    demo_image = create_demo_bamboo_image()
    
    # 保存原始图像
    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)
    cv2.imwrite("demo_output/demo_original.jpg", demo_image)
    
    # 调试图像特征
    debug_image_features(demo_image)
    
    # 获取可用的检测算法
    detector = BambooDetector()
    available_algorithms = detector.get_available_algorithms()
    
    print(f"\n可用的检测算法: {available_algorithms}")
    
    # 选择最佳算法
    if 'hybrid' in available_algorithms:
        algorithm = 'hybrid'
        print("使用混合检测器（传统CV + YOLOv8）")
    elif 'yolo' in available_algorithms:
        algorithm = 'yolo'
        print("使用YOLOv8检测器")
    else:
        algorithm = 'traditional'
        print("使用传统CV检测器")
    
    # 创建检测器
    detector = BambooDetector(algorithm)
    
    # 设置自定义配置 - 降低阈值使其更敏感
    custom_config = {
        'gaussian_blur_kernel': 3,  # 减少模糊
        'canny_low_threshold': 20,  # 降低边缘检测阈值
        'canny_high_threshold': 80,
        'node_confidence_threshold': 0.1,  # 降低置信度阈值
        'min_contour_area': 10.0,  # 降低最小轮廓面积
        'node_min_width': 3.0,     # 降低最小节点宽度
        'node_max_width': 60.0,    # 增加最大节点宽度
        'segment_min_length': 20.0,
        'segment_max_length': 300.0
    }
    
    detector.config = detector._create_algorithm_config(custom_config)
    
    # 设置标定数据
    calibration_data = {
        'pixel_to_mm_ratio': 0.3,  # 1像素 = 0.3毫米
        'camera_matrix': np.eye(3).tolist(),
        'dist_coeffs': np.zeros(5).tolist(),
        'rotation_angle': 0.0
    }
    detector.set_calibration(calibration_data)
    
    # 初始化检测器
    if not detector.initialize():
        print("检测器初始化失败！")
        return
    
    print("检测器初始化成功，开始检测...")
    
    # 执行检测
    start_time = time.time()
    result = detector.detect_nodes(demo_image)
    end_time = time.time()
    
    # 显示检测结果
    print(f"\n检测完成！耗时: {end_time - start_time:.3f}秒")
    print(f"使用算法: {result.metadata.get('algorithm', algorithm)}")
    print(f"检测到 {result.total_nodes} 个竹节")
    print(f"分割出 {result.total_segments} 个竹筒段")
    print(f"可用竹筒段: {len(result.usable_segments)} 个")
    print(f"总可用长度: {result.total_usable_length:.1f}mm")
    print(f"推荐切割点: {len(result.cutting_points)} 个")
    
    # 显示算法特定信息
    if algorithm == 'hybrid' and 'hybrid_strategy' in result.metadata:
        print(f"混合策略: {result.metadata['hybrid_strategy']}")
        if 'fusion_weights' in result.metadata:
            weights = result.metadata['fusion_weights']
            print(f"融合权重: YOLO={weights.get('yolo', 0):.1f}, 传统={weights.get('traditional', 0):.1f}")
    
    if 'inference_time' in result.metadata:
        print(f"推理时间: {result.metadata['inference_time']:.3f}秒")
    
    # 显示竹节详情
    if result.nodes:
        print("\n竹节详情:")
        for i, node in enumerate(result.nodes):
            node_info = f"  竹节 {i+1}: 位置({node.position.x:.1f}, {node.position.y:.1f}), "
            node_info += f"宽度 {node.width_mm:.1f}mm, 置信度 {node.confidence:.3f}"
            if hasattr(node, 'node_type'):
                node_info += f", 类型 {node.node_type.value}"
            print(node_info)
    else:
        print("\n未检测到竹节 - 检查调试图像查看原因")
    
    # 显示竹筒段详情
    if result.segments:
        print("\n竹筒段详情:")
        for i, segment in enumerate(result.segments):
            print(f"  竹筒段 {i+1}: 长度 {segment.length_mm:.1f}mm, "
                  f"直径 {segment.diameter_mm:.1f}mm, 质量 {segment.quality.value}")
    
    # 显示切割点详情
    if result.cutting_points:
        print("\n切割点详情:")
        high_priority_cuts = result.get_high_priority_cuts(min_priority=7)
        print(f"  高优先级切割点: {len(high_priority_cuts)} 个")
        
        for i, cut_point in enumerate(result.cutting_points[:5]):  # 只显示前5个
            print(f"  切割点 {i+1}: 位置({cut_point.position_mm.x:.1f}, {cut_point.position_mm.y:.1f})mm, "
                  f"类型 {cut_point.cutting_type.value}, 优先级 {cut_point.priority}")
    
    # 生成可视化结果
    vis_image = detector.visualize_result(demo_image, result)
    cv2.imwrite("demo_output/demo_result.jpg", vis_image)
    
    # 生成调试图像
    debug_image = detector.create_debug_image(demo_image, result)
    cv2.imwrite("demo_output/demo_debug.jpg", debug_image)
    
    # 保存检测结果
    detector.save_result(result, "demo_output/demo_result.json")
    
    # 显示性能统计
    stats = detector.get_performance_stats()
    print(f"\n性能统计:")
    print(f"  总处理图像数: {stats.get('total_processed', 0)}")
    print(f"  平均处理时间: {stats.get('avg_processing_time', 0):.3f}秒")
    
    # 清理资源
    detector.cleanup()
    
    print(f"\n演示完成！")
    print(f"结果图像已保存到: demo_output/demo_result.jpg")
    print(f"调试图像已保存到: demo_output/demo_debug.jpg")
    print(f"检测数据已保存到: demo_output/demo_result.json")
    print(f"原始图像: demo_output/demo_original.jpg")
    print(f"灰度图像: demo_output/debug_gray.jpg")
    print(f"边缘图像: demo_output/debug_edges.jpg")
    print(f"简单检测结果: demo_output/debug_simple_detection.jpg")


def interactive_demo():
    """交互式演示"""
    print("\n" + "=" * 40)
    print("交互式演示模式")
    print("=" * 40)
    
    while True:
        print("\n请选择演示选项:")
        print("1. 运行标准检测演示")
        print("2. 测试不同算法参数")
        print("3. 查看算法性能对比")
        print("4. 测试混合检测策略")
        print("5. 退出")
        
        choice = input("\n请输入选项 (1-5): ").strip()
        
        if choice == '1':
            run_detection_demo()
        
        elif choice == '2':
            test_different_params()
        
        elif choice == '3':
            algorithm_comparison()
        
        elif choice == '4':
            test_hybrid_strategies()
        
        elif choice == '5':
            print("感谢使用演示程序！")
            break
        
        else:
            print("无效选项，请重新选择。")


def test_different_params():
    """测试不同参数设置"""
    print("\n测试不同算法参数...")
    
    demo_image = create_demo_bamboo_image()
    
    # 不同的参数配置
    configs = {
        '敏感检测': {
            'node_confidence_threshold': 0.1,
            'canny_low_threshold': 20,
            'canny_high_threshold': 80,
            'min_contour_area': 20.0
        },
        '标准检测': {
            'node_confidence_threshold': 0.3,
            'canny_low_threshold': 50,
            'canny_high_threshold': 150,
            'min_contour_area': 50.0
        },
        '保守检测': {
            'node_confidence_threshold': 0.6,
            'canny_low_threshold': 80,
            'canny_high_threshold': 200,
            'min_contour_area': 100.0
        }
    }
    
    for config_name, config_params in configs.items():
        print(f"\n--- {config_name} ---")
        
        detector = BambooDetector("traditional", config_params)
        
        if detector.initialize():
            result = detector.detect_nodes(demo_image)
            
            print(f"检测到竹节: {result.total_nodes} 个")
            print(f"竹筒段: {result.total_segments} 个")
            print(f"处理时间: {result.processing_time:.3f}秒")
            
            # 保存结果
            vis_image = detector.visualize_result(demo_image, result)
            cv2.imwrite(f"demo_output/param_test_{config_name}.jpg", vis_image)
            
            detector.cleanup()
        else:
            print("初始化失败")


def algorithm_comparison():
    """算法性能对比"""
    print("\n算法性能对比...")
    
    demo_image = create_demo_bamboo_image()
    
    # 获取可用算法
    detector = BambooDetector()
    available_algorithms = detector.get_available_algorithms()
    
    print(f"可用算法: {available_algorithms}")
    
    results = {}
    
    for algorithm in available_algorithms:
        print(f"\n测试算法: {algorithm}")
        
        try:
            detector = BambooDetector(algorithm)
            
            if detector.initialize():
                # 多次检测取平均值
                times = []
                node_counts = []
                
                for i in range(3):
                    result = detector.detect_nodes(demo_image)
                    times.append(result.processing_time)
                    node_counts.append(result.total_nodes)
                
                results[algorithm] = {
                    'avg_time': np.mean(times),
                    'avg_nodes': np.mean(node_counts),
                    'success': True,
                    'algorithm_info': result.metadata.get('algorithm', algorithm)
                }
                
                print(f"  平均处理时间: {np.mean(times):.3f}秒")
                print(f"  平均检测节点: {np.mean(node_counts):.1f}个")
                
                # 保存结果图像
                vis_image = detector.visualize_result(demo_image, result)
                cv2.imwrite(f"demo_output/comparison_{algorithm}.jpg", vis_image)
                
                detector.cleanup()
            else:
                results[algorithm] = {'success': False}
                print(f"  初始化失败")
                
        except Exception as e:
            results[algorithm] = {'success': False, 'error': str(e)}
            print(f"  测试失败: {e}")
    
    # 显示对比结果
    print(f"\n{'算法':<15} {'平均时间(s)':<12} {'平均节点数':<12} {'状态'}")
    print("-" * 60)
    
    for algorithm, result in results.items():
        if result.get('success', False):
            print(f"{algorithm:<15} {result['avg_time']:<12.3f} {result['avg_nodes']:<12.1f} 成功")
        else:
            error_msg = result.get('error', '失败')
            print(f"{algorithm:<15} {'N/A':<12} {'N/A':<12} {error_msg}")


def test_hybrid_strategies():
    """测试混合检测策略"""
    print("\n测试混合检测策略...")
    
    demo_image = create_demo_bamboo_image()
    
    # 检查是否支持混合检测
    detector = BambooDetector()
    available_algorithms = detector.get_available_algorithms()
    
    if 'hybrid' not in available_algorithms:
        print("混合检测器不可用，请确保已安装YOLOv8依赖")
        return
    
    # 不同的混合策略
    strategies = [
        ('yolo_first', 'YOLO优先策略'),
        ('traditional_first', '传统算法优先策略'),
        ('parallel_fusion', '并行融合策略'),
        ('adaptive', '自适应策略'),
        ('consensus', '共识验证策略')
    ]
    
    results = {}
    
    for strategy_name, strategy_desc in strategies:
        print(f"\n--- {strategy_desc} ---")
        
        try:
            detector = BambooDetector('hybrid')
            
            if detector.initialize():
                # 设置策略
                if hasattr(detector.processor, 'set_strategy'):
                    from src.vision.hybrid_detector import HybridStrategy
                    strategy_enum = getattr(HybridStrategy, strategy_name.upper())
                    detector.processor.set_strategy(strategy_enum)
                
                result = detector.detect_nodes(demo_image)
                
                results[strategy_name] = {
                    'nodes': result.total_nodes,
                    'segments': result.total_segments,
                    'time': result.processing_time,
                    'success': True
                }
                
                print(f"检测到竹节: {result.total_nodes} 个")
                print(f"竹筒段: {result.total_segments} 个")
                print(f"处理时间: {result.processing_time:.3f}秒")
                
                # 显示策略特定信息
                if 'hybrid_strategy' in result.metadata:
                    print(f"实际使用策略: {result.metadata['hybrid_strategy']}")
                
                # 保存结果
                vis_image = detector.visualize_result(demo_image, result)
                cv2.imwrite(f"demo_output/hybrid_{strategy_name}.jpg", vis_image)
                
                detector.cleanup()
            else:
                results[strategy_name] = {'success': False}
                print("初始化失败")
                
        except Exception as e:
            results[strategy_name] = {'success': False, 'error': str(e)}
            print(f"测试失败: {e}")
    
    # 显示策略对比结果
    print(f"\n{'策略':<20} {'节点数':<8} {'竹筒段':<8} {'时间(s)':<10} {'状态'}")
    print("-" * 55)
    
    for strategy_name, result in results.items():
        if result.get('success', False):
            print(f"{strategy_name:<20} {result['nodes']:<8} {result['segments']:<8} {result['time']:<10.3f} 成功")
        else:
            error_msg = result.get('error', '失败')
            print(f"{strategy_name:<20} {'N/A':<8} {'N/A':<8} {'N/A':<10} {error_msg}")


def main():
    """主函数"""
    try:
        print("智能切竹机 - AI视觉识别算法演示程序")
        print("版本: 3.0")
        print("功能: 竹节检测、竹筒分割、切割点计算")
        print("支持: 传统CV、YOLOv8、混合策略")
        
        # 检查输出目录
        output_dir = Path("demo_output")
        output_dir.mkdir(exist_ok=True)
        
        # 运行演示
        run_detection_demo()
        
        # 询问是否进入交互模式
        while True:
            user_input = input("\n是否进入交互式演示？(y/n): ").strip().lower()
            if user_input in ['y', 'yes', '是']:
                interactive_demo()
                break
            elif user_input in ['n', 'no', '否']:
                print("演示结束！")
                break
            else:
                print("请输入 y 或 n")
        
    except KeyboardInterrupt:
        print("\n演示被用户中断")
    except Exception as e:
        logger.error(f"演示程序执行失败: {e}")
        print(f"错误: {e}")


if __name__ == "__main__":
    main() 