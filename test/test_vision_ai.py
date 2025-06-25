#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能切竹机 - AI视觉识别算法测试程序
测试传统计算机视觉和深度学习算法的检测性能
"""

import os
import sys
import cv2
import numpy as np
import time
import logging
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vision.bamboo_detector import BambooDetector
from src.vision.vision_types import create_default_config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VisionTestSuite:
    """AI视觉识别测试套件"""
    
    def __init__(self):
        """初始化测试套件"""
        self.test_images = []
        self.results = {}
        
    def generate_synthetic_bamboo_image(self, width: int = 800, height: int = 600) -> np.ndarray:
        """
        生成合成竹子图像用于测试
        Args:
            width: 图像宽度
            height: 图像高度
        Returns:
            合成的竹子图像
        """
        # 创建白色背景
        image = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # 绘制竹子主体（浅绿色）
        bamboo_color = (180, 220, 140)  # BGR格式
        cv2.rectangle(image, (100, 50), (700, 550), bamboo_color, -1)
        
        # 添加竹节（深绿色条纹）
        node_color = (100, 150, 80)
        node_positions = [150, 250, 350, 450, 550, 650]
        
        for x_pos in node_positions:
            if x_pos < width - 20:
                # 绘制竹节条纹
                cv2.rectangle(image, (x_pos - 10, 50), (x_pos + 10, 550), node_color, -1)
                
                # 添加纹理细节
                for y in range(60, 540, 20):
                    cv2.line(image, (x_pos - 8, y), (x_pos + 8, y), (80, 120, 60), 2)
        
        # 添加一些噪声
        noise = np.random.normal(0, 10, image.shape).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # 添加轻微模糊
        image = cv2.GaussianBlur(image, (3, 3), 0)
        
        return image
    
    def generate_test_images(self, count: int = 5) -> List[np.ndarray]:
        """
        生成多个测试图像
        Args:
            count: 图像数量
        Returns:
            测试图像列表
        """
        images = []
        
        for i in range(count):
            # 生成不同参数的图像
            width = 600 + i * 50
            height = 400 + i * 30
            image = self.generate_synthetic_bamboo_image(width, height)
            images.append(image)
            
            logger.info(f"生成测试图像 {i+1}: {width}x{height}")
        
        return images
    
    def test_traditional_algorithm(self, images: List[np.ndarray]) -> Dict:
        """
        测试传统计算机视觉算法
        Args:
            images: 测试图像列表
        Returns:
            测试结果
        """
        logger.info("开始测试传统计算机视觉算法")
        
        # 创建检测器
        config = {
            'gaussian_blur_kernel': 5,
            'canny_low_threshold': 50,
            'canny_high_threshold': 150,
            'node_confidence_threshold': 0.3,
            'min_contour_area': 50.0,
            'node_min_width': 8.0,
            'node_max_width': 30.0
        }
        
        detector = BambooDetector("traditional", config)
        
        if not detector.initialize():
            logger.error("传统算法初始化失败")
            return {}
        
        results = {
            'algorithm': 'traditional',
            'total_images': len(images),
            'processing_times': [],
            'detection_counts': [],
            'success_rate': 0.0,
            'avg_processing_time': 0.0
        }
        
        successful_tests = 0
        
        for i, image in enumerate(images):
            try:
                # 执行检测
                result = detector.detect_nodes(image)
                
                # 记录结果
                results['processing_times'].append(result.processing_time)
                results['detection_counts'].append(result.total_nodes)
                
                successful_tests += 1
                
                logger.info(f"图像 {i+1}: 检测到 {result.total_nodes} 个竹节, "
                          f"耗时 {result.processing_time:.3f}s")
                
                # 保存可视化结果
                vis_image = detector.visualize_result(image, result)
                cv2.imwrite(f"test_output/traditional_result_{i+1}.jpg", vis_image)
                
            except Exception as e:
                logger.error(f"图像 {i+1} 检测失败: {e}")
        
        # 计算统计信息
        if successful_tests > 0:
            results['success_rate'] = successful_tests / len(images)
            results['avg_processing_time'] = np.mean(results['processing_times'])
        
        detector.cleanup()
        return results
    
    def test_algorithm_performance(self, algorithm: str, images: List[np.ndarray]) -> Dict:
        """
        测试指定算法的性能
        Args:
            algorithm: 算法名称
            images: 测试图像
        Returns:
            性能测试结果
        """
        logger.info(f"开始性能测试: {algorithm}")
        
        detector = BambooDetector(algorithm)
        
        if not detector.initialize():
            logger.error(f"算法 {algorithm} 初始化失败")
            return {}
        
        # 预热
        warmup_image = self.generate_synthetic_bamboo_image(400, 300)
        detector.detect_nodes(warmup_image)
        
        # 性能测试
        processing_times = []
        memory_usage = []
        
        for i in range(len(images)):
            image = images[i]
            
            # 记录内存使用
            import psutil
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # 执行检测
            start_time = time.time()
            result = detector.detect_nodes(image)
            end_time = time.time()
            
            processing_time = end_time - start_time
            processing_times.append(processing_time)
            
            # 记录内存使用
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_usage.append(memory_after - memory_before)
            
            logger.info(f"图像 {i+1}: {processing_time:.3f}s, "
                       f"内存增长: {memory_after - memory_before:.1f}MB")
        
        detector.cleanup()
        
        return {
            'algorithm': algorithm,
            'processing_times': processing_times,
            'memory_usage': memory_usage,
            'avg_processing_time': np.mean(processing_times),
            'max_processing_time': np.max(processing_times),
            'min_processing_time': np.min(processing_times),
            'avg_memory_usage': np.mean(memory_usage)
        }
    
    def test_algorithm_accuracy(self, algorithm: str, images: List[np.ndarray]) -> Dict:
        """
        测试算法准确性
        Args:
            algorithm: 算法名称
            images: 测试图像
        Returns:
            准确性测试结果
        """
        logger.info(f"开始准确性测试: {algorithm}")
        
        detector = BambooDetector(algorithm)
        
        if not detector.initialize():
            logger.error(f"算法 {algorithm} 初始化失败")
            return {}
        
        # 预期结果（基于合成图像的已知竹节数量）
        expected_nodes = [6, 6, 6, 6, 6]  # 每个图像预期6个竹节
        
        accuracy_results = {
            'algorithm': algorithm,
            'total_images': len(images),
            'expected_nodes': expected_nodes,
            'detected_nodes': [],
            'accuracy_scores': [],
            'precision_scores': [],
            'recall_scores': []
        }
        
        for i, image in enumerate(images):
            try:
                result = detector.detect_nodes(image)
                detected_count = result.total_nodes
                expected_count = expected_nodes[i] if i < len(expected_nodes) else 6
                
                accuracy_results['detected_nodes'].append(detected_count)
                
                # 计算准确性指标
                if expected_count > 0:
                    # 简化的准确性计算
                    accuracy = 1.0 - abs(detected_count - expected_count) / expected_count
                    accuracy = max(0.0, accuracy)
                else:
                    accuracy = 1.0 if detected_count == 0 else 0.0
                
                accuracy_results['accuracy_scores'].append(accuracy)
                
                # 简化的精确率和召回率
                precision = min(1.0, detected_count / max(1, expected_count))
                recall = min(1.0, detected_count / max(1, expected_count))
                
                accuracy_results['precision_scores'].append(precision)
                accuracy_results['recall_scores'].append(recall)
                
                logger.info(f"图像 {i+1}: 期望 {expected_count}, 检测到 {detected_count}, "
                          f"准确率 {accuracy:.3f}")
                
            except Exception as e:
                logger.error(f"图像 {i+1} 准确性测试失败: {e}")
                accuracy_results['detected_nodes'].append(0)
                accuracy_results['accuracy_scores'].append(0.0)
                accuracy_results['precision_scores'].append(0.0)
                accuracy_results['recall_scores'].append(0.0)
        
        # 计算平均指标
        accuracy_results['avg_accuracy'] = np.mean(accuracy_results['accuracy_scores'])
        accuracy_results['avg_precision'] = np.mean(accuracy_results['precision_scores'])
        accuracy_results['avg_recall'] = np.mean(accuracy_results['recall_scores'])
        
        detector.cleanup()
        return accuracy_results
    
    def test_edge_cases(self) -> Dict:
        """
        测试边界情况
        Returns:
            边界测试结果
        """
        logger.info("开始边界情况测试")
        
        detector = BambooDetector("traditional")
        
        if not detector.initialize():
            logger.error("边界测试初始化失败")
            return {}
        
        edge_cases = {
            'empty_image': np.zeros((100, 100, 3), dtype=np.uint8),
            'noise_image': np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8),
            'single_color': np.ones((150, 250, 3), dtype=np.uint8) * 128,
            'high_contrast': np.zeros((180, 320, 3), dtype=np.uint8)
        }
        
        # 创建高对比度图像
        edge_cases['high_contrast'][:, :160] = 255
        
        results = {}
        
        for case_name, image in edge_cases.items():
            try:
                result = detector.detect_nodes(image)
                results[case_name] = {
                    'success': True,
                    'nodes_detected': result.total_nodes,
                    'processing_time': result.processing_time
                }
                logger.info(f"边界测试 {case_name}: 检测到 {result.total_nodes} 个竹节")
                
            except Exception as e:
                results[case_name] = {
                    'success': False,
                    'error': str(e)
                }
                logger.error(f"边界测试 {case_name} 失败: {e}")
        
        detector.cleanup()
        return results
    
    def run_comprehensive_test(self):
        """运行综合测试"""
        logger.info("开始AI视觉识别算法综合测试")
        
        # 创建输出目录
        output_dir = Path("test_output")
        output_dir.mkdir(exist_ok=True)
        
        # 生成测试图像
        test_images = self.generate_test_images(5)
        
        # 保存测试图像
        for i, image in enumerate(test_images):
            cv2.imwrite(f"test_output/test_image_{i+1}.jpg", image)
        
        # 获取可用算法
        detector = BambooDetector()
        available_algorithms = detector.get_available_algorithms()
        logger.info(f"可用算法: {available_algorithms}")
        
        # 测试所有算法
        all_results = {}
        
        for algorithm in available_algorithms:
            logger.info(f"\n=== 测试算法: {algorithm} ===")
            
            try:
                # 性能测试
                perf_results = self.test_algorithm_performance(algorithm, test_images)
                
                # 准确性测试
                accuracy_results = self.test_algorithm_accuracy(algorithm, test_images)
                
                all_results[algorithm] = {
                    'performance': perf_results,
                    'accuracy': accuracy_results
                }
                
            except Exception as e:
                logger.error(f"算法 {algorithm} 测试失败: {e}")
                all_results[algorithm] = {'error': str(e)}
        
        # 边界测试
        edge_results = self.test_edge_cases()
        all_results['edge_cases'] = edge_results
        
        # 生成测试报告
        self.generate_test_report(all_results)
        
        logger.info("综合测试完成")
        return all_results
    
    def generate_test_report(self, results: Dict):
        """
        生成测试报告
        Args:
            results: 测试结果
        """
        report_path = "test_output/ai_vision_test_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("智能切竹机 - AI视觉识别算法测试报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 算法性能比较
            f.write("算法性能比较:\n")
            f.write("-" * 30 + "\n")
            
            for algorithm, result in results.items():
                if algorithm == 'edge_cases':
                    continue
                    
                if 'error' in result:
                    f.write(f"{algorithm}: 测试失败 - {result['error']}\n")
                    continue
                
                perf = result.get('performance', {})
                accuracy = result.get('accuracy', {})
                
                f.write(f"\n{algorithm.upper()}:\n")
                f.write(f"  平均处理时间: {perf.get('avg_processing_time', 0):.3f}s\n")
                f.write(f"  最大处理时间: {perf.get('max_processing_time', 0):.3f}s\n")
                f.write(f"  最小处理时间: {perf.get('min_processing_time', 0):.3f}s\n")
                f.write(f"  平均内存使用: {perf.get('avg_memory_usage', 0):.1f}MB\n")
                f.write(f"  平均准确率: {accuracy.get('avg_accuracy', 0):.3f}\n")
                f.write(f"  平均精确率: {accuracy.get('avg_precision', 0):.3f}\n")
                f.write(f"  平均召回率: {accuracy.get('avg_recall', 0):.3f}\n")
            
            # 边界测试结果
            if 'edge_cases' in results:
                f.write(f"\n边界测试结果:\n")
                f.write("-" * 30 + "\n")
                
                for case_name, case_result in results['edge_cases'].items():
                    f.write(f"{case_name}: ")
                    if case_result.get('success', False):
                        f.write(f"成功 (检测到 {case_result['nodes_detected']} 个竹节)\n")
                    else:
                        f.write(f"失败 - {case_result.get('error', '未知错误')}\n")
            
            f.write(f"\n测试报告已保存到: {report_path}\n")
        
        logger.info(f"测试报告已保存到: {report_path}")


def main():
    """主函数"""
    try:
        # 创建测试套件
        test_suite = VisionTestSuite()
        
        # 运行综合测试
        results = test_suite.run_comprehensive_test()
        
        print("\n=== 测试总结 ===")
        
        for algorithm, result in results.items():
            if algorithm == 'edge_cases':
                continue
                
            if 'error' in result:
                print(f"{algorithm}: 测试失败")
                continue
            
            perf = result.get('performance', {})
            accuracy = result.get('accuracy', {})
            
            print(f"\n{algorithm.upper()}:")
            print(f"  平均处理时间: {perf.get('avg_processing_time', 0):.3f}s")
            print(f"  平均准确率: {accuracy.get('avg_accuracy', 0):.3f}")
        
        print(f"\n详细报告请查看: test_output/ai_vision_test_report.txt")
        
    except Exception as e:
        logger.error(f"测试程序执行失败: {e}")


if __name__ == "__main__":
    main() 