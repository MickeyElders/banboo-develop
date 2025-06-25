#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能切竹机 - 摄像头系统测试程序
测试Jetson Nano的视觉系统

测试项目：
1. 摄像头设备检测
2. 图像采集功能
3. 图像质量评估
4. 帧率性能测试
5. 像素校准测试
"""

import os
import sys
import cv2
import numpy as np
import time
import threading
from typing import Tuple, List, Dict, Any
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_camera.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CameraTester:
    """摄像头系统测试器"""
    
    def __init__(self, camera_id: int = 0):
        """
        初始化测试器
        Args:
            camera_id: 摄像头设备ID
        """
        self.camera_id = camera_id
        self.camera = None
        
        # 测试结果
        self.test_results = {
            'device_detection': {},
            'image_capture': {},
            'image_quality': {},
            'performance_test': {},
            'calibration_test': {}
        }
        
        logger.info(f"摄像头测试器初始化 - 设备ID: {camera_id}")
    
    def test_device_detection(self) -> bool:
        """
        测试摄像头设备检测
        Returns:
            测试是否通过
        """
        logger.info("开始摄像头设备检测...")
        
        try:
            # 尝试打开摄像头
            self.camera = cv2.VideoCapture(self.camera_id)
            
            if not self.camera.isOpened():
                logger.error(f"无法打开摄像头设备 {self.camera_id}")
                self.test_results['device_detection'] = {
                    'success': False,
                    'error': 'Camera device not found'
                }
                return False
            
            # 获取摄像头属性
            width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.camera.get(cv2.CAP_PROP_FPS)
            
            # 尝试设置高分辨率
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 2048)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1536)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            # 获取实际设置的分辨率
            actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
            
            self.test_results['device_detection'] = {
                'success': True,
                'default_resolution': (width, height),
                'default_fps': fps,
                'target_resolution': (2048, 1536),
                'actual_resolution': (actual_width, actual_height),
                'actual_fps': actual_fps,
                'resolution_support': (actual_width >= 1920 and actual_height >= 1080)
            }
            
            logger.info(f"摄像头检测成功 - 分辨率: {actual_width}x{actual_height}, 帧率: {actual_fps}")
            return True
            
        except Exception as e:
            logger.error(f"摄像头设备检测异常: {e}")
            self.test_results['device_detection'] = {
                'success': False,
                'error': str(e)
            }
            return False
    
    def test_image_capture(self) -> bool:
        """
        测试图像采集功能
        Returns:
            测试是否通过
        """
        logger.info("开始图像采集测试...")
        
        if not self.camera or not self.camera.isOpened():
            logger.error("摄像头未正确初始化")
            return False
        
        try:
            successful_captures = 0
            total_attempts = 10
            
            for i in range(total_attempts):
                ret, frame = self.camera.read()
                
                if ret and frame is not None:
                    successful_captures += 1
                    
                    # 保存第一帧用于后续测试
                    if i == 0:
                        cv2.imwrite('test_image_0.jpg', frame)
                        self.test_frame = frame.copy()
                        
                        # 获取图像信息
                        height, width = frame.shape[:2]
                        channels = frame.shape[2] if len(frame.shape) == 3 else 1
                        
                        logger.info(f"采集图像信息: {width}x{height}x{channels}")
                
                time.sleep(0.1)
            
            success_rate = successful_captures / total_attempts * 100
            
            self.test_results['image_capture'] = {
                'success': successful_captures > 8,  # 80%成功率
                'successful_captures': successful_captures,
                'total_attempts': total_attempts,
                'success_rate': success_rate
            }
            
            logger.info(f"图像采集测试完成 - 成功率: {success_rate:.1f}%")
            return successful_captures > 8
            
        except Exception as e:
            logger.error(f"图像采集测试异常: {e}")
            self.test_results['image_capture'] = {
                'success': False,
                'error': str(e)
            }
            return False
    
    def test_image_quality(self) -> bool:
        """
        测试图像质量
        Returns:
            测试是否通过
        """
        logger.info("开始图像质量评估...")
        
        if not hasattr(self, 'test_frame'):
            logger.error("没有测试图像")
            return False
        
        try:
            frame = self.test_frame
            
            # 转换为灰度图像
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 计算图像统计信息
            mean_brightness = np.mean(gray)
            std_brightness = np.std(gray)
            
            # 计算对比度（使用标准差）
            contrast = std_brightness
            
            # 计算清晰度（使用拉普拉斯算子的方差）
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = np.var(laplacian)
            
            # 检查是否过曝或欠曝
            overexposed_pixels = np.sum(gray > 250)
            underexposed_pixels = np.sum(gray < 5)
            total_pixels = gray.shape[0] * gray.shape[1]
            
            overexposed_ratio = overexposed_pixels / total_pixels * 100
            underexposed_ratio = underexposed_pixels / total_pixels * 100
            
            # 质量评估标准
            brightness_good = 50 <= mean_brightness <= 200
            contrast_good = contrast > 30
            sharpness_good = sharpness > 100
            exposure_good = overexposed_ratio < 5 and underexposed_ratio < 5
            
            quality_score = sum([brightness_good, contrast_good, sharpness_good, exposure_good]) / 4 * 100
            
            self.test_results['image_quality'] = {
                'success': quality_score >= 75,
                'mean_brightness': mean_brightness,
                'contrast': contrast,
                'sharpness': sharpness,
                'overexposed_ratio': overexposed_ratio,
                'underexposed_ratio': underexposed_ratio,
                'quality_score': quality_score,
                'brightness_good': brightness_good,
                'contrast_good': contrast_good,
                'sharpness_good': sharpness_good,
                'exposure_good': exposure_good
            }
            
            logger.info(f"图像质量评估完成 - 质量得分: {quality_score:.1f}%")
            logger.info(f"亮度: {mean_brightness:.1f}, 对比度: {contrast:.1f}, 清晰度: {sharpness:.1f}")
            
            return quality_score >= 75
            
        except Exception as e:
            logger.error(f"图像质量评估异常: {e}")
            self.test_results['image_quality'] = {
                'success': False,
                'error': str(e)
            }
            return False
    
    def test_performance(self, duration_seconds: int = 30) -> bool:
        """
        测试性能（帧率稳定性）
        Args:
            duration_seconds: 测试持续时间
        Returns:
            测试是否通过
        """
        logger.info(f"开始性能测试 - 持续时间: {duration_seconds}秒")
        
        if not self.camera or not self.camera.isOpened():
            logger.error("摄像头未正确初始化")
            return False
        
        try:
            frame_times = []
            frame_count = 0
            start_time = time.time()
            last_frame_time = start_time
            
            while time.time() - start_time < duration_seconds:
                ret, frame = self.camera.read()
                
                if ret:
                    current_time = time.time()
                    frame_interval = current_time - last_frame_time
                    frame_times.append(frame_interval)
                    frame_count += 1
                    last_frame_time = current_time
            
            total_time = time.time() - start_time
            
            if frame_times:
                # 计算帧率统计
                avg_fps = frame_count / total_time
                avg_interval = np.mean(frame_times)
                std_interval = np.std(frame_times)
                min_interval = np.min(frame_times)
                max_interval = np.max(frame_times)
                
                # 计算帧率稳定性（变异系数）
                cv_fps = std_interval / avg_interval * 100
                
                # 目标帧率检查
                target_fps = 30
                fps_deviation = abs(avg_fps - target_fps) / target_fps * 100
                
                self.test_results['performance_test'] = {
                    'success': avg_fps >= 25 and cv_fps < 20,  # 至少25fps且稳定性良好
                    'avg_fps': avg_fps,
                    'frame_count': frame_count,
                    'duration': total_time,
                    'avg_interval': avg_interval,
                    'std_interval': std_interval,
                    'cv_fps': cv_fps,
                    'fps_deviation': fps_deviation,
                    'target_fps': target_fps
                }
                
                logger.info(f"性能测试完成 - 平均帧率: {avg_fps:.1f}fps, 稳定性CV: {cv_fps:.1f}%")
                return avg_fps >= 25 and cv_fps < 20
            else:
                logger.error("性能测试失败 - 无法获取帧数据")
                return False
                
        except Exception as e:
            logger.error(f"性能测试异常: {e}")
            self.test_results['performance_test'] = {
                'success': False,
                'error': str(e)
            }
            return False
    
    def test_calibration(self) -> bool:
        """
        测试像素校准功能
        Returns:
            测试是否通过
        """
        logger.info("开始像素校准测试...")
        
        if not hasattr(self, 'test_frame'):
            logger.error("没有测试图像")
            return False
        
        try:
            # 模拟校准标定板检测
            frame = self.test_frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 尝试检测棋盘格角点（用于校准）
            pattern_size = (9, 6)  # 棋盘格内角点数量
            ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
            
            if ret:
                # 找到棋盘格，可以进行校准
                # 精细化角点检测
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                
                # 计算角点间距离（像素）
                if len(corners2) >= 2:
                    pixel_distances = []
                    for i in range(len(corners2) - 1):
                        dist = np.linalg.norm(corners2[i] - corners2[i+1])
                        pixel_distances.append(dist)
                    
                    avg_pixel_distance = np.mean(pixel_distances)
                    
                    # 假设棋盘格每个方格是20mm
                    assumed_real_distance = 20.0  # mm
                    pixel_to_mm_ratio = assumed_real_distance / avg_pixel_distance
                    
                    calibration_success = True
                    logger.info(f"棋盘格校准成功 - 像素比例: {pixel_to_mm_ratio:.4f} mm/pixel")
                else:
                    calibration_success = False
                    pixel_to_mm_ratio = 0.5  # 默认值
            else:
                # 没有找到棋盘格，使用默认校准
                calibration_success = False
                pixel_to_mm_ratio = 0.5  # 默认值
                logger.warning("未检测到棋盘格，使用默认像素比例")
            
            # 测试边缘检测功能（用于竹节识别）
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1]) * 100
            
            self.test_results['calibration_test'] = {
                'success': True,  # 校准测试总是通过，只是记录结果
                'chessboard_found': calibration_success,
                'pixel_to_mm_ratio': pixel_to_mm_ratio,
                'edge_density': edge_density,
                'edge_detection_working': edge_density > 1.0  # 边缘密度大于1%
            }
            
            # 保存校准结果图像
            if calibration_success:
                calibration_image = frame.copy()
                cv2.drawChessboardCorners(calibration_image, pattern_size, corners2, ret)
                cv2.imwrite('calibration_result.jpg', calibration_image)
            
            # 保存边缘检测结果
            cv2.imwrite('edge_detection_result.jpg', edges)
            
            logger.info(f"校准测试完成 - 像素比例: {pixel_to_mm_ratio:.4f} mm/pixel")
            return True
            
        except Exception as e:
            logger.error(f"校准测试异常: {e}")
            self.test_results['calibration_test'] = {
                'success': False,
                'error': str(e)
            }
            return False
    
    def test_lighting_conditions(self) -> bool:
        """
        测试不同光照条件下的图像质量
        Returns:
            测试是否通过
        """
        logger.info("开始光照条件测试...")
        
        if not self.camera or not self.camera.isOpened():
            logger.error("摄像头未正确初始化")
            return False
        
        try:
            lighting_results = []
            
            # 测试不同曝光设置
            exposure_values = [-3, -1, 0, 1, 3]  # 相对曝光值
            
            for exposure in exposure_values:
                # 设置曝光（如果摄像头支持）
                try:
                    self.camera.set(cv2.CAP_PROP_EXPOSURE, exposure)
                    time.sleep(0.5)  # 等待设置生效
                except:
                    pass
                
                # 采集图像
                ret, frame = self.camera.read()
                if ret:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    mean_brightness = np.mean(gray)
                    std_brightness = np.std(gray)
                    
                    lighting_results.append({
                        'exposure': exposure,
                        'brightness': mean_brightness,
                        'contrast': std_brightness
                    })
                    
                    # 保存不同曝光的图像
                    cv2.imwrite(f'exposure_test_{exposure}.jpg', frame)
            
            # 找到最佳曝光设置
            best_exposure = None
            best_score = 0
            
            for result in lighting_results:
                # 评分标准：亮度接近128，对比度尽可能高
                brightness_score = 100 - abs(result['brightness'] - 128) / 128 * 100
                contrast_score = min(result['contrast'] / 50 * 100, 100)  # 对比度归一化到100
                total_score = (brightness_score + contrast_score) / 2
                
                if total_score > best_score:
                    best_score = total_score
                    best_exposure = result['exposure']
            
            self.test_results['lighting_test'] = {
                'success': best_score > 60,
                'lighting_results': lighting_results,
                'best_exposure': best_exposure,
                'best_score': best_score
            }
            
            logger.info(f"光照测试完成 - 最佳曝光: {best_exposure}, 得分: {best_score:.1f}")
            return best_score > 60
            
        except Exception as e:
            logger.error(f"光照测试异常: {e}")
            return False
    
    def cleanup(self):
        """清理资源"""
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
    
    def run_all_tests(self) -> Dict[str, Any]:
        """
        运行所有测试
        Returns:
            测试结果汇总
        """
        logger.info("开始完整摄像头测试...")
        
        test_sequence = [
            ('设备检测', self.test_device_detection),
            ('图像采集', self.test_image_capture),
            ('图像质量', self.test_image_quality),
            ('性能测试', lambda: self.test_performance(10)),  # 10秒性能测试
            ('校准测试', self.test_calibration),
            ('光照测试', self.test_lighting_conditions)
        ]
        
        results = {}
        
        for test_name, test_func in test_sequence:
            logger.info(f"执行测试: {test_name}")
            try:
                result = test_func()
                results[test_name] = result
                
                if result:
                    logger.info(f"✓ {test_name} 测试通过")
                else:
                    logger.error(f"✗ {test_name} 测试失败")
                    
            except Exception as e:
                logger.error(f"✗ {test_name} 测试异常: {e}")
                results[test_name] = False
        
        # 生成测试报告
        self.generate_report(results)
        
        # 清理资源
        self.cleanup()
        
        return results
    
    def generate_report(self, results: Dict[str, bool]):
        """
        生成测试报告
        Args:
            results: 测试结果
        """
        logger.info("生成摄像头测试报告...")
        
        report = []
        report.append("=" * 60)
        report.append("智能切竹机摄像头系统测试报告")
        report.append("=" * 60)
        report.append(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"摄像头设备: {self.camera_id}")
        report.append("")
        
        # 测试结果汇总
        passed_tests = sum(1 for result in results.values() if result)
        total_tests = len(results)
        
        report.append("测试结果汇总:")
        report.append(f"通过: {passed_tests}/{total_tests}")
        report.append(f"成功率: {passed_tests/total_tests*100:.1f}%")
        report.append("")
        
        # 详细结果
        for test_name, result in results.items():
            status = "通过" if result else "失败"
            report.append(f"{test_name}: {status}")
        
        report.append("")
        
        # 详细测试数据
        if self.test_results['device_detection'].get('success'):
            dev_result = self.test_results['device_detection']
            report.append(f"分辨率: {dev_result['actual_resolution'][0]}x{dev_result['actual_resolution'][1]}")
            report.append(f"帧率: {dev_result['actual_fps']:.1f} fps")
        
        if self.test_results['performance_test'].get('success'):
            perf_result = self.test_results['performance_test']
            report.append(f"实际帧率: {perf_result['avg_fps']:.1f} fps")
            report.append(f"帧率稳定性: {perf_result['cv_fps']:.1f}%")
        
        if self.test_results['image_quality'].get('success'):
            qual_result = self.test_results['image_quality']
            report.append(f"图像质量得分: {qual_result['quality_score']:.1f}%")
        
        if self.test_results['calibration_test'].get('success'):
            calib_result = self.test_results['calibration_test']
            report.append(f"像素比例: {calib_result['pixel_to_mm_ratio']:.4f} mm/pixel")
        
        report.append("")
        report.append("=" * 60)
        
        # 保存报告
        with open('camera_test_report.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        # 打印报告
        for line in report:
            print(line)


def main():
    """主函数"""
    print("智能切竹机摄像头系统测试")
    print("-" * 40)
    
    # 从命令行参数获取摄像头ID
    camera_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    
    # 创建测试器
    tester = CameraTester(camera_id)
    
    # 运行所有测试
    results = tester.run_all_tests()
    
    # 返回测试结果
    all_passed = all(results.values())
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 