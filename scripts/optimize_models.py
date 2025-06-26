#!/usr/bin/env python3
"""
模型优化工具
自动优化和导出YOLO模型，进行性能基准测试
"""

import os
import sys
import argparse
import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import shutil

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.vision.yolo_detector import OptimizedYOLODetector, ModelFormat, InferenceMode
from src.vision.hybrid_detector import OptimizedHybridDetector, HybridStrategy
from src.vision.vision_types import AlgorithmConfig, CalibrationData

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('model_optimization.log')
    ]
)
logger = logging.getLogger(__name__)


class ModelOptimizer:
    """模型优化器"""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.output_dir = Path("models/optimized")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建测试图像
        self.test_images = self._create_test_images()
        
        # 配置
        self.config = AlgorithmConfig()
        self.calibration = CalibrationData()
        
    def _create_test_images(self) -> List[np.ndarray]:
        """创建测试图像"""
        test_images = []
        
        # 创建不同尺寸和内容的测试图像
        sizes = [(640, 640), (1280, 720), (1920, 1080)]
        
        for i, (w, h) in enumerate(sizes):
            # 创建模拟竹子图像
            img = np.random.randint(50, 200, (h, w, 3), dtype=np.uint8)
            
            # 添加一些模拟的竹节特征
            for j in range(3 + i):
                center_x = np.random.randint(w//4, 3*w//4)
                center_y = np.random.randint(h//4, 3*h//4)
                radius = np.random.randint(20, 50)
                
                cv2.circle(img, (center_x, center_y), radius, (100, 150, 100), -1)
                cv2.circle(img, (center_x, center_y), radius+5, (80, 120, 80), 2)
            
            test_images.append(img)
        
        logger.info(f"创建了{len(test_images)}张测试图像")
        return test_images
    
    def export_all_formats(self, model_path: str = None) -> Dict[str, str]:
        """导出所有支持的模型格式"""
        if model_path:
            self.model_path = model_path
        
        if not self.model_path:
            raise ValueError("未指定模型路径")
        
        logger.info(f"开始导出模型: {self.model_path}")
        
        try:
            # 创建YOLO检测器
            detector = OptimizedYOLODetector(self.config, self.calibration)
            detector.model_path = self.model_path
            detector.model = YOLO(self.model_path)
            
            # 要导出的格式列表
            export_formats = [
                ('onnx', '导出ONNX格式（跨平台推理）'),
                ('openvino', '导出OpenVINO格式（Intel硬件优化）'),
                ('ncnn', '导出NCNN格式（移动端/ARM优化）'),
                ('tflite', '导出TensorFlow Lite格式（移动端）'),
            ]
            
            # 如果有GPU，添加TensorRT
            if torch.cuda.is_available():
                export_formats.insert(1, ('engine', '导出TensorRT格式（NVIDIA GPU优化）'))
            
            exported_models = {}
            
            for format_name, description in export_formats:
                try:
                    logger.info(f"导出{description}...")
                    
                    # 设置导出参数
                    export_kwargs = {
                        'imgsz': 640,
                        'optimize': True,
                        'simplify': True,
                    }
                    
                    if format_name == 'engine':
                        export_kwargs.update({
                            'half': True,
                            'workspace': 4,  # GB
                            'batch': detector.optimal_batch_size if hasattr(detector, 'optimal_batch_size') else 1
                        })
                    elif format_name == 'onnx':
                        export_kwargs.update({
                            'dynamic': False,
                            'opset': 16
                        })
                    elif format_name == 'tflite':
                        export_kwargs.update({
                            'int8': True,  # 量化
                        })
                    
                    exported_path = detector.export_optimized_model(format_name, **export_kwargs)
                    exported_models[format_name] = exported_path
                    
                    logger.info(f"✓ {format_name.upper()}导出成功: {exported_path}")
                    
                except Exception as e:
                    logger.error(f"✗ {format_name.upper()}导出失败: {e}")
                    exported_models[format_name] = f"ERROR: {e}"
            
            # 保存导出报告
            self._save_export_report(exported_models)
            
            return exported_models
            
        except Exception as e:
            logger.error(f"模型导出过程失败: {e}")
            raise
    
    def benchmark_models(self, model_paths: List[str] = None, num_runs: int = 10) -> Dict[str, Any]:
        """对比不同模型格式的性能"""
        if model_paths is None:
            # 自动找到可用的模型
            model_paths = self._find_available_models()
        
        if not model_paths:
            raise ValueError("未找到可用的模型文件")
        
        logger.info(f"开始性能基准测试，{len(model_paths)}个模型，{num_runs}次运行")
        
        benchmark_results = {}
        
        for model_path in model_paths:
            try:
                logger.info(f"测试模型: {model_path}")
                
                # 创建检测器
                detector = OptimizedYOLODetector(self.config, self.calibration)
                detector.model_path = str(model_path)
                
                if not detector.initialize():
                    logger.warning(f"模型初始化失败: {model_path}")
                    continue
                
                # 运行基准测试
                model_results = detector.benchmark_performance(self.test_images, num_runs)
                
                # 添加模型信息
                model_results['model_path'] = str(model_path)
                model_results['model_size_mb'] = self._get_model_size(model_path)
                
                # 使用模型名称作为键
                model_name = Path(model_path).stem
                benchmark_results[model_name] = model_results
                
                detector.cleanup()
                
            except Exception as e:
                logger.error(f"模型{model_path}基准测试失败: {e}")
                benchmark_results[Path(model_path).stem] = {'error': str(e)}
        
        # 保存基准测试报告
        self._save_benchmark_report(benchmark_results)
        
        return benchmark_results
    
    def optimize_for_device(self, target_device: str, target_fps: float = 10.0) -> Dict[str, Any]:
        """针对特定设备优化模型"""
        logger.info(f"针对{target_device}设备优化，目标{target_fps}FPS")
        
        optimization_results = {
            'target_device': target_device,
            'target_fps': target_fps,
            'recommendations': [],
            'exported_models': {},
            'configuration': {}
        }
        
        try:
            # 创建混合检测器进行优化
            hybrid_detector = OptimizedHybridDetector(self.config, self.calibration)
            
            if not hybrid_detector.initialize():
                raise RuntimeError("混合检测器初始化失败")
            
            # 执行设备优化
            deployment_config = hybrid_detector.optimize_for_deployment(target_device, target_fps)
            optimization_results['configuration'] = deployment_config
            
            # 根据设备推荐模型格式
            if target_device.lower() in ['cpu', 'intel']:
                recommended_formats = ['openvino', 'onnx', 'ncnn']
                optimization_results['recommendations'].extend([
                    '推荐使用OpenVINO格式获得最佳CPU性能',
                    '考虑使用量化模型减少内存占用',
                    '启用多线程推理'
                ])
            elif target_device.lower() in ['gpu', 'cuda', 'nvidia']:
                recommended_formats = ['engine', 'onnx']
                optimization_results['recommendations'].extend([
                    '推荐使用TensorRT引擎获得最佳GPU性能',
                    '启用FP16精度加速推理',
                    '使用批处理提高吞吐量'
                ])
            elif target_device.lower() in ['mobile', 'arm', 'android', 'ios']:
                recommended_formats = ['tflite', 'ncnn']
                optimization_results['recommendations'].extend([
                    '推荐使用TensorFlow Lite格式',
                    '启用INT8量化减少模型大小',
                    '优化模型架构以适应移动设备'
                ])
            else:
                recommended_formats = ['onnx', 'openvino']
                optimization_results['recommendations'].append('使用通用ONNX格式确保兼容性')
            
            # 导出推荐格式
            if self.model_path:
                try:
                    detector = OptimizedYOLODetector(self.config, self.calibration)
                    detector.model_path = self.model_path
                    detector.model = YOLO(self.model_path)
                    
                    for format_name in recommended_formats:
                        try:
                            exported_path = detector.export_optimized_model(format_name)
                            optimization_results['exported_models'][format_name] = exported_path
                        except Exception as e:
                            optimization_results['exported_models'][format_name] = f"ERROR: {e}"
                except Exception as e:
                    logger.warning(f"模型导出失败: {e}")
            
            # 性能测试
            test_results = hybrid_detector.benchmark_performance(
                self.test_images[:2],  # 使用较少图像快速测试
                strategies=[HybridStrategy.PERFORMANCE_OPTIMIZED],
                num_runs=3
            )
            optimization_results['performance_test'] = test_results
            
            hybrid_detector.cleanup()
            
            # 保存优化报告
            self._save_optimization_report(optimization_results)
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"设备优化失败: {e}")
            optimization_results['error'] = str(e)
            return optimization_results
    
    def auto_optimize(self) -> Dict[str, Any]:
        """自动检测环境并优化"""
        logger.info("开始自动优化...")
        
        auto_results = {
            'system_info': self._get_system_info(),
            'optimization_results': {},
            'recommendations': []
        }
        
        # 检测系统环境
        system_info = auto_results['system_info']
        
        # 根据系统自动选择优化策略
        if system_info['has_cuda']:
            target_device = 'gpu'
            target_fps = 30.0 if system_info['gpu_memory_gb'] >= 4 else 15.0
        elif 'Intel' in system_info.get('cpu_info', ''):
            target_device = 'intel'
            target_fps = 15.0
        else:
            target_device = 'cpu'
            target_fps = 10.0
        
        # 执行优化
        optimization_results = self.optimize_for_device(target_device, target_fps)
        auto_results['optimization_results'] = optimization_results
        
        # 生成总体建议
        auto_results['recommendations'] = [
            f"检测到{system_info['device_type']}环境，推荐{target_device}优化",
            f"目标帧率设置为{target_fps}FPS",
            "建议定期更新模型以获得最佳性能",
            "在生产环境中使用量化模型以减少资源占用"
        ]
        
        return auto_results
    
    def _find_available_models(self) -> List[Path]:
        """查找可用的模型文件"""
        model_paths = []
        
        # 搜索目录
        search_dirs = [
            Path("models"),
            Path("runs/detect/train/weights"),
            self.output_dir
        ]
        
        # 支持的文件扩展名
        extensions = ['.pt', '.onnx', '.engine', '*.tflite']
        
        for search_dir in search_dirs:
            if search_dir.exists():
                for ext in extensions:
                    model_paths.extend(search_dir.glob(f"*{ext}"))
                
                # 查找OpenVINO模型目录
                for openvino_dir in search_dir.glob("*_openvino_model"):
                    if openvino_dir.is_dir():
                        model_paths.append(openvino_dir)
                
                # 查找NCNN模型目录
                for ncnn_dir in search_dir.glob("*_ncnn_model"):
                    if ncnn_dir.is_dir():
                        model_paths.append(ncnn_dir)
        
        return list(set(model_paths))  # 去重
    
    def _get_model_size(self, model_path: Path) -> float:
        """获取模型大小(MB)"""
        try:
            if model_path.is_file():
                return model_path.stat().st_size / (1024 * 1024)
            elif model_path.is_dir():
                total_size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
                return total_size / (1024 * 1024)
            else:
                return 0.0
        except Exception:
            return 0.0
    
    def _get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        system_info = {
            'python_version': sys.version,
            'pytorch_version': torch.__version__,
            'has_cuda': torch.cuda.is_available(),
            'device_type': 'GPU' if torch.cuda.is_available() else 'CPU'
        }
        
        if torch.cuda.is_available():
            system_info.update({
                'cuda_version': torch.version.cuda,
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3)
            })
        
        try:
            import cpuinfo
            system_info['cpu_info'] = cpuinfo.get_cpu_info()['brand_raw']
        except ImportError:
            system_info['cpu_info'] = 'Unknown'
        
        return system_info
    
    def _save_export_report(self, exported_models: Dict[str, str]):
        """保存导出报告"""
        report_path = self.output_dir / "export_report.json"
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'source_model': self.model_path,
            'exported_models': exported_models,
            'system_info': self._get_system_info()
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"导出报告已保存: {report_path}")
    
    def _save_benchmark_report(self, benchmark_results: Dict[str, Any]):
        """保存基准测试报告"""
        report_path = self.output_dir / "benchmark_report.json"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(benchmark_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"基准测试报告已保存: {report_path}")
    
    def _save_optimization_report(self, optimization_results: Dict[str, Any]):
        """保存优化报告"""
        report_path = self.output_dir / "optimization_report.json"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(optimization_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"优化报告已保存: {report_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='YOLO模型优化工具')
    parser.add_argument('--model', '-m', type=str, help='模型文件路径')
    parser.add_argument('--action', '-a', choices=[
        'export', 'benchmark', 'optimize', 'auto'
    ], default='auto', help='执行的操作')
    parser.add_argument('--device', '-d', type=str, default='auto',
                       help='目标设备 (cpu/gpu/intel/mobile/auto)')
    parser.add_argument('--fps', '-f', type=float, default=10.0,
                       help='目标帧率')
    parser.add_argument('--runs', '-r', type=int, default=10,
                       help='基准测试运行次数')
    parser.add_argument('--formats', nargs='+', 
                       help='要导出的格式列表')
    
    args = parser.parse_args()
    
    try:
        optimizer = ModelOptimizer(args.model)
        
        if args.action == 'export':
            if not args.model:
                logger.error("导出操作需要指定模型路径")
                return
            
            results = optimizer.export_all_formats()
            
            print("\n=== 模型导出结果 ===")
            for format_name, path in results.items():
                status = "✓" if not path.startswith("ERROR") else "✗"
                print(f"{status} {format_name.upper()}: {path}")
        
        elif args.action == 'benchmark':
            results = optimizer.benchmark_models(num_runs=args.runs)
            
            print("\n=== 性能基准测试结果 ===")
            for model_name, stats in results.items():
                if 'error' in stats:
                    print(f"✗ {model_name}: {stats['error']}")
                else:
                    single = stats.get('single_image', {})
                    print(f"✓ {model_name}:")
                    print(f"  平均时间: {single.get('avg_time', 0):.3f}s")
                    print(f"  吞吐量: {single.get('throughput_fps', 0):.1f} FPS")
                    print(f"  模型大小: {stats.get('model_size_mb', 0):.1f} MB")
        
        elif args.action == 'optimize':
            results = optimizer.optimize_for_device(args.device, args.fps)
            
            print(f"\n=== {args.device}设备优化结果 ===")
            print(f"目标帧率: {args.fps} FPS")
            print("\n推荐:")
            for rec in results.get('recommendations', []):
                print(f"  • {rec}")
            
            print("\n导出的模型:")
            for format_name, path in results.get('exported_models', {}).items():
                status = "✓" if not path.startswith("ERROR") else "✗"
                print(f"  {status} {format_name}: {path}")
        
        elif args.action == 'auto':
            results = optimizer.auto_optimize()
            
            print("\n=== 自动优化结果 ===")
            
            system_info = results['system_info']
            print(f"系统类型: {system_info['device_type']}")
            if system_info['has_cuda']:
                print(f"GPU: {system_info['gpu_name']} ({system_info['gpu_memory_gb']:.1f}GB)")
            
            print("\n建议:")
            for rec in results.get('recommendations', []):
                print(f"  • {rec}")
            
            opt_results = results.get('optimization_results', {})
            if 'exported_models' in opt_results:
                print("\n导出的模型:")
                for format_name, path in opt_results['exported_models'].items():
                    status = "✓" if not path.startswith("ERROR") else "✗"
                    print(f"  {status} {format_name}: {path}")
        
        print(f"\n详细报告已保存到: {optimizer.output_dir}")
        
    except Exception as e:
        logger.error(f"操作失败: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main()) 