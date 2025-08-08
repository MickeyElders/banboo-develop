#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能切竹机模型转换和优化脚本
支持PyTorch -> ONNX -> TensorRT转换，包括INT8量化
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import numpy as np
import torch
import onnx
import onnxsim
from ultralytics import YOLO
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelConverter:
    """模型转换器类"""
    
    def __init__(self, model_path: str, output_dir: str = "./"):
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 检查模型文件是否存在
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件未找到: {self.model_path}")
        
        logger.info(f"初始化模型转换器: {self.model_path}")
    
    def pt_to_onnx(self, input_shape=(1, 3, 640, 640), dynamic_axes=True):
        """将PyTorch模型转换为ONNX格式"""
        logger.info("开始PyTorch -> ONNX转换...")
        
        try:
            # 加载YOLO模型
            model = YOLO(str(self.model_path))
            
            # 输出路径
            onnx_path = self.output_dir / "bamboo_detection.onnx"
            
            # 导出为ONNX
            if dynamic_axes:
                # 支持动态输入尺寸
                model.export(
                    format="onnx",
                    imgsz=input_shape[-2:],  # (height, width)
                    dynamic=True,
                    simplify=True,
                    opset=11
                )
                # 移动文件到指定位置
                src_onnx = self.model_path.parent / f"{self.model_path.stem}.onnx"
                if src_onnx.exists():
                    src_onnx.rename(onnx_path)
            else:
                # 固定输入尺寸
                model.export(
                    format="onnx",
                    imgsz=input_shape[-2:],
                    dynamic=False,
                    simplify=True,
                    opset=11
                )
                src_onnx = self.model_path.parent / f"{self.model_path.stem}.onnx"
                if src_onnx.exists():
                    src_onnx.rename(onnx_path)
            
            logger.info(f"ONNX模型已保存: {onnx_path}")
            return str(onnx_path)
            
        except Exception as e:
            logger.error(f"PyTorch -> ONNX转换失败: {e}")
            raise
    
    def optimize_onnx(self, onnx_path: str):
        """优化ONNX模型"""
        logger.info("开始ONNX模型优化...")
        
        try:
            # 加载ONNX模型
            onnx_model = onnx.load(onnx_path)
            
            # 优化模型
            optimized_model = onnxsim.simplify(onnx_model)
            
            # 保存优化后的模型
            optimized_path = self.output_dir / "bamboo_detection_optimized.onnx"
            onnx.save(optimized_model[0], str(optimized_path))
            
            logger.info(f"优化后的ONNX模型已保存: {optimized_path}")
            return str(optimized_path)
            
        except Exception as e:
            logger.error(f"ONNX模型优化失败: {e}")
            raise
    
    def onnx_to_tensorrt(self, onnx_path: str, precision="fp16", max_batch_size=1, calibration_data=None):
        """将ONNX模型转换为TensorRT引擎"""
        logger.info(f"开始ONNX -> TensorRT转换 (精度: {precision})...")
        
        try:
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(TRT_LOGGER)
            config = builder.create_builder_config()
            
            # 设置内存池大小 (1GB)
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
            
            # 创建网络
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, TRT_LOGGER)
            
            # 解析ONNX模型
            with open(onnx_path, 'rb') as model:
                if not parser.parse(model.read()):
                    for error in range(parser.num_errors):
                        logger.error(f"ONNX解析错误: {parser.get_error(error)}")
                    raise RuntimeError("ONNX模型解析失败")
            
            # 配置精度
            if precision == "fp16":
                if builder.platform_has_fast_fp16:
                    config.set_flag(trt.BuilderFlag.FP16)
                    logger.info("启用FP16精度")
                else:
                    logger.warning("平台不支持FP16，使用FP32")
            elif precision == "int8":
                if builder.platform_has_fast_int8:
                    config.set_flag(trt.BuilderFlag.INT8)
                    if calibration_data:
                        # 设置INT8校准器
                        calibrator = self._create_calibrator(calibration_data)
                        config.int8_calibrator = calibrator
                    logger.info("启用INT8精度")
                else:
                    logger.warning("平台不支持INT8，使用FP16")
                    if builder.platform_has_fast_fp16:
                        config.set_flag(trt.BuilderFlag.FP16)
            
            # 构建引擎
            logger.info("正在构建TensorRT引擎，请稍候...")
            serialized_engine = builder.build_serialized_network(network, config)
            
            if serialized_engine is None:
                raise RuntimeError("TensorRT引擎构建失败")
            
            # 保存引擎
            engine_suffix = f"_{precision}" if precision != "fp16" else ""
            engine_path = self.output_dir / f"bamboo_detection{engine_suffix}.trt"
            
            with open(engine_path, "wb") as f:
                f.write(serialized_engine)
            
            logger.info(f"TensorRT引擎已保存: {engine_path}")
            
            # 显示模型信息
            self._print_engine_info(serialized_engine, precision)
            
            return str(engine_path)
            
        except Exception as e:
            logger.error(f"ONNX -> TensorRT转换失败: {e}")
            raise
    
    def _create_calibrator(self, calibration_data_path):
        """创建INT8校准器"""
        # 这里应该实现自定义的校准器
        # 暂时返回None，使用TensorRT的默认校准器
        logger.info("使用默认INT8校准器")
        return None
    
    def _print_engine_info(self, serialized_engine, precision):
        """打印引擎信息"""
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        
        logger.info("=== TensorRT引擎信息 ===")
        logger.info(f"精度模式: {precision.upper()}")
        logger.info(f"最大批处理大小: {engine.max_batch_size}")
        logger.info(f"输入数量: {engine.num_bindings}")
        
        for i in range(engine.num_bindings):
            binding_name = engine.get_binding_name(i)
            binding_shape = engine.get_binding_shape(i)
            is_input = engine.binding_is_input(i)
            logger.info(f"  {'输入' if is_input else '输出'} {i}: {binding_name} - 形状: {binding_shape}")
    
    def benchmark_model(self, model_path: str, num_iterations=100):
        """模型性能基准测试"""
        logger.info(f"开始模型性能测试: {model_path} ({num_iterations} 次迭代)")
        
        try:
            if model_path.endswith('.trt'):
                return self._benchmark_tensorrt(model_path, num_iterations)
            elif model_path.endswith('.onnx'):
                return self._benchmark_onnx(model_path, num_iterations)
            else:
                logger.error("不支持的模型格式")
                return None
        except Exception as e:
            logger.error(f"性能测试失败: {e}")
            return None
    
    def _benchmark_tensorrt(self, engine_path: str, num_iterations: int):
        """TensorRT模型性能测试"""
        import time
        
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)
        
        with open(engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        
        context = engine.create_execution_context()
        
        # 准备输入数据
        input_shape = engine.get_binding_shape(0)
        input_data = np.random.randn(*input_shape).astype(np.float32)
        
        # GPU内存分配
        d_input = cuda.mem_alloc(input_data.nbytes)
        cuda.memcpy_htod(d_input, input_data)
        
        # 预热
        for _ in range(10):
            context.execute_v2([int(d_input)])
        
        # 性能测试
        start_time = time.time()
        for _ in range(num_iterations):
            context.execute_v2([int(d_input)])
        cuda.Context.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_iterations * 1000  # ms
        fps = 1000 / avg_time
        
        logger.info(f"TensorRT性能: 平均推理时间 {avg_time:.2f}ms, FPS: {fps:.1f}")
        return {"avg_time_ms": avg_time, "fps": fps}
    
    def _benchmark_onnx(self, onnx_path: str, num_iterations: int):
        """ONNX模型性能测试"""
        try:
            import onnxruntime as ort
            import time
            
            # 创建推理会话
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            session = ort.InferenceSession(onnx_path, providers=providers)
            
            # 获取输入信息
            input_name = session.get_inputs()[0].name
            input_shape = session.get_inputs()[0].shape
            
            # 处理动态尺寸
            if any(dim is None or dim < 0 for dim in input_shape):
                input_shape = [1, 3, 640, 640]  # 默认尺寸
            
            # 准备输入数据
            input_data = np.random.randn(*input_shape).astype(np.float32)
            
            # 预热
            for _ in range(10):
                session.run(None, {input_name: input_data})
            
            # 性能测试
            start_time = time.time()
            for _ in range(num_iterations):
                session.run(None, {input_name: input_data})
            end_time = time.time()
            
            avg_time = (end_time - start_time) / num_iterations * 1000  # ms
            fps = 1000 / avg_time
            
            logger.info(f"ONNX性能: 平均推理时间 {avg_time:.2f}ms, FPS: {fps:.1f}")
            return {"avg_time_ms": avg_time, "fps": fps}
            
        except ImportError:
            logger.error("缺少onnxruntime，无法进行ONNX性能测试")
            return None

def main():
    parser = argparse.ArgumentParser(description="智能切竹机模型转换和优化工具")
    parser.add_argument("--model", required=True, help="输入模型路径")
    parser.add_argument("--output", default="./", help="输出目录")
    parser.add_argument("--format", choices=["onnx", "tensorrt", "all"], default="all", help="转换格式")
    parser.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp16", help="TensorRT精度")
    parser.add_argument("--input-shape", nargs=4, type=int, default=[1, 3, 640, 640], help="输入形状 NCHW")
    parser.add_argument("--dynamic", action="store_true", help="支持动态输入尺寸")
    parser.add_argument("--benchmark", action="store_true", help="进行性能基准测试")
    parser.add_argument("--calibration-data", help="INT8校准数据路径")
    
    args = parser.parse_args()
    
    try:
        converter = ModelConverter(args.model, args.output)
        
        if args.format in ["onnx", "all"]:
            # PT -> ONNX
            onnx_path = converter.pt_to_onnx(tuple(args.input_shape), args.dynamic)
            
            # 优化ONNX
            optimized_onnx_path = converter.optimize_onnx(onnx_path)
            
            if args.benchmark:
                converter.benchmark_model(optimized_onnx_path)
        
        if args.format in ["tensorrt", "all"]:
            # 确保有ONNX文件
            if args.format == "tensorrt":
                if not Path(args.output).joinpath("bamboo_detection_optimized.onnx").exists():
                    onnx_path = converter.pt_to_onnx(tuple(args.input_shape), args.dynamic)
                    optimized_onnx_path = converter.optimize_onnx(onnx_path)
                else:
                    optimized_onnx_path = str(Path(args.output).joinpath("bamboo_detection_optimized.onnx"))
            
            # ONNX -> TensorRT
            engine_path = converter.onnx_to_tensorrt(
                optimized_onnx_path, 
                args.precision,
                calibration_data=args.calibration_data
            )
            
            if args.benchmark:
                converter.benchmark_model(engine_path)
        
        logger.info("模型转换完成！")
        
        # 显示文件信息
        output_dir = Path(args.output)
        for file_path in output_dir.glob("bamboo_detection*"):
            file_size = file_path.stat().st_size / (1024 * 1024)  # MB
            logger.info(f"输出文件: {file_path.name} ({file_size:.1f} MB)")
        
    except Exception as e:
        logger.error(f"转换过程出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()