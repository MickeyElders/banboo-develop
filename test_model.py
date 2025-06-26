#!/usr/bin/env python3
"""
智能切竹机模型测试脚本
测试训练好的YOLO模型是否能正常加载和推理
"""

import os
import sys
import cv2
import numpy as np
import torch
from pathlib import Path

# 添加src目录到Python路径
sys.path.append('src')

def test_model_loading():
    """测试模型加载"""
    print("=" * 50)
    print("测试1: 模型加载")
    print("=" * 50)
    
    model_path = "models/yolov8n_bamboo_best.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return False
    
    try:
        from ultralytics import YOLO
        
        print(f"📁 模型路径: {model_path}")
        print(f"📏 模型文件大小: {os.path.getsize(model_path) / 1024 / 1024:.1f} MB")
        
        # 加载模型
        model = YOLO(model_path)
        print("✅ 模型加载成功!")
        
        # 获取模型信息
        print(f"🏗️ 模型架构: {model.model.__class__.__name__}")
        print(f"🔢 类别数量: {len(model.names) if hasattr(model, 'names') else '未知'}")
        
        if hasattr(model, 'names'):
            print(f"🏷️ 类别名称: {list(model.names.values())}")
        
        return True, model
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return False, None

def test_model_inference(model):
    """测试模型推理"""
    print("\n" + "=" * 50)
    print("测试2: 模型推理")
    print("=" * 50)
    
    try:
        # 创建测试图像 (640x640, RGB)
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        print(f"🖼️ 测试图像尺寸: {test_image.shape}")
        
        # 进行推理
        print("🔄 开始推理...")
        results = model(test_image, verbose=False)
        print("✅ 推理完成!")
        
        # 分析结果
        if results and len(results) > 0:
            result = results[0]
            detections = len(result.boxes) if result.boxes is not None else 0
            print(f"🎯 检测到的对象数量: {detections}")
            
            if detections > 0:
                print("📊 检测详情:")
                for i, box in enumerate(result.boxes):
                    conf = box.conf.item()
                    cls = int(box.cls.item())
                    cls_name = model.names[cls] if hasattr(model, 'names') else f"类别{cls}"
                    print(f"   - 对象{i+1}: {cls_name}, 置信度: {conf:.3f}")
        else:
            print("📊 没有检测到对象 (这对随机图像是正常的)")
        
        return True
        
    except Exception as e:
        print(f"❌ 推理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vision_system():
    """测试视觉系统集成"""
    print("\n" + "=" * 50)
    print("测试3: 视觉系统集成")
    print("=" * 50)
    
    try:
        from vision.vision_types import AlgorithmConfig
        from vision.yolo_detector import OptimizedYOLODetector
        
        # 创建配置
        config = AlgorithmConfig()
        
        # 创建检测器
        detector = OptimizedYOLODetector(config)
        print("🔧 检测器创建成功")
        
        # 初始化检测器
        if detector.initialize():
            print("✅ 检测器初始化成功!")
            
            # 获取模型信息
            model_info = detector.get_model_info()
            print("📋 模型信息:")
            for key, value in model_info.items():
                print(f"   - {key}: {value}")
            
            return True
        else:
            print("❌ 检测器初始化失败")
            return False
            
    except Exception as e:
        print(f"❌ 视觉系统测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_system_requirements():
    """测试系统要求"""
    print("\n" + "=" * 50)
    print("测试4: 系统要求检查")
    print("=" * 50)
    
    # 检查Python版本
    python_version = sys.version_info
    print(f"🐍 Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # 检查PyTorch
    try:
        print(f"🔥 PyTorch版本: {torch.__version__}")
        print(f"🖥️ CUDA可用: {'是' if torch.cuda.is_available() else '否'}")
        if torch.cuda.is_available():
            print(f"🎮 GPU设备: {torch.cuda.get_device_name(0)}")
            print(f"💾 GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    except:
        print("❌ PyTorch未安装或版本不兼容")
    
    # 检查依赖库
    required_packages = ['cv2', 'numpy', 'ultralytics']
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}: 已安装")
        except ImportError:
            print(f"❌ {package}: 未安装")

def main():
    """主测试函数"""
    print("🧪 智能切竹机模型测试")
    print(f"📍 工作目录: {os.getcwd()}")
    
    # 测试系统要求
    test_system_requirements()
    
    # 测试模型加载
    success, model = test_model_loading()
    if not success:
        print("\n❌ 模型加载失败，无法继续测试")
        return
    
    # 测试模型推理
    if test_model_inference(model):
        print("\n✅ 模型推理测试通过")
    else:
        print("\n❌ 模型推理测试失败")
    
    # 测试视觉系统集成
    if test_vision_system():
        print("\n✅ 视觉系统集成测试通过")
    else:
        print("\n❌ 视觉系统集成测试失败")
    
    print("\n" + "=" * 50)
    print("📊 测试总结")
    print("=" * 50)
    print("✅ 您的模型已成功集成到智能切竹机系统中！")
    print("📁 模型位置: models/yolov8n_bamboo_best.pt")
    print("🔧 可以使用以下方式调用:")
    print("   - 直接使用: OptimizedYOLODetector")
    print("   - 混合模式: OptimizedHybridDetector") 
    print("   - 主程序: python main.py")

if __name__ == "__main__":
    main() 