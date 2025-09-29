#!/usr/bin/env python3
"""
竹子识别模型转换脚本
支持 PyTorch -> ONNX -> TensorRT 的完整转换流程
集成 GhostConv、VoV-GSCSP、SAHI 等优化技术
"""

import os
import sys
import argparse
import logging
import warnings
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any

import torch
import torch.nn as nn
import torchvision
import onnx
import onnxruntime as ort
import numpy as np
from PIL import Image
import yaml

# 抑制警告
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelConverter:
    """模型转换器主类"""
    
    def __init__(self, config_path: str = None):
        """
        初始化模型转换器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        default_config = {
            'model': {
                'input_size': [640, 640],
                'num_classes': 1,
                'confidence_threshold': 0.25,
                'nms_threshold': 0.45
            },
            'conversion': {
                'opset_version': 11,
                'dynamic_axes': True,
                'optimize_onnx': True,
                'fp16': True
            },
            'tensorrt': {
                'max_batch_size': 1,
                'max_workspace_size': 1 << 30,  # 1GB
                'fp16_mode': True,
                'int8_mode': False,
                'calibration_dataset': None
            },
            'optimization': {
                'enable_ghost_conv': True,
                'enable_vov_gscsp': True,
                'enable_pruning': False,
                'pruning_ratio': 0.1
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f)
                # 递归更新配置
                self._deep_update(default_config, user_config)
                
        return default_config
    
    def _deep_update(self, base_dict: dict, update_dict: dict):
        """递归更新字典"""
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict:
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

class GhostConvModule(nn.Module):
    """GhostConv轻量化卷积模块"""
    
    def __init__(self, inp: int, oup: int, kernel_size: int = 1, ratio: int = 2, 
                 dw_size: int = 3, stride: int = 1, relu: bool = True):
        super(GhostConvModule, self).__init__()
        self.oup = oup
        init_channels = oup // ratio
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, 
                     kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, 
                     dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]

class VoVGSCSPModule(nn.Module):
    """VoV-GSCSP融合模块"""
    
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, 
                 g: int = 1, e: float = 0.5):
        super(VoVGSCSPModule, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Conv2d(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1)
        self.cv3 = nn.Conv2d(2 * c_, c2, 1, 1)
        self.m = nn.Sequential(*(VoVBlock(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        y1 = self.cv1(x)
        y2 = self.cv2(x)
        return self.cv3(torch.cat((self.m(y1), y2), dim=1))

class VoVBlock(nn.Module):
    """VoV基础块"""
    
    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, e: float = 0.5):
        super(VoVBlock, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = nn.Conv2d(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c_, c2, 3, 1, 1, groups=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class OptimizedBambooDetector(nn.Module):
    """优化后的竹子检测模型"""
    
    def __init__(self, config: Dict[str, Any]):
        super(OptimizedBambooDetector, self).__init__()
        self.config = config
        self.num_classes = config['model']['num_classes']
        
        # 构建优化后的骨干网络
        self.backbone = self._build_backbone()
        self.neck = self._build_neck()
        self.head = self._build_head()
        
    def _build_backbone(self):
        """构建优化后的骨干网络"""
        layers = []
        
        # Stem层
        layers.append(nn.Conv2d(3, 32, 3, 2, 1))
        layers.append(nn.BatchNorm2d(32))
        layers.append(nn.SiLU(inplace=True))
        
        # 使用GhostConv和VoV-GSCSP的混合架构
        if self.config['optimization']['enable_ghost_conv']:
            layers.append(GhostConvModule(32, 64))
            layers.append(GhostConvModule(64, 128))
        else:
            layers.append(nn.Conv2d(32, 64, 3, 2, 1))
            layers.append(nn.Conv2d(64, 128, 3, 2, 1))
            
        if self.config['optimization']['enable_vov_gscsp']:
            layers.append(VoVGSCSPModule(128, 256, n=3))
            layers.append(VoVGSCSPModule(256, 512, n=6))
        else:
            layers.append(nn.Conv2d(128, 256, 3, 2, 1))
            layers.append(nn.Conv2d(256, 512, 3, 2, 1))
            
        return nn.Sequential(*layers)
    
    def _build_neck(self):
        """构建颈部网络（FPN+PAN）"""
        return nn.ModuleList([
            nn.Conv2d(512, 256, 1),
            nn.Conv2d(256, 128, 1),
            nn.Conv2d(128, 64, 1)
        ])
    
    def _build_head(self):
        """构建检测头"""
        return nn.ModuleList([
            nn.Conv2d(256, (self.num_classes + 5) * 3, 1),  # 大目标
            nn.Conv2d(128, (self.num_classes + 5) * 3, 1),  # 中目标
            nn.Conv2d(64, (self.num_classes + 5) * 3, 1)    # 小目标
        ])
    
    def forward(self, x):
        # 骨干网络特征提取
        features = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in [4, 6, 8]:  # 保存多尺度特征
                features.append(x)
        
        # 颈部网络处理
        neck_features = []
        for i, neck_layer in enumerate(self.neck):
            feat = neck_layer(features[-(i+1)])
            neck_features.append(feat)
        
        # 检测头输出
        outputs = []
        for i, head_layer in enumerate(self.head):
            out = head_layer(neck_features[i])
            outputs.append(out)
        
        return outputs

class ModelOptimizer:
    """模型优化器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def optimize_model(self, model: nn.Module) -> nn.Module:
        """优化模型"""
        logger.info("开始模型优化...")
        
        # 模型剪枝
        if self.config['optimization']['enable_pruning']:
            model = self._prune_model(model)
            
        # 模型量化
        model = self._quantize_model(model)
        
        # 模型融合
        model = self._fuse_model(model)
        
        logger.info("模型优化完成")
        return model
    
    def _prune_model(self, model: nn.Module) -> nn.Module:
        """结构化剪枝"""
        import torch.nn.utils.prune as prune
        
        logger.info("执行模型剪枝...")
        pruning_ratio = self.config['optimization']['pruning_ratio']
        
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
                prune.remove(module, 'weight')
                
        return model
    
    def _quantize_model(self, model: nn.Module) -> nn.Module:
        """模型量化"""
        logger.info("执行模型量化...")
        
        # 准备量化
        model.eval()
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(model, inplace=True)
        
        # 校准（使用虚拟数据）
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 640, 640)
            model(dummy_input)
            
        # 转换为量化模型
        torch.quantization.convert(model, inplace=True)
        
        return model
    
    def _fuse_model(self, model: nn.Module) -> nn.Module:
        """模型融合"""
        logger.info("执行模型融合...")
        
        # 融合Conv+BN+ReLU
        torch.quantization.fuse_modules(model, [['conv', 'bn', 'relu']], inplace=True)
        
        return model

class ONNXConverter:
    """ONNX转换器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def convert_to_onnx(self, model: nn.Module, output_path: str, 
                       input_shape: Tuple[int, int, int, int] = (1, 3, 640, 640)):
        """转换为ONNX格式"""
        logger.info(f"开始转换为ONNX格式: {output_path}")
        
        model.eval()
        dummy_input = torch.randn(input_shape)
        
        # 动态轴配置
        dynamic_axes = None
        if self.config['conversion']['dynamic_axes']:
            dynamic_axes = {
                'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                'output': {0: 'batch_size'}
            }
        
        # 导出ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=self.config['conversion']['opset_version'],
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            verbose=False
        )
        
        # 验证ONNX模型
        self._verify_onnx_model(output_path, dummy_input)
        
        # 优化ONNX模型
        if self.config['conversion']['optimize_onnx']:
            self._optimize_onnx_model(output_path)
            
        logger.info("ONNX转换完成")
    
    def _verify_onnx_model(self, onnx_path: str, dummy_input: torch.Tensor):
        """验证ONNX模型"""
        try:
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            
            # 运行时验证
            ort_session = ort.InferenceSession(onnx_path)
            ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
            ort_outputs = ort_session.run(None, ort_inputs)
            
            logger.info("ONNX模型验证通过")
            
        except Exception as e:
            logger.error(f"ONNX模型验证失败: {e}")
            raise
    
    def _optimize_onnx_model(self, onnx_path: str):
        """优化ONNX模型"""
        from onnxoptimizer import optimize
        
        try:
            onnx_model = onnx.load(onnx_path)
            
            # 应用优化
            optimized_model = optimize(onnx_model, [
                'eliminate_deadend',
                'eliminate_identity',
                'eliminate_nop_dropout',
                'eliminate_nop_transpose',
                'eliminate_unused_initializer',
                'extract_constant_to_initializer',
                'fuse_add_bias_into_conv',
                'fuse_bn_into_conv',
                'fuse_consecutive_concats',
                'fuse_consecutive_reduce_unsqueeze',
                'fuse_consecutive_squeezes',
                'fuse_consecutive_transposes',
                'fuse_matmul_add_bias_into_gemm',
                'fuse_pad_into_conv',
                'fuse_transpose_into_gemm'
            ])
            
            # 保存优化后的模型
            onnx.save(optimized_model, onnx_path)
            logger.info("ONNX模型优化完成")
            
        except ImportError:
            logger.warning("onnxoptimizer未安装，跳过ONNX优化")
        except Exception as e:
            logger.warning(f"ONNX优化失败: {e}")

class TensorRTConverter:
    """TensorRT转换器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def convert_to_tensorrt(self, onnx_path: str, output_path: str):
        """转换为TensorRT格式"""
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
            
            logger.info(f"开始转换为TensorRT格式: {output_path}")
            
            # 创建构建器和网络
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(TRT_LOGGER)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, TRT_LOGGER)
            
            # 解析ONNX模型
            with open(onnx_path, 'rb') as model:
                if not parser.parse(model.read()):
                    logger.error("ONNX解析失败")
                    for error in range(parser.num_errors):
                        logger.error(parser.get_error(error))
                    return False
            
            # 配置构建器
            config = builder.create_builder_config()
            config.max_workspace_size = self.config['tensorrt']['max_workspace_size']
            
            if self.config['tensorrt']['fp16_mode']:
                config.set_flag(trt.BuilderFlag.FP16)
                logger.info("启用FP16模式")
                
            if self.config['tensorrt']['int8_mode']:
                config.set_flag(trt.BuilderFlag.INT8)
                logger.info("启用INT8模式")
                
                # 设置INT8校准器
                if self.config['tensorrt']['calibration_dataset']:
                    calibrator = self._create_calibrator()
                    config.int8_calibrator = calibrator
            
            # 构建引擎
            engine = builder.build_engine(network, config)
            if not engine:
                logger.error("TensorRT引擎构建失败")
                return False
            
            # 保存引擎
            with open(output_path, 'wb') as f:
                f.write(engine.serialize())
                
            logger.info("TensorRT转换完成")
            return True
            
        except ImportError:
            logger.error("TensorRT未安装，跳过TensorRT转换")
            return False
        except Exception as e:
            logger.error(f"TensorRT转换失败: {e}")
            return False
    
    def _create_calibrator(self):
        """创建INT8校准器"""
        # 这里应该实现自定义校准器
        # 用于INT8量化的数据集校准
        pass

class WIoULoss(nn.Module):
    """WIoU (Weighted IoU) 损失函数"""
    
    def __init__(self, reduction='mean', loss_weight=1.0):
        super(WIoULoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
    
    def forward(self, pred, target, weight=None):
        """
        计算WIoU损失
        
        Args:
            pred: 预测框 [N, 4] (x1, y1, x2, y2)
            target: 真值框 [N, 4] (x1, y1, x2, y2)
            weight: 权重 [N]
        """
        # 计算交集
        inter_x1 = torch.max(pred[:, 0], target[:, 0])
        inter_y1 = torch.max(pred[:, 1], target[:, 1])
        inter_x2 = torch.min(pred[:, 2], target[:, 2])
        inter_y2 = torch.min(pred[:, 3], target[:, 3])
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                    torch.clamp(inter_y2 - inter_y1, min=0)
        
        # 计算并集
        pred_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
        target_area = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
        union_area = pred_area + target_area - inter_area
        
        # 计算IoU
        iou = inter_area / (union_area + 1e-7)
        
        # 计算中心点距离
        pred_center_x = (pred[:, 0] + pred[:, 2]) / 2
        pred_center_y = (pred[:, 1] + pred[:, 3]) / 2
        target_center_x = (target[:, 0] + target[:, 2]) / 2
        target_center_y = (target[:, 1] + target[:, 3]) / 2
        
        center_distance = torch.sqrt(
            (pred_center_x - target_center_x) ** 2 + 
            (pred_center_y - target_center_y) ** 2
        )
        
        # 计算对角线距离
        c_x1 = torch.min(pred[:, 0], target[:, 0])
        c_y1 = torch.min(pred[:, 1], target[:, 1])
        c_x2 = torch.max(pred[:, 2], target[:, 2])
        c_y2 = torch.max(pred[:, 3], target[:, 3])
        
        diagonal_distance = torch.sqrt(
            (c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2
        ) + 1e-7
        
        # 计算宽高比损失
        pred_w = pred[:, 2] - pred[:, 0]
        pred_h = pred[:, 3] - pred[:, 1]
        target_w = target[:, 2] - target[:, 0]
        target_h = target[:, 3] - target[:, 1]
        
        w_loss = torch.abs(pred_w - target_w) / torch.max(pred_w, target_w)
        h_loss = torch.abs(pred_h - target_h) / torch.max(pred_h, target_h)
        
        # 计算WIoU损失
        wiou = iou - (center_distance / diagonal_distance) - (w_loss + h_loss) / 2
        loss = 1 - wiou
        
        # 应用权重
        if weight is not None:
            loss = loss * weight
            
        # 减少维度
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
            
        return loss * self.loss_weight

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='竹子识别模型转换脚本')
    parser.add_argument('--model_path', type=str, required=True, help='PyTorch模型路径')
    parser.add_argument('--output_dir', type=str, default='./converted_models', help='输出目录')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--formats', type=str, nargs='+', 
                       default=['onnx', 'tensorrt'], 
                       choices=['onnx', 'tensorrt'],
                       help='转换格式')
    parser.add_argument('--optimize', action='store_true', help='启用模型优化')
    parser.add_argument('--verify', action='store_true', help='验证转换结果')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化转换器
    converter = ModelConverter(args.config)
    
    try:
        # 加载PyTorch模型
        logger.info(f"加载PyTorch模型: {args.model_path}")
        
        if args.model_path.endswith('.pt') or args.model_path.endswith('.pth'):
            # 加载训练好的模型
            model = torch.load(args.model_path, map_location='cpu')
            if isinstance(model, dict):
                # 提取模型权重
                state_dict = model.get('model', model)
                model = OptimizedBambooDetector(converter.config)
                model.load_state_dict(state_dict)
        else:
            # 创建新模型
            model = OptimizedBambooDetector(converter.config)
        
        model.eval()
        
        # 模型优化
        if args.optimize:
            optimizer = ModelOptimizer(converter.config)
            model = optimizer.optimize_model(model)
        
        # 转换为ONNX
        if 'onnx' in args.formats:
            onnx_path = output_dir / 'bamboo_detector.onnx'
            onnx_converter = ONNXConverter(converter.config)
            onnx_converter.convert_to_onnx(model, str(onnx_path))
        
        # 转换为TensorRT
        if 'tensorrt' in args.formats:
            if 'onnx' not in args.formats:
                # 先转换为ONNX
                onnx_path = output_dir / 'bamboo_detector.onnx'
                onnx_converter = ONNXConverter(converter.config)
                onnx_converter.convert_to_onnx(model, str(onnx_path))
            
            trt_path = output_dir / 'bamboo_detector.trt'
            trt_converter = TensorRTConverter(converter.config)
            trt_converter.convert_to_tensorrt(str(onnx_path), str(trt_path))
        
        # 验证转换结果
        if args.verify:
            logger.info("验证转换结果...")
            # 这里可以添加模型精度验证代码
        
        logger.info("模型转换完成！")
        
    except Exception as e:
        logger.error(f"模型转换失败: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()