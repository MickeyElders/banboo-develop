"""
YOLOv8竹节检测模型训练脚本
用于训练专门的竹节检测模型
"""

import os
import argparse
import logging
import yaml
from pathlib import Path
from typing import Dict, Any

import torch
from ultralytics import YOLO
import cv2
import numpy as np

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BambooYOLOTrainer:
    """竹节YOLO模型训练器"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path
        self.config = self._load_config()
        self.model = None
        
        # 创建必要的目录
        self._setup_directories()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载训练配置"""
        default_config = {
            'model': {
                'name': 'yolov8n.pt',  # 基础模型
                'imgsz': 640,          # 图像尺寸
                'classes': 4,          # 类别数量
            },
            'training': {
                'epochs': 100,
                'batch': 16,
                'lr0': 0.01,
                'lrf': 0.01,
                'momentum': 0.937,
                'weight_decay': 0.0005,
                'warmup_epochs': 3,
                'warmup_momentum': 0.8,
                'warmup_bias_lr': 0.1,
                'patience': 50,
                'save_period': 10,
                'workers': 8,
                'device': 'auto',
            },
            'data': {
                'train_dir': 'datasets/bamboo/train',
                'val_dir': 'datasets/bamboo/val',
                'test_dir': 'datasets/bamboo/test',
                'names': {
                    0: 'natural_node',
                    1: 'artificial_node', 
                    2: 'branch_node',
                    3: 'damage_node'
                }
            },
            'augmentation': {
                'hsv_h': 0.015,
                'hsv_s': 0.7,
                'hsv_v': 0.4,
                'degrees': 0.0,
                'translate': 0.1,
                'scale': 0.5,
                'shear': 0.0,
                'perspective': 0.0,
                'flipud': 0.0,
                'fliplr': 0.5,
                'mosaic': 1.0,
                'mixup': 0.0,
                'copy_paste': 0.0
            }
        }
        
        if self.config_path and Path(self.config_path).exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f)
                # 递归更新配置
                self._update_config(default_config, user_config)
        
        return default_config
    
    def _update_config(self, base_config: Dict, update_config: Dict):
        """递归更新配置字典"""
        for key, value in update_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._update_config(base_config[key], value)
            else:
                base_config[key] = value
    
    def _setup_directories(self):
        """创建必要的目录"""
        directories = [
            'datasets/bamboo/train/images',
            'datasets/bamboo/train/labels',
            'datasets/bamboo/val/images', 
            'datasets/bamboo/val/labels',
            'datasets/bamboo/test/images',
            'datasets/bamboo/test/labels',
            'models',
            'runs/detect'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        logger.info("目录结构创建完成")
    
    def create_dataset_yaml(self) -> str:
        """创建数据集配置文件"""
        dataset_config = {
            'path': str(Path('datasets/bamboo').absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': self.config['model']['classes'],
            'names': self.config['data']['names']
        }
        
        yaml_path = 'datasets/bamboo/dataset.yaml'
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(dataset_config, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"数据集配置文件已创建: {yaml_path}")
        return yaml_path
    
    def generate_synthetic_data(self, num_images: int = 1000):
        """生成合成训练数据"""
        logger.info(f"开始生成 {num_images} 张合成训练数据...")
        
        train_images_dir = Path('datasets/bamboo/train/images')
        train_labels_dir = Path('datasets/bamboo/train/labels')
        val_images_dir = Path('datasets/bamboo/val/images')
        val_labels_dir = Path('datasets/bamboo/val/labels')
        
        for i in range(num_images):
            # 生成图像和标注
            image, annotations = self._create_synthetic_bamboo_image()
            
            # 分配到训练集或验证集 (80/20分割)
            if i < int(num_images * 0.8):
                img_path = train_images_dir / f"bamboo_{i:06d}.jpg"
                label_path = train_labels_dir / f"bamboo_{i:06d}.txt"
            else:
                img_path = val_images_dir / f"bamboo_{i:06d}.jpg"
                label_path = val_labels_dir / f"bamboo_{i:06d}.txt"
            
            # 保存图像
            cv2.imwrite(str(img_path), image)
            
            # 保存YOLO格式标注
            with open(label_path, 'w') as f:
                for ann in annotations:
                    f.write(f"{ann['class']} {ann['x_center']} {ann['y_center']} {ann['width']} {ann['height']}\n")
            
            if (i + 1) % 100 == 0:
                logger.info(f"已生成 {i + 1}/{num_images} 张图像")
        
        logger.info("合成数据生成完成")
    
    def _create_synthetic_bamboo_image(self):
        """创建单张合成竹子图像和标注"""
        # 图像尺寸
        img_height, img_width = 640, 640
        image = np.ones((img_height, img_width, 3), dtype=np.uint8) * 240
        
        annotations = []
        
        # 随机参数
        num_nodes = np.random.randint(3, 8)  # 3-7个竹节
        bamboo_y = np.random.randint(200, 440)  # 竹子y位置
        bamboo_height = np.random.randint(120, 200)  # 竹子高度
        
        # 绘制竹子主体
        bamboo_color = (
            np.random.randint(160, 200),
            np.random.randint(200, 240), 
            np.random.randint(120, 160)
        )
        cv2.rectangle(image, (50, bamboo_y), (img_width - 50, bamboo_y + bamboo_height), bamboo_color, -1)
        
        # 生成竹节位置
        node_x_positions = np.linspace(100, img_width - 100, num_nodes)
        
        for i, x_pos in enumerate(node_x_positions):
            x_pos = int(x_pos + np.random.randint(-20, 21))  # 添加随机偏移
            
            # 随机选择节点类型
            node_type = np.random.choice([0, 1, 2, 3], p=[0.6, 0.2, 0.15, 0.05])
            
            # 根据类型设置颜色
            if node_type == 0:  # 自然节点
                node_color = (
                    np.random.randint(80, 120),
                    np.random.randint(120, 160),
                    np.random.randint(60, 100)
                )
            elif node_type == 1:  # 人工节点
                node_color = (100, 100, 100)
            elif node_type == 2:  # 分支节点
                node_color = (
                    np.random.randint(60, 100),
                    np.random.randint(100, 140),
                    np.random.randint(40, 80)
                )
            else:  # 损坏节点
                node_color = (40, 60, 80)
            
            # 节点尺寸
            node_width = np.random.randint(15, 35)
            node_height = bamboo_height + np.random.randint(-10, 21)
            
            # 绘制节点
            cv2.rectangle(image, 
                         (x_pos - node_width//2, bamboo_y - 10), 
                         (x_pos + node_width//2, bamboo_y + node_height + 10),
                         node_color, -1)
            
            # 添加纹理
            for y in range(bamboo_y, bamboo_y + node_height, 8):
                cv2.line(image, 
                        (x_pos - node_width//2 + 2, y), 
                        (x_pos + node_width//2 - 2, y), 
                        tuple(max(0, c - 20) for c in node_color), 1)
            
            # 创建YOLO格式标注 (归一化坐标)
            x_center = x_pos / img_width
            y_center = (bamboo_y + node_height // 2) / img_height
            bbox_width = node_width / img_width
            bbox_height = (node_height + 20) / img_height
            
            annotations.append({
                'class': node_type,
                'x_center': x_center,
                'y_center': y_center, 
                'width': bbox_width,
                'height': bbox_height
            })
        
        # 添加噪声和模糊
        noise = np.random.normal(0, 5, image.shape).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        image = cv2.GaussianBlur(image, (3, 3), 0)
        
        return image, annotations
    
    def train(self, resume: bool = False):
        """训练模型"""
        try:
            # 创建数据集配置文件
            dataset_yaml = self.create_dataset_yaml()
            
            # 检查是否有训练数据
            train_images_dir = Path('datasets/bamboo/train/images')
            if not any(train_images_dir.glob('*.jpg')):
                logger.info("未找到训练数据，生成合成数据...")
                self.generate_synthetic_data()
            
            # 初始化模型
            model_name = self.config['model']['name']
            if resume and Path('runs/detect/train/weights/last.pt').exists():
                logger.info("从上次训练检查点恢复...")
                self.model = YOLO('runs/detect/train/weights/last.pt')
            else:
                logger.info(f"初始化模型: {model_name}")
                self.model = YOLO(model_name)
            
            # 训练参数
            train_params = {
                'data': dataset_yaml,
                'epochs': self.config['training']['epochs'],
                'imgsz': self.config['model']['imgsz'],
                'batch': self.config['training']['batch'],
                'lr0': self.config['training']['lr0'],
                'lrf': self.config['training']['lrf'],
                'momentum': self.config['training']['momentum'],
                'weight_decay': self.config['training']['weight_decay'],
                'warmup_epochs': self.config['training']['warmup_epochs'],
                'warmup_momentum': self.config['training']['warmup_momentum'],
                'warmup_bias_lr': self.config['training']['warmup_bias_lr'],
                'patience': self.config['training']['patience'],
                'save_period': self.config['training']['save_period'],
                'workers': self.config['training']['workers'],
                'device': self.config['training']['device'],
                'project': 'runs/detect',
                'name': 'train',
                'exist_ok': True,
                'pretrained': True,
                'optimizer': 'auto',
                'verbose': True,
                'seed': 0,
                'deterministic': True,
                'single_cls': False,
                'rect': False,
                'cos_lr': False,
                'close_mosaic': 10,
                'resume': resume,
                'amp': True,
                'fraction': 1.0,
                'profile': False,
                'overlap_mask': True,
                'mask_ratio': 4,
                'dropout': 0.0,
                'val': True,
                'split': 'val',
                'save_json': False,
                'save_hybrid': False,
                'conf': None,
                'iou': 0.7,
                'max_det': 300,
                'half': False,
                'dnn': False,
                'plots': True,
                'source': None,
                'show': False,
                'save_txt': False,
                'save_conf': False,
                'save_crop': False,
                'show_labels': True,
                'show_conf': True,
                'vid_stride': 1,
                'stream_buffer': False,
                'line_width': None,
                'visualize': False,
                'augment': False,
                'agnostic_nms': False,
                'classes': None,
                'retina_masks': False,
                'boxes': True,
                'format': 'torchscript',
                'keras': False,
                'optimize': False,
                'int8': False,
                'dynamic': False,
                'simplify': False,
                'opset': None,
                'workspace': 4,
                'nms': False,
                'lr0': self.config['training']['lr0'],
                'lrf': self.config['training']['lrf'],
                'momentum': self.config['training']['momentum'],
                'weight_decay': self.config['training']['weight_decay'],
                'warmup_epochs': self.config['training']['warmup_epochs'],
                'warmup_momentum': self.config['training']['warmup_momentum'],
                'warmup_bias_lr': self.config['training']['warmup_bias_lr'],
                'box': 7.5,
                'cls': 0.5,
                'dfl': 1.5,
                'pose': 12.0,
                'kobj': 1.0,
                'label_smoothing': 0.0,
                'nbs': 64,
                'hsv_h': self.config['augmentation']['hsv_h'],
                'hsv_s': self.config['augmentation']['hsv_s'],
                'hsv_v': self.config['augmentation']['hsv_v'],
                'degrees': self.config['augmentation']['degrees'],
                'translate': self.config['augmentation']['translate'],
                'scale': self.config['augmentation']['scale'],
                'shear': self.config['augmentation']['shear'],
                'perspective': self.config['augmentation']['perspective'],
                'flipud': self.config['augmentation']['flipud'],
                'fliplr': self.config['augmentation']['fliplr'],
                'mosaic': self.config['augmentation']['mosaic'],
                'mixup': self.config['augmentation']['mixup'],
                'copy_paste': self.config['augmentation']['copy_paste']
            }
            
            logger.info("开始训练...")
            logger.info(f"训练参数: {train_params}")
            
            # 开始训练
            results = self.model.train(**train_params)
            
            # 保存最终模型
            best_model_path = 'runs/detect/train/weights/best.pt'
            if Path(best_model_path).exists():
                # 复制到models目录
                import shutil
                shutil.copy2(best_model_path, 'models/bamboo_yolo_best.pt')
                logger.info("最佳模型已保存到: models/bamboo_yolo_best.pt")
            
            logger.info("训练完成！")
            return results
            
        except Exception as e:
            logger.error(f"训练失败: {e}")
            raise
    
    def validate(self, model_path: str = None):
        """验证模型"""
        try:
            if model_path:
                self.model = YOLO(model_path)
            elif not self.model:
                self.model = YOLO('models/bamboo_yolo_best.pt')
            
            dataset_yaml = self.create_dataset_yaml()
            
            logger.info("开始模型验证...")
            results = self.model.val(data=dataset_yaml, split='val')
            
            logger.info("验证完成！")
            return results
            
        except Exception as e:
            logger.error(f"验证失败: {e}")
            raise
    
    def export_model(self, model_path: str = None, format: str = 'onnx'):
        """导出模型"""
        try:
            if model_path:
                self.model = YOLO(model_path)
            elif not self.model:
                self.model = YOLO('models/bamboo_yolo_best.pt')
            
            logger.info(f"导出模型为 {format} 格式...")
            export_path = self.model.export(format=format, optimize=True)
            
            logger.info(f"模型已导出: {export_path}")
            return export_path
            
        except Exception as e:
            logger.error(f"模型导出失败: {e}")
            raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='YOLOv8竹节检测模型训练')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--action', type=str, choices=['train', 'val', 'export', 'generate'], 
                       default='train', help='执行的操作')
    parser.add_argument('--model', type=str, help='模型文件路径')
    parser.add_argument('--resume', action='store_true', help='从检查点恢复训练')
    parser.add_argument('--format', type=str, default='onnx', help='导出格式')
    parser.add_argument('--num-images', type=int, default=1000, help='生成的合成图像数量')
    
    args = parser.parse_args()
    
    # 创建训练器
    trainer = BambooYOLOTrainer(args.config)
    
    try:
        if args.action == 'generate':
            trainer.generate_synthetic_data(args.num_images)
        elif args.action == 'train':
            trainer.train(resume=args.resume)
        elif args.action == 'val':
            trainer.validate(args.model)
        elif args.action == 'export':
            trainer.export_model(args.model, args.format)
        
    except KeyboardInterrupt:
        logger.info("训练被用户中断")
    except Exception as e:
        logger.error(f"执行失败: {e}")
        raise


if __name__ == '__main__':
    main() 