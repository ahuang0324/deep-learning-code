#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorBoard实现Fashion-MNIST数据集可视化
替代原始的matplotlib可视化方法
"""

import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime
import os

# 设置随机种子以确保结果可重现
torch.manual_seed(42)

def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def load_fashion_mnist_data():
    """加载Fashion-MNIST数据集"""
    print("正在加载Fashion-MNIST数据集...")
    
    # 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式
    # 并除以255使得所有像素的数值均在0～1之间
    trans = transforms.ToTensor()
    
    # 加载训练集
    mnist_train = torchvision.datasets.FashionMNIST(
        root="./data", train=True, transform=trans, download=True)
    
    # 加载测试集
    mnist_test = torchvision.datasets.FashionMNIST(
        root="./data", train=False, transform=trans, download=True)
    
    print(f"训练集大小: {len(mnist_train)}")
    print(f"测试集大小: {len(mnist_test)}")
    print(f"图像形状: {mnist_train[0][0].shape}")
    
    return mnist_train, mnist_test

def visualize_with_tensorboard(mnist_train, mnist_test, num_samples=18):
    """使用TensorBoard可视化Fashion-MNIST数据集"""
    
    # 创建TensorBoard日志目录
    log_dir = f"runs/fashion_mnist_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir)
    
    print(f"TensorBoard日志保存在: {log_dir}")
    print("启动TensorBoard命令: tensorboard --logdir=runs")
    
    # 1. 可视化单个样本
    print("\n1. 添加单个样本到TensorBoard...")
    for i in range(min(10, len(mnist_train))):
        img, label = mnist_train[i]
        label_text = get_fashion_mnist_labels([label])[0]
        
        # 添加图像到TensorBoard
        writer.add_image(f'Single_Samples/{i:02d}_{label_text}', img, 0)
    
    # 2. 创建图像网格
    print("2. 创建图像网格...")
    
    # 获取一批训练数据
    train_loader = data.DataLoader(mnist_train, batch_size=num_samples, shuffle=True)
    X, y = next(iter(train_loader))
    
    # 获取标签文本
    labels_text = get_fashion_mnist_labels(y)
    
    # 创建图像网格 (重塑为正确的形状)
    img_grid = torchvision.utils.make_grid(
        X.reshape(num_samples, 1, 28, 28), 
        nrow=6,  # 每行6张图片
        normalize=True,
        scale_each=True,
        pad_value=1
    )
    
    # 添加图像网格到TensorBoard
    writer.add_image('Fashion_MNIST_Grid', img_grid, 0)
    
    # 3. 按类别分组可视化
    print("3. 按类别分组可视化...")
    class_samples = {}
    
    # 收集每个类别的样本
    for i, (img, label) in enumerate(mnist_train):
        label_int = int(label)
        if label_int not in class_samples:
            class_samples[label_int] = []
        if len(class_samples[label_int]) < 6:  # 每个类别收集6个样本
            class_samples[label_int].append(img)
        
        # 如果所有类别都收集够了，就停止
        if all(len(samples) >= 6 for samples in class_samples.values()) and len(class_samples) == 10:
            break
    
    # 为每个类别创建网格
    for label_int, images in class_samples.items():
        if len(images) > 0:
            label_text = get_fashion_mnist_labels([label_int])[0]
            class_grid = torchvision.utils.make_grid(
                torch.stack(images), 
                nrow=3,
                normalize=True,
                scale_each=True,
                pad_value=1
            )
            writer.add_image(f'By_Class/{label_int:02d}_{label_text}', class_grid, 0)
    
    # 4. 添加数据集统计信息
    print("4. 添加数据集统计信息...")
    
    # 计算每个类别的样本数量
    class_counts = torch.zeros(10)
    for _, label in mnist_train:
        class_counts[label] += 1
    
    # 添加直方图
    writer.add_histogram('Dataset/Class_Distribution', class_counts, 0)
    
    # 添加标量统计
    writer.add_scalar('Dataset/Total_Samples', len(mnist_train), 0)
    writer.add_scalar('Dataset/Num_Classes', 10, 0)
    writer.add_scalar('Dataset/Image_Height', 28, 0)
    writer.add_scalar('Dataset/Image_Width', 28, 0)
    
    # 5. 添加样本图像的像素值分布
    print("5. 分析像素值分布...")
    sample_batch = torch.stack([mnist_train[i][0] for i in range(1000)])
    writer.add_histogram('Dataset/Pixel_Values', sample_batch, 0)
    
    # 6. 添加文本信息
    class_names = "\n".join([f"{i}: {name}" for i, name in enumerate(
        ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
         'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    )])
    writer.add_text('Dataset/Class_Names', class_names, 0)
    
    # 关闭writer
    writer.close()
    
    print("\n✅ TensorBoard可视化完成!")
    print(f"📁 日志目录: {log_dir}")
    print("🚀 启动TensorBoard: tensorboard --logdir=runs")
    print("🌐 然后在浏览器中打开: http://localhost:6006")
    
    return log_dir

def compare_with_matplotlib(mnist_train, num_samples=18):
    """对比原始matplotlib方法和TensorBoard方法"""
    print("\n=== 方法对比 ===")
    print("\n📊 原始matplotlib方法:")
    print("- 优点: 直接在notebook中显示，无需额外工具")
    print("- 缺点: 静态显示，功能有限，不支持交互")
    
    print("\n📈 TensorBoard方法:")
    print("- 优点: 交互式可视化，支持缩放、过滤、动态更新")
    print("- 优点: 可以保存历史记录，支持多种可视化类型")
    print("- 优点: 专业的深度学习可视化工具")
    print("- 缺点: 需要启动额外的服务")

def main():
    """主函数"""
    print("=== 使用TensorBoard可视化Fashion-MNIST数据集 ===")
    
    # 加载数据
    mnist_train, mnist_test = load_fashion_mnist_data()
    
    # 使用TensorBoard可视化
    log_dir = visualize_with_tensorboard(mnist_train, mnist_test)
    
    # 方法对比
    compare_with_matplotlib(mnist_train)
    
    print("\n=== 使用说明 ===")
    print("1. 确保已安装tensorboard: pip install tensorboard")
    print("2. 在终端运行: tensorboard --logdir=runs")
    print("3. 在浏览器中打开: http://localhost:6006")
    print("4. 在TensorBoard界面中切换不同的标签页查看可视化结果")
    
if __name__ == "__main__":
    main()