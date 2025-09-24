import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, ConnectionPatch
import matplotlib.font_manager as fm

# 设置中文字体支持
try:
    # 尝试使用微软雅黑字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
except:
    print("警告: 可能无法正确显示中文字体")

def create_softmax_network_visualization(save_path='softmax_network.png'):
    """
    创建类似图片中的softmax回归网络结构图
    """
    # 创建深色背景的图
    plt.figure(figsize=(8, 6), facecolor='#1e1e1e')
    ax = plt.gca()
    ax.set_facecolor('#1e1e1e')
    
    # 设置坐标轴不可见
    plt.axis('off')
    
    # 定义节点位置
    input_nodes = 4  # 输入节点数量
    output_nodes = 3  # 输出节点数量
    
    # 输入层和输出层的y坐标
    input_y = 0.3
    output_y = 0.7
    
    # 节点颜色
    node_color = '#80d0ff'  # 浅蓝色
    
    # 绘制输入层节点
    input_positions = []
    for i in range(input_nodes):
        x = 0.2 + i * 0.2
        input_positions.append((x, input_y))
        circle = Circle((x, input_y), 0.05, color=node_color, alpha=0.9)
        ax.add_patch(circle)
        plt.text(x, input_y, f'$x_{i+1}$', ha='center', va='center', color='black', fontsize=12)
    
    # 绘制输出层节点
    output_positions = []
    for i in range(output_nodes):
        x = 0.3 + i * 0.2
        output_positions.append((x, output_y))
        circle = Circle((x, output_y), 0.05, color=node_color, alpha=0.9)
        ax.add_patch(circle)
        plt.text(x, output_y, f'$o_{i+1}$', ha='center', va='center', color='black', fontsize=12)
    
    # 绘制连接线
    for i, (x_in, y_in) in enumerate(input_positions):
        for j, (x_out, y_out) in enumerate(output_positions):
            con = ConnectionPatch(
                (x_in, y_in), (x_out, y_out),
                'data', 'data',
                arrowstyle='-',
                color='#555555',
                linewidth=1
            )
            ax.add_artist(con)
    
    # 添加层标签
    plt.text(0.1, input_y, '输入层', ha='right', va='center', color='white', fontsize=12)
    plt.text(0.1, output_y, '输出层', ha='right', va='center', color='white', fontsize=12)
    
    # 添加图标签
    plt.figtext(0.5, 0.05, ':label: fig_softmaxreg', ha='center', color='#f0d060', fontsize=12)
    
    # 设置坐标轴范围
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#1e1e1e')
    print(f"网络结构图已保存至 {save_path}")
    
    # 显示图像
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    create_softmax_network_visualization()