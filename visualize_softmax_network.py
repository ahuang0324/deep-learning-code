import torch
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

def visualize_softmax_network():
    """可视化softmax回归网络结构"""
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 1. 网络结构图
    G = nx.DiGraph()
    
    # 添加节点
    input_nodes = [f'x{i}' for i in range(1, 6)]  # 为了可视化，只显示部分输入
    hidden_nodes = ['h1', 'h2', 'h3', 'h4', 'h5']  # 为了可视化
    output_nodes = [f'y{i}' for i in range(1, 11)]
    
    # 添加输入层节点
    for node in input_nodes:
        G.add_node(node, layer='input', pos=(0, 0))
    
    # 添加输出层节点
    for i, node in enumerate(output_nodes):
        G.add_node(node, layer='output', pos=(2, 4-i*0.8))
    
    # 添加边（连接）
    for input_node in input_nodes:
        for output_node in output_nodes:
            G.add_edge(input_node, output_node)
    
    # 设置节点位置
    pos = {}
    # 输入层位置
    for i, node in enumerate(input_nodes):
        pos[node] = (0, 4-i*2)
    
    # 输出层位置
    for i, node in enumerate(output_nodes):
        pos[node] = (2, 4.5-i*1)
    
    # 绘制网络结构
    nx.draw_networkx_nodes(G, pos, nodelist=input_nodes, node_color='lightblue', 
                          node_size=800, ax=ax1)
    nx.draw_networkx_nodes(G, pos, nodelist=output_nodes, node_color='lightgreen', 
                          node_size=800, ax=ax1)
    nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax1)
    nx.draw_networkx_labels(G, pos, ax=ax1, font_size=10)
    
    # 添加层标签
    ax1.text(0, 6, '输入层\n(784个神经元)', ha='center', fontsize=12, fontweight='bold')
    ax1.text(2, 6, '输出层\n(10个神经元)', ha='center', fontsize=12, fontweight='bold')
    
    # 添加权重矩阵可视化
    ax1.text(1, 2, '权重矩阵 W\n(784×10)', ha='center', fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
    
    ax1.set_xlim(-0.5, 2.5)
    ax1.set_ylim(-1, 7)
    ax1.set_title('Softmax回归网络结构图', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # 2. 参数维度可视化
    # 创建权重矩阵热力图
    W = torch.normal(0, 0.01, size=(28, 10))  # 简化的权重矩阵
    
    im = ax2.imshow(W.numpy(), cmap='RdBu', aspect='auto')
    ax2.set_xlabel('输出神经元 (10类)', fontsize=12)
    ax2.set_ylabel('输入特征 (28×28=784)', fontsize=12)
    ax2.set_title('权重矩阵 W 的维度可视化\n(784×10 = 7,840个参数)', fontsize=14, fontweight='bold')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('权重值', fontsize=10)
    
    # 添加文本标注
    ax2.text(5, 15, '每个输入像素\n连接到所有\n10个输出神经元', 
             ha='center', va='center', 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.7),
             fontsize=10)
    
    # 添加统计信息
    stats_text = f"""
    网络统计：
    • 输入维度：784 (28×28像素)
    • 输出类别：10
    • 总参数：7,850
      - 权重：7,840
      - 偏置：10
    • 激活函数：Softmax
    • 损失函数：交叉熵
    """
    
    fig.text(0.02, 0.02, stats_text, fontsize=11, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('softmax_network_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    visualize_softmax_network()