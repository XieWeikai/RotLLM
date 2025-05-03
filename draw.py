import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import numpy as np
from skimage.measure import block_reduce


def plot_3d_bar_chart(data,
                      xlabel='tokens',
                      ylabel='dim',
                      zlabel='Value',
                      title='hidden_states',
                      bar_thickness_x=0.1,
                      bar_thickness_y=0.1,
                      alpha=0.6,
                      cmap_name='coolwarm',
                      elev=30, # 视角：仰角
                      azim=-45, # 视角：方位角
                      show=True # 是否显示图像
                     ):
    """
    绘制一个3D柱状图。

    Args:
        data (np.ndarray): 形状为 [num_time_steps, num_channels] 的二维数据数组。
                           数据值将决定柱子的高度和颜色。
        xlabel (str): X轴标签 (默认为 'Channel')。
        ylabel (str): Y轴标签 (默认为 'Time / Sample Index')。
        zlabel (str): Z轴标签 (默认为 'Value')。
        title (str): 图表标题 (默认为 '3D Bar Chart')。
        bar_thickness_x (float): 柱子在X方向的厚度 (默认为 0.1)。
        bar_thickness_y (float): 柱子在Y方向的厚度 (默认为 0.1)。
        alpha (float): 柱子的透明度 (0.0 完全透明，1.0 完全不透明，默认为 0.6)。
        cmap_name (str): 用于柱子着色的 Colormap 名称 (默认为 'coolwarm')。
                         例如: 'viridis', 'plasma', 'jet', 'hot' 等。
        elev (float): 3D图的仰角视角 (默认为 30)。
        azim (float): 3D图的方位角视角 (默认为 -45)。
    """
    data_transposed = data.T # 转置
    num_channels, num_time_steps = data_transposed.shape
    

    # 创建 x 和 y 的网格坐标
    # xpos 对应 channels (原代码的 x), ypos 对应 time_steps (原代码的 y)
    xpos, ypos = np.meshgrid(np.arange(num_channels), np.arange(num_time_steps))
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros_like(xpos) # 柱子的底部都在 z=0

    # 柱子的高度
    dz = data_transposed.flatten()

    # 柱子的宽度和深度
    # 使用传入的粗细参数
    dx = bar_thickness_x * np.ones_like(xpos)
    dy = bar_thickness_y * np.ones_like(ypos)

    # 创建 3D 图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 根据数据值确定柱子的颜色
    try:
        cmap = cm.get_cmap(cmap_name)
    except ValueError:
        print(f"警告: Colormap '{cmap_name}' 不存在，使用默认的 'coolwarm'。")
        cmap = cm.get_cmap('coolwarm')

    norm = plt.Normalize(dz.min(), dz.max())
    colors = cmap(norm(dz))

    # 绘制 3D 柱状图
    ax.bar3d(xpos, ypos, zpos,
             dx=dx, dy=dy, dz=dz,
             color=colors,
             shade=True,
             alpha=alpha) # 使用传入的 alpha 参数

    # 设置坐标轴标签和标题
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_zlabel(zlabel, fontsize=12)
    ax.set_title(title, fontsize=14)

    # 添加颜色条
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=zlabel) # 颜色条标签与 Z轴标签一致

    # 调整视角
    ax.view_init(elev=elev, azim=azim)

    # 显示图
    plt.tight_layout()
    plt.savefig('plot.png', dpi=150)
    if show:
        plt.show()
    
def plot_3d_bar_chart_fast(data, downsample_factor=10, **kwargs):
    """优化版：降采样加速"""
    data_downsampled = block_reduce(data, block_size=(downsample_factor, downsample_factor), func=np.mean)
    kwargs['bar_thickness_x'] = kwargs.get('bar_thickness_x', 0.1) * downsample_factor
    kwargs['bar_thickness_y'] = kwargs.get('bar_thickness_y', 0.1) * downsample_factor
    plot_3d_bar_chart(data_downsampled, **kwargs)


def plot_heat_map(data, xticklabels=200, yticklabels=200):
    plt.figure(figsize=(12, 8))  # 设置图像大小
    sns.heatmap(
        data,
        cmap="coolwarm",          # 颜色映射（可选：coolwarm, magma, viridis etc.）
        xticklabels=xticklabels,          # X 轴标签间隔
        yticklabels=yticklabels,           # Y 轴标签间隔
        cbar=True,               # 显示颜色条
    )
    plt.xlabel("Dimension")      # X 轴：维度
    plt.ylabel("Token Index")    # Y 轴：token 序号
    plt.title("hidden_states Heatmap")  # 标题
    plt.show()
