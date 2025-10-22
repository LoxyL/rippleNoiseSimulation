# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'Times New Roman'
rcParams['font.sans-serif'] = ['Times New Roman', 'SimHei']
rcParams['axes.unicode_minus'] = False
rcParams['font.size'] = 20
rcParams['axes.titlesize'] = 32
rcParams['axes.labelsize'] = 24
rcParams['xtick.labelsize'] = 20
rcParams['ytick.labelsize'] = 20
rcParams['legend.fontsize'] = 20

def add_pole_zero_labels(ax, points, labels, color, custom_offsets=None):
    """
    在给定的ax上为零极点添加符号标签，并处理重叠问题。
    使用 ax.annotate 以兼容对数坐标轴。
    允许传入自定义偏移量列表进行微调。
    """
    if labels is None:
        return
    
    # 使用一个字典来处理完全重叠点位的标签偏移
    plotted_locations = {}

    for i, point in enumerate(points):
        # 确定标签位置和偏移
        pos = (np.real(point), np.imag(point))
        offset_count = plotted_locations.get(pos, 0)
        
        # 检查是否有自定义偏移量
        if custom_offsets and i < len(custom_offsets) and custom_offsets[i] is not None:
            x_offset_points, y_offset_points = custom_offsets[i]
        else:
            # 默认逻辑
            x_offset_points = 10
            # 增加一个基础偏移量 (e.g., 15)，让所有标签的起始位置都更高
            y_offset_points = 15 + offset_count * 25 # 垂直堆叠重叠的标签

        ax.annotate(labels[i],
                    xy=pos,
                    xytext=(x_offset_points, y_offset_points),
                    textcoords='offset points',
                    fontsize=24,
                    color=color,
                    ha='left',
                    va='center',
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2'),
                    arrowprops=dict(arrowstyle='-', color='gray', connectionstyle='arc3,rad=0.1', shrinkB=5)
                   )
        
        plotted_locations[pos] = offset_count + 1

def plot_pole_zero(ax, poles, zeros, title, pole_labels=None, zero_labels=None):
    """
    在给定的ax上绘制零极点图，并可以选择性地添加符号标签。
    """
    # 绘制零点
    if len(zeros) > 0:
        ax.scatter(np.real(zeros), np.imag(zeros), s=100, marker='o', facecolors='none', edgecolors='b', lw=2, label='Zeros')
    # 绘制极点
    if len(poles) > 0:
        ax.scatter(np.real(poles), np.imag(poles), s=100, marker='x', color='r', lw=2, label='Poles')

    ax.set_title(title, fontsize=28)
    ax.set_xlabel('Real Axis (σ)', fontsize=20)
    ax.set_ylabel('Imaginary Axis (jω)', fontsize=20)
    ax.grid(True, which="both", ls="--", linewidth=0.5)
    ax.axhline(0, color='black', lw=0.5)
    ax.axvline(0, color='black', lw=0.5)
    ax.legend()
    
    # 动态调整坐标轴范围，确保横纵轴等长且原点在中心
    all_points = np.concatenate((poles, zeros)) if len(poles) > 0 or len(zeros) > 0 else np.array([0])
    
    # 找到所有点中实部和虚部绝对值的最大值
    max_coord = np.max(np.abs(np.concatenate((np.real(all_points), np.imag(all_points))))) if all_points.any() else 0

    # 设置一个边界，防止图像太拥挤
    margin = max_coord * 0.15 if max_coord > 0 else 1
    limit = max_coord + margin

    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    
    ax.set_aspect('equal', adjustable='box')

    # --- 添加符号标签 ---
    # 注意：这里的调用没有传递custom_offsets，因为它主要用于解决总图中不同点之间的拥挤问题
    add_pole_zero_labels(ax, poles, pole_labels, 'darkred')
    add_pole_zero_labels(ax, zeros, zero_labels, 'darkblue')
