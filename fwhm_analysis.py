import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 全局字体设置：英文 Times New Roman，中文回退 SimHei；避免负号变方块
rcParams['font.family'] = 'Times New Roman'
rcParams['font.sans-serif'] = ['Times New Roman', 'SimHei']
rcParams['axes.unicode_minus'] = False
rcParams['font.size'] = 12
rcParams['axes.titlesize'] = 18
rcParams['axes.labelsize'] = 15
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
rcParams['legend.fontsize'] = 12
rcParams['mathtext.fontset'] = 'custom'
rcParams['mathtext.rm'] = 'Times New Roman'
rcParams['mathtext.it'] = 'Times New Roman:italic'
rcParams['mathtext.bf'] = 'Times New Roman:bold'
rcParams['mathtext.tt'] = 'Times New Roman'

def analyze_fwhm_change(v=1.0, v_conv_range=np.logspace(-1, 1, 50)):
    """
    计算FWHM的相对差值随 (v_conv/v)^2 变化的函数。
    相对差值定义为 (FWHM_sine_conv - FWHM_gauss_conv) / FWHM_gauss_orig。

    参数:
    v (float): 原始信号噪声的标准差。
    v_conv_range (array): 用于卷积的高斯核的标准差范围。

    返回:
    tuple: (v_ratios, fwhm_diff_ratios)
           v_ratios: (v_conv / v)^2 的数组
           fwhm_diff_ratios: FWHM相对差值的数组
    """
    v_ratios = []
    fwhm_diff_ratios = []

    # 原始高斯噪声的FWHM (作为参考基准)
    fwhm_gaussian_original = 2 * np.sqrt(2 * np.log(2)) * v
    A = v * np.sqrt(2)  # 正弦波幅值

    # 为所有卷积计算设置一个足够宽的统一网格
    max_v_conv = v_conv_range[-1]
    x = np.linspace(-4 * (v + max_v_conv), 4 * (v + max_v_conv), 8000)
    dx = x[1] - x[0]

    # 在此网格上定义正弦波的概率密度函数
    sine_pdf_template = np.full_like(x, np.nan)
    sine_domain_mask = np.abs(x) < A
    x_sine = x[sine_domain_mask]
    with np.errstate(divide='ignore'):
        sine_pdf_template[sine_domain_mask] = 1 / (np.pi * np.sqrt(A**2 - x_sine**2))
    
    # 将NaN和inf替换为0，以便进行卷积
    sine_pdf_for_conv = np.nan_to_num(sine_pdf_template)
    # 数值归一化
    sine_pdf_for_conv /= np.trapz(sine_pdf_for_conv, x)

    for v_conv in v_conv_range:
        # 1. 计算 高斯*高斯 的 FWHM (解析解)
        v_total_gauss = np.sqrt(v**2 + v_conv**2)
        fwhm_gauss_conv = 2 * np.sqrt(2 * np.log(2)) * v_total_gauss

        # 2. 计算 正弦*高斯 的 FWHM (数值解)
        # 定义卷积核
        conv_kernel = (1 / (v_conv * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x / v_conv)**2)
        conv_kernel /= np.trapz(conv_kernel, x) # 归一化

        # 执行卷积
        sine_convolved = np.convolve(sine_pdf_for_conv, conv_kernel, mode='same') * dx
        
        # 使用插值更精确地寻找FWHM，以提高稳定性
        max_val = np.max(sine_convolved)
        half_max = max_val / 2.0
        
        # 找到穿过半高点的所有索引
        indices = np.where(np.diff(np.sign(sine_convolved - half_max)))[0]
        
        fwhm_sine_conv = np.nan
        # 确保我们有一对点来定义宽度
        if len(indices) >= 2:
            # 左侧点
            idx1 = indices[0]
            x1, y1 = x[idx1], sine_convolved[idx1]
            x2, y2 = x[idx1 + 1], sine_convolved[idx1 + 1]
            x_left = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1)
            
            # 右侧点
            idx2 = indices[-1]
            x1, y1 = x[idx2], sine_convolved[idx2]
            x2, y2 = x[idx2 + 1], sine_convolved[idx2 + 1]
            x_right = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1)
            
            fwhm_sine_conv = x_right - x_left

        if not np.isnan(fwhm_sine_conv):
            # 3. 计算差值和比值
            fwhm_difference = fwhm_sine_conv - fwhm_gauss_conv
            fwhm_diff_ratio = fwhm_difference / fwhm_gaussian_original
            
            # 4. 存储结果
            v_ratios.append((v_conv / v)**2)
            fwhm_diff_ratios.append(fwhm_diff_ratio)
            
    return np.array(v_ratios), np.array(fwhm_diff_ratios)

def plot_fwhm_analysis(v_ratios, fwhm_diff_ratios):
    """
    绘制 FWHM 相对差值 vs. (v_conv/v)^2 的关系图。
    """
    plt.figure(figsize=(10, 6))
    plt.tick_params(direction='in')
    plt.plot(v_ratios, fwhm_diff_ratios, 'ko-', markersize=4)
    
    plt.title('Relative FWHM Broadening: Sine vs. Gaussian Noise')
    plt.xlabel(r'$v_\mathrm{{other}}^2 / v_\mathrm{{ripple}}^2$')
    plt.ylabel('$\mathrm{(FWHM_{sine} - FWHM_{gauss}) / FWHM_{gauss, orig}}$')
    plt.grid(True)
    plt.xscale('log') # v_conv/v 跨越数量级，使用对数坐标轴更佳
    plt.legend()
    plt.tight_layout()
    plt.xlim(1e-2, 1e4)
    plt.show()

if __name__ == '__main__':
    # 设置参数
    v_param = 1.0
    # v_conv 从 0.1*v 到 100*v, 增加点数以获得更平滑的曲线
    v_conv_values = np.logspace(-1, 2, 100) * v_param 
    
    # 执行计算并绘图
    ratios_x, ratios_y = analyze_fwhm_change(v=v_param, v_conv_range=v_conv_values)
    plot_fwhm_analysis(ratios_x, ratios_y)
