import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 全局字体设置：英文 Times New Roman，中文回退 SimHei；避免负号变方块
rcParams['font.family'] = 'Times New Roman'
rcParams['font.sans-serif'] = ['Times New Roman', 'SimHei']
rcParams['axes.unicode_minus'] = False
rcParams['font.size'] = 12
rcParams['axes.titlesize'] = 15
rcParams['axes.labelsize'] = 12
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 12
rcParams['mathtext.fontset'] = 'custom'
rcParams['mathtext.rm'] = 'Times New Roman'
rcParams['mathtext.it'] = 'Times New Roman:italic'
rcParams['mathtext.bf'] = 'Times New Roman:bold'
rcParams['mathtext.tt'] = 'Times New Roman'
import matplotlib.gridspec as gridspec

def plot_noise_comparison(v=1.0, v_conv_list=[1.0, 2.0, 4.0]):
    """
    绘制噪声分布图。
    左侧为原始高斯和正弦噪声分布。
    右侧为三种不同标准差的高斯核对原始分布进行卷积后的结果。
    """
    fig = plt.figure(figsize=(18, 12))
    # 使用GridSpec创建1/3和2/3的布局
    gs = gridspec.GridSpec(3, 3, figure=fig)
    
    ax_left = fig.add_subplot(gs[:, 0])
    axes_right = [fig.add_subplot(gs[i, 1:]) for i in range(len(v_conv_list))]

    # --- 左侧图: 原始分布 (v=1.0) ---
    x_left = np.linspace(-5 * v, 5 * v, 2000)
    
    # 原始高斯噪声
    gaussian_pdf = (1 / (v * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x_left / v)**2)
    
    # 原始正弦波噪声
    A = v * np.sqrt(2)
    sine_pdf = np.full_like(x_left, np.nan)
    sine_domain_mask = np.abs(x_left) < A
    x_sine_left = x_left[sine_domain_mask]
    with np.errstate(divide='ignore'):
        sine_pdf[sine_domain_mask] = 1 / (np.pi * np.sqrt(A**2 - x_sine_left**2))

    ax_left.plot(x_left / v, gaussian_pdf * v, label=r'Gaussian Hypothesis (P=$v_{\text{ripple}}^2$)')
    ax_left.plot(x_left / v, sine_pdf * v, linestyle='--', label=r'Sine Hypothesis (P=$v_{\text{ripple}}^2$)')
    ax_left.axvline(x=A / v, color='r', linestyle=':')
    ax_left.axvline(x=-A / v, color='r', linestyle=':')
    ax_left.set_title('Ripple Distributions')
    ax_left.set_xlabel(r'Amplitude / $v_{\text{ripple}}$')
    ax_left.set_ylabel(r'Probability Density $\times v_{\text{ripple}}$')
    ax_left.tick_params(direction='in')
    ax_left.legend()
    ax_left.grid(True)
    ax_left.set_ylim(0, 0.5)

    # --- 右侧图: 卷积后的分布 ---
    max_v_conv = max(v_conv_list)
    x_right = np.linspace(-4 * (v + max_v_conv), 4 * (v + max_v_conv), 2000)
    dx_right = x_right[1] - x_right[0]

    # 需要在更宽的x范围上重新计算原始分布以进行卷积
    sine_pdf_right = np.full_like(x_right, np.nan)
    sine_domain_mask_right = np.abs(x_right) < A
    x_sine_right = x_right[sine_domain_mask_right]
    with np.errstate(divide='ignore'):
        sine_pdf_right[sine_domain_mask_right] = 1 / (np.pi * np.sqrt(A**2 - x_sine_right**2))
    sine_pdf_for_conv = np.nan_to_num(sine_pdf_right)

    for i, v_conv in enumerate(v_conv_list):
        ax = axes_right[i]

        # 定义卷积核
        conv_kernel = (1 / (v_conv * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x_right / v_conv)**2)
        conv_kernel /= np.trapz(conv_kernel, x_right)

        # 高斯 * 高斯
        v_total = np.sqrt(v**2 + v_conv**2)
        gaussian_convolved = (1 / (v_total * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x_right / v_total)**2)

        # 正弦波 * 高斯
        sine_convolved = np.convolve(sine_pdf_for_conv, conv_kernel, mode='same') * dx_right
        
        # --- FWHM 计算 ---
        fwhm_gaussian = 2 * np.sqrt(2 * np.log(2)) * v_total
        
        max_sine_conv = np.max(sine_convolved)
        half_max_sine = max_sine_conv / 2.0
        above_half_max_indices = np.where(sine_convolved > half_max_sine)[0]
        if above_half_max_indices.size > 0:
            x_left_fwhm = x_right[above_half_max_indices[0]]
            x_right_fwhm = x_right[above_half_max_indices[-1]]
            fwhm_sine = x_right_fwhm - x_left_fwhm
        else:
            fwhm_sine = np.nan

        # --- 绘图 ---
        ax.plot(x_right / v, gaussian_convolved * v, label=f'Gaussian Hypothesis (FWHM={fwhm_gaussian/v:.2f})')
        ax.plot(x_right / v, sine_convolved * v, linestyle='--', label=f'Sine Hypothesis (FWHM={fwhm_sine/v:.2f})')
        
        # FWHM 辅助线 - 高斯
        max_gauss_conv = np.max(gaussian_convolved)
        half_max_gauss = max_gauss_conv / 2.0
        gauss_fwhm_x_left = -fwhm_gaussian / 2
        gauss_fwhm_x_right = fwhm_gaussian / 2
        ax.hlines(half_max_gauss * v, gauss_fwhm_x_left / v, gauss_fwhm_x_right / v, color='b', linestyle=':')
        ax.axvline(x=gauss_fwhm_x_left / v, color='b', linestyle=':', alpha=0.7)
        ax.axvline(x=gauss_fwhm_x_right / v, color='b', linestyle=':', alpha=0.7)

        # FWHM 辅助线 - 正弦波
        if not np.isnan(fwhm_sine):
            ax.hlines(half_max_sine * v, x_left_fwhm / v, x_right_fwhm / v, color='g', linestyle=':')
            ax.axvline(x=x_left_fwhm / v, color='g', linestyle=':', alpha=0.7)
            ax.axvline(x=x_right_fwhm / v, color='g', linestyle=':', alpha=0.7)

        # FWHM 比值
        if not np.isnan(fwhm_sine) and fwhm_gaussian > 0:
            difference = (fwhm_sine - fwhm_gaussian) / v
            ax.text(0.95, 0.9, f'FWHM Diff (Sine-Gauss): {difference:.4f}',
                    horizontalalignment='right',
                    verticalalignment='top',
                    transform=ax.transAxes,
                    bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.5))
        
        ax.set_title(f'Noise Distribution ($v^2_\mathrm{{other}}/v^2_\mathrm{{ripple}}$ = {(v_conv/v)**2:.1f}'+')')
        ax.set_xlabel(r'Amplitude / $v_{\text{ripple}}$')
        ax.set_ylabel(r'Probability Density $\times v_{\text{ripple}}$')
        ax.legend()
        ax.tick_params(direction='in')
        ax.grid(True)

    # 确保右侧所有子图的x轴范围相同
    xlim = axes_right[0].get_xlim()
    for ax in axes_right[1:]:
        ax.set_xlim(xlim)

    plt.tight_layout()
    # 根据要求调整垂直间距
    plt.subplots_adjust(hspace=0.4)
    plt.show()

if __name__ == '__main__':
    plot_noise_comparison(v=1.0, v_conv_list=[1.0, 2.0, 4.0])
