import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 全局字体设置：英文 Times New Roman，中文回退 SimHei；避免负号变方块
rcParams['font.family'] = 'Times New Roman'
rcParams['font.sans-serif'] = ['Times New Roman', 'SimHei']
rcParams['axes.unicode_minus'] = False
rcParams['font.size'] = 20
rcParams['axes.titlesize'] = 32
rcParams['axes.labelsize'] = 24
rcParams['xtick.labelsize'] = 20
rcParams['ytick.labelsize'] = 20
rcParams['legend.fontsize'] = 20

def simulate_spectrum(peaks, amplitudes, fwhm, energy_range, num_points):
    """
    模拟探测器能谱。

    参数:
    peaks (list of float): 理想能谱峰的中心位置列表。
    amplitudes (list of float): 对应每个峰的相对幅度列表。
    fwhm (float): 高斯噪声的半高全宽 (Full Width at Half Maximum)。
    energy_range (tuple of float): 能谱的能量范围 (min_energy, max_energy)。
    num_points (int): 在能量范围内生成的点数。

    返回:
    tuple: (energy_bins, spectrum)
        - energy_bins (np.array): 能量轴。
        - spectrum (np.array): 模拟的谱线强度。
    """
    # 1. 从 FWHM 计算高斯分布的标准差 (sigma)
    # FWHM = 2 * sqrt(2 * ln(2)) * sigma
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))

    # 2. 创建能量轴
    energy_bins = np.linspace(energy_range[0], energy_range[1], num_points)

    # 3. 初始化能谱
    spectrum = np.zeros_like(energy_bins)

    # 高斯函数
    def gaussian(x, mu, sigma, amplitude):
        return amplitude * np.exp(-0.5 * ((x - mu) / sigma)**2)

    # 4. 将每个峰的高斯展宽贡献加到总能谱中
    for peak, amp in zip(peaks, amplitudes):
        spectrum += gaussian(energy_bins, peak, sigma, amp)

    return energy_bins, spectrum, sigma

def plot_spectrum(energy_bins, spectrum, peaks, amplitudes, fwhm, sigma):
    """
    绘制能谱图, 分为三个子图: 理想能谱、噪声分布和最终能谱。

    参数:
    energy_bins (np.array): 能量轴。
    spectrum (np.array): 模拟的谱线强度。
    peaks (list of float): 理想峰位。
    amplitudes (list of float): 理想峰的相对幅度。
    fwhm (float): FWHM值。
    sigma (float): 高斯噪声的标准差。
    """
    fig, axs = plt.subplots(3, 1, figsize=(12, 18), sharex=True)

    # --- 子图1: 理想能谱 ---
    axs[0].stem(peaks, amplitudes, linefmt='b-', markerfmt='bo', basefmt=' ')
    axs[0].set_title('Ideal Spectrum (Dirac Delta Functions)')
    axs[0].set_ylabel('Intensity (Arbitrary Units)')
    axs[0].grid(True, which='both', linestyle='--', linewidth=0.5)

    # --- 子图2: 噪声分布 ---
    # 将高斯分布绘制在图表中心以便可视化
    plot_center = (energy_bins[0] + energy_bins[-1]) / 2
    gaussian_pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((energy_bins - plot_center) / sigma)**2)
    axs[1].plot(energy_bins, gaussian_pdf, 'g-')
    axs[1].set_title('Noise Distribution (Centered for Visualization)')
    axs[1].set_ylabel('Probability Density')
    axs[1].grid(True, which='both', linestyle='--', linewidth=0.5)

    # 添加FWHM和Sigma标注
    max_pdf = np.max(gaussian_pdf)
    half_max_pdf = max_pdf / 2.0
    fwhm_left = plot_center - fwhm / 2
    fwhm_right = plot_center + fwhm / 2
    
    # 绘制贯穿全图的红色水平半高线, 并标注半高值
    axs[1].axhline(y=half_max_pdf, color='r', linestyle='--', label=f'Half Maximum ({half_max_pdf:.4f})')

    # 绘制贯穿全图的红色垂直FWHM边界线
    axs[1].axvline(x=fwhm_left, color='r', linestyle=':', alpha=0.7)
    axs[1].axvline(x=fwhm_right, color='r', linestyle=':', alpha=0.7)

    # 添加文本框显示FWHM和sigma的值
    text_str = f'FWHM = {fwhm}\nσ ≈ {sigma:.2f}'
    axs[1].text(0.95, 0.95, text_str, transform=axs[1].transAxes, fontsize=24,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    
    # 显示图例 (包含半高线标签)
    axs[1].legend(loc='upper left')

    # --- 子图3: 最终能谱 ---
    axs[2].plot(energy_bins, spectrum, label='Final Simulated Spectrum')
    
    # 标记理想峰位
    for peak in peaks:
        axs[2].axvline(x=peak, color='r', linestyle='--', alpha=0.6, label=f'Ideal Peak at {peak}' if peak == peaks[0] else "")

    axs[2].set_title('Final Spectrum (Ideal Spectrum convolved with Noise)')
    axs[2].set_xlabel('Counts')
    axs[2].set_ylabel('Intensity (Arbitrary Units)')
    axs[2].grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # 移除重复的图例标签
    handles, labels = axs[2].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axs[2].legend(by_label.values(), by_label.keys())
    
    # 设置所有子图共享的X轴范围
    plt.xlim(energy_bins[0], energy_bins[-1])
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # --- 输入参数 ---
    # 理想峰位
    IDEAL_PEAKS = [5000, 5500, 7100, 7200]
    
    # 为了让能谱更真实, 我们给每个峰设置一个相对强度
    PEAK_AMPLITUDES = [1.0, 0.85, 0.5, 0.4]

    # 高斯噪声的半高全宽
    FWHM = 100

    # 能谱的显示范围
    ENERGY_RANGE = (4500, 8000)

    # 模拟的点数，点数越多曲线越平滑
    NUM_POINTS = 2000

    # --- 模拟和绘图 ---
    energy_bins, spectrum, sigma = simulate_spectrum(
        peaks=IDEAL_PEAKS,
        amplitudes=PEAK_AMPLITUDES,
        fwhm=FWHM,
        energy_range=ENERGY_RANGE,
        num_points=NUM_POINTS
    )

    plot_spectrum(energy_bins, spectrum, IDEAL_PEAKS, PEAK_AMPLITUDES, FWHM, sigma)
