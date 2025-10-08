# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from plot_utils import plot_pole_zero, add_pole_zero_labels
from ripple_calculation import calculate_transfer_function, t, t1, t2, Cx, Cf, Cd, Rb, Rf, A0, q, V_in_peak, f_ripple

# 配置matplotlib以支持中文显示
try:
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"无法设置中文字体，图形中的中文可能无法正常显示: {e}")
    print("请尝试安装 'SimHei' 或其他中文字体。")


def analyze_and_plot():
    # --- 计算指定纹波频率下的输出 ---
    H_at_ripple_freq = calculate_transfer_function(f_ripple, verbose=True)
    H_magnitude = np.abs(H_at_ripple_freq)
    H_phase_deg = np.angle(H_at_ripple_freq, deg=True)
    V_out_peak = V_in_peak * H_magnitude
    V_out_peak_mV = V_out_peak * 1000

    # --- 打印计算结果 ---
    print("--- 纹波响应计算结果 ---")
    print(f"输入纹波峰值: {V_in_peak * 1000:.1f} mV")
    print(f"输入纹波频率: {f_ripple} Hz")
    print("-" * 25)
    print(f"在 {f_ripple} Hz 处的传递函数幅值 |H(f)|: {H_magnitude:.4g}")
    print(f"在 {f_ripple} Hz 处的传递函数相位: {H_phase_deg:.2f} 度")
    print("-" * 25)
    print(f"输出纹波电压峰值: {V_out_peak_mV*1e6:.4f} nV")
    print(f"等效FWHM: {8.5*2.718/A0*V_out_peak*Cf/q/1000:.4f} keV")
    print("=" * 25)

    # --- 计算各个传递函数的零极点 ---
    poles_H1 = np.array([-1/t1, -1/t2])
    zeros_H1 = np.array([])
    poles_Hg = np.array([-1/t, -1/t])
    zeros_Hg = np.array([0])
    poles_H2_p1 = np.array([-1/((Cx+Cd)*Rb)])
    zeros_H2_p1 = np.array([0])
    poles_H2_p2 = np.array([-1/(Cf*Rf)])
    zeros_H2_p2 = np.array([0])

    # --- 为零极点定义符号标签 ---
    pole_labels_H1 = [r'$-1/\tau_1=1/R_1C_1$', r'$-1/\tau_2=1/R_2C_2$']
    zero_labels_H1 = []
    pole_labels_Hg = [r'$-1/\tau$', r'$-1/\tau$ (double)']
    zero_labels_Hg = [r'$0$']
    pole_labels_H2_p1 = [r'$-1/((C_x+C_d)R_b)$']
    zero_labels_H2_p1 = [r'$0$']
    pole_labels_H2_p2 = [r'$-1/(C_f R_f)$']
    zero_labels_H2_p2 = [r'$0$']

    # --- 创建图形 ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Pole-Zero Plots of Transfer Functions', fontsize=18)
    plot_pole_zero(axes[0, 0], poles_H1, zeros_H1, '$H_1(s)$', pole_labels=pole_labels_H1, zero_labels=zero_labels_H1)
    plot_pole_zero(axes[0, 1], poles_Hg, zeros_Hg, '$H_g(s)$', pole_labels=pole_labels_Hg, zero_labels=zero_labels_Hg)
    plot_pole_zero(axes[1, 0], poles_H2_p1, zeros_H2_p1, '$H_2(s)$ Part 1', pole_labels=pole_labels_H2_p1, zero_labels=zero_labels_H2_p1)
    plot_pole_zero(axes[1, 1], poles_H2_p2, zeros_H2_p2, '$H_2(s)$ Part 2', pole_labels=pole_labels_H2_p2, zero_labels=zero_labels_H2_p2)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # --- 汇总零极点图 & 系统波特图 ---
    all_poles = np.concatenate([poles_H1, poles_Hg, poles_H2_p1, poles_H2_p2])
    all_zeros = np.concatenate([zeros_H1, zeros_Hg, zeros_H2_p1, zeros_H2_p2])
    all_points = np.concatenate([all_poles, all_zeros])

    # --- 汇总所有符号标签 ---
    all_pole_labels = pole_labels_H1 + pole_labels_Hg + pole_labels_H2_p1 + pole_labels_H2_p2
    all_zero_labels = zero_labels_H1 + zero_labels_Hg + zero_labels_H2_p1 + zero_labels_H2_p2

    # --- 为总图中拥挤的标签定义自定义偏移量 (x_offset, y_offset) in points ---
    # 顺序必须与 all_poles 和 all_pole_labels 严格对应:
    # H1_p1, H1_p2, Hg_p1, Hg_p2, H2_p1, H2_p2
    # -769,  -1,    -357k, -357k, -0.1,  -1667
    custom_pole_offsets = [
        (10, 15),    # p at -769 (-1/t1): 默认向上
        (10, -25),   # p at -1 (-1/t2):   向下移动以避开 -0.1
        None,        # p at -357k (-1/t): 使用默认堆叠逻辑
        None,        # p at -357k (-1/t): 使用默认堆叠逻辑
        (10, 15),    # p at -0.1:         默认向上
        (10, -40),   # p at -1667:        向下移动以避开 -769
    ]

    fig_analysis = plt.figure(figsize=(18, 9))
    gs = fig_analysis.add_gridspec(1, 2, width_ratios=[1, 1])
    fig_analysis.suptitle('System Analysis: Pole-Zero Map and Bode Plot', fontsize=18)

    ax_pz = fig_analysis.add_subplot(gs[0, 0])

    frequencies_hz = np.logspace(-2, 7, 2000)
    omegas_rad_s = frequencies_hz * 2 * np.pi
    H_values = np.array([calculate_transfer_function(f, verbose=False) for f in frequencies_hz])
    H_magnitudes_db = 20 * np.log10(np.abs(H_values) + 1e-20)

    ax_mag = fig_analysis.add_subplot(gs[0, 1])
    ax_mag.plot(H_magnitudes_db, omegas_rad_s)
    ax_mag.set_xscale('linear')
    ax_mag.set_yscale('log')
    ax_mag.set_title('Bode Plot (Magnitude)')
    ax_mag.set_xlabel('Magnitude (dB)')
    ax_mag.set_ylabel('Frequency (rad/s)')
    ax_mag.grid(True, which="both", ls="--", linewidth=0.5)

    # --- 计算与零极点图匹配的Y轴范围 ---
    # 严格复用零极点图的上限计算逻辑，且不改动零极点图
    max_re = -np.min(np.real(all_points))
    offset_top = 2
    y_max_for_bode = max_re * 10**offset_top
    
    # 为了与零极点图的'symlog'标度(linthresh=1)在视觉上对齐,
    # 我们将波特图的对数Y轴下限也设为1
    y_min_for_bode = 10**-1.5

    # 仅设置波特图的Y轴范围
    ax_mag.set_ylim(y_min_for_bode, y_max_for_bode)

    corner_freqs_poles = np.abs(np.real(all_poles[all_poles != 0]))
    corner_freqs_zeros = np.abs(np.real(all_zeros[all_zeros != 0]))
    all_corner_freqs = np.sort(np.unique(np.concatenate([corner_freqs_poles, corner_freqs_zeros])))

    num_zeros_at_origin = np.sum(np.abs(all_zeros) < 1e-9)
    num_poles_at_origin = np.sum(np.abs(all_poles) < 1e-9)
    initial_slope_db = (num_zeros_at_origin - num_poles_at_origin) * 20

    # 使用一个固定的、足够低的频率作为参考点来计算K_db，以避免数值不稳定
    ref_freq_hz = 1e-2 
    ref_H = calculate_transfer_function(ref_freq_hz, verbose=False)
    ref_mag_db = 20 * np.log10(np.abs(ref_H) + 1e-20)
    ref_freq_rad = ref_freq_hz * 2 * np.pi
    K_db = ref_mag_db - initial_slope_db * np.log10(ref_freq_rad)

    asymptote_freqs_rad = np.concatenate([[omegas_rad_s[0]], all_corner_freqs, [omegas_rad_s[-1]]])
    asymptote_mags_db = []
    current_slope_db = initial_slope_db
    current_mag_db = current_slope_db * np.log10(asymptote_freqs_rad[0]) + K_db
    asymptote_mags_db.append(current_mag_db)

    for i in range(len(all_corner_freqs)):
        mag_at_corner = current_slope_db * np.log10(all_corner_freqs[i]) + K_db
        asymptote_mags_db.append(mag_at_corner)
        num_poles_at_freq = np.sum(np.isclose(corner_freqs_poles, all_corner_freqs[i]))
        num_zeros_at_freq = np.sum(np.isclose(corner_freqs_zeros, all_corner_freqs[i]))
        current_slope_db += (num_zeros_at_freq - num_poles_at_freq) * 20
        K_db = mag_at_corner - current_slope_db * np.log10(all_corner_freqs[i])

    final_mag = current_slope_db * np.log10(asymptote_freqs_rad[-1]) + K_db
    asymptote_mags_db.append(final_mag)

    ax_mag.plot(asymptote_mags_db, asymptote_freqs_rad, 'r--', label='Asymptotic Approximation', linewidth=2)
    ax_mag.legend()

    ax_pz.scatter(np.real(all_poles), np.imag(all_poles), s=120, marker='x', color='r', lw=2, label='Poles', zorder=10)
    ax_pz.scatter(np.real(all_zeros), np.imag(all_zeros), s=120, marker='o', facecolors='none', edgecolors='b', lw=2, label='Zeros', zorder=10)
    
    # --- 为汇总图添加标签 ---
    add_pole_zero_labels(ax_pz, all_poles, all_pole_labels, 'darkred', custom_offsets=custom_pole_offsets)
    add_pole_zero_labels(ax_pz, all_zeros, all_zero_labels, 'darkblue')

    unique_magnitudes = np.unique(np.abs(np.concatenate([all_poles, all_zeros])))
    theta = np.linspace(0, 2 * np.pi, 200)
    for r in unique_magnitudes:
        if r > 1e-9:
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            ax_pz.plot(x, y, color='gray', linestyle=':', linewidth=1.2, zorder=5)
            ax_pz.axhline(y=r, color='coral', linestyle='--', linewidth=0.9, zorder=1)
            ax_pz.axhline(y=-r, color='coral', linestyle='--', linewidth=0.9, zorder=1)
            ax_mag.axhline(y=r, color='coral', linestyle='--', linewidth=0.9, zorder=1)

    ax_pz.set_title('Combined Pole-Zero Plot')
    ax_pz.set_xlabel('Real Axis (σ)')
    ax_pz.set_ylabel('Imaginary Axis (jω) [rad/s]')
    ax_pz.grid(True, which="both", ls="--", linewidth=0.5)
    ax_pz.axhline(0, color='black', lw=0.5)
    ax_pz.axvline(0, color='black', lw=0.5)
    ax_pz.legend()
    ax_pz.set_xscale('symlog', linthresh=1)
    ax_pz.set_yscale('symlog', linthresh=1)

    max_re = -np.min(np.real(all_points))
    offset_bottom = 6
    offset_top = 2
    x_min = -max_re * 10**offset_top
    x_max = max_re / 10**offset_bottom
    y_min = -max_re / 10**offset_bottom
    y_max = max_re * 10**offset_top
    ax_pz.set_xlim(x_min, x_max)
    ax_pz.set_ylim(y_min, y_max)
    ax_pz.set_aspect('equal', adjustable='box')

    fig_analysis.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

if __name__ == '__main__':
    analyze_and_plot()
