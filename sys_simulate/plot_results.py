import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 全局字体设置：英文 Times New Roman，中文回退 SimHei；避免负号变方块
rcParams['font.family'] = 'Times New Roman'
rcParams['font.sans-serif'] = ['Times New Roman', 'SimHei']
rcParams['axes.unicode_minus'] = False
rcParams['font.size'] = 12
rcParams['axes.titlesize'] = 18
rcParams['axes.labelsize'] = 18
rcParams['xtick.labelsize'] = 15
rcParams['ytick.labelsize'] = 15
rcParams['legend.fontsize'] = 12
rcParams['mathtext.fontset'] = 'custom'
rcParams['mathtext.rm'] = 'Times New Roman'
rcParams['mathtext.it'] = 'Times New Roman:italic'
rcParams['mathtext.bf'] = 'Times New Roman:bold'
rcParams['mathtext.tt'] = 'Times New Roman'
import numpy as np

def plot_sweep_results(csv_path='fwhm_sweep_results.csv'):
    """
    Loads data from the fwhm_sweep_results.csv file and plots the results
    in separate subplots for each amplitude.
    """
    try:
        data = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The data file '{csv_path}' was not found.")
        print("Please run 'main.py' first to generate the simulation results.")
        return

    # Get the unique amplitudes from the data, and sort them
    amplitudes = np.sort(data['ripple_amplitude_V'].unique())
    num_amps = len(amplitudes)

    plt.style.use('seaborn-v0_8-whitegrid')
    # seaborn 可能覆盖字号，这里再次设置固定的数值字号
    rcParams['font.family'] = 'Times New Roman'
    rcParams['font.sans-serif'] = ['Times New Roman', 'SimHei']
    rcParams['axes.unicode_minus'] = False
    rcParams['font.size'] = 12
    rcParams['axes.titlesize'] = 18
    rcParams['axes.labelsize'] = 18
    rcParams['xtick.labelsize'] = 15
    rcParams['ytick.labelsize'] = 15
    rcParams['legend.fontsize'] = 12
    rcParams['mathtext.fontset'] = 'custom'
    rcParams['mathtext.rm'] = 'Times New Roman'
    rcParams['mathtext.it'] = 'Times New Roman:italic'
    rcParams['mathtext.bf'] = 'Times New Roman:bold'
    rcParams['mathtext.tt'] = 'Times New Roman'
    # Create a subplot for each amplitude, sharing the X-axis
    fig, axes = plt.subplots(num_amps, 1, figsize=(18, 4 * num_amps), sharex=True)
    
    # If there's only one amplitude, axes will not be an array, so we wrap it
    if num_amps == 1:
        axes = [axes]

    for i, amp in enumerate(amplitudes):
        ax = axes[i]
        amp_data = data[data['ripple_amplitude_V'] == amp].sort_values(by='ripple_frequency_Hz')

        # --- Plot Simulated and Theoretical FWHM ---
        ax.plot(amp_data['ripple_frequency_Hz'], 
                amp_data['net_simulated_fwhm_keV'],
                color='royalblue',
                marker='o',
                linestyle='-',
                label='Simulated')

        ax.plot(amp_data['ripple_frequency_Hz'],
                amp_data['theoretical_ripple_fwhm_keV'],
                color='darkorange',
                marker='.',
                linestyle='--',
                label='Theoretical')
        
        # --- Plot Relative Error on Twin Axis ---
        ax_err = ax.twinx()
        # Calculate relative error: (Simulated - Theoretical) / Theoretical * 100
        rel_err = (amp_data['net_simulated_fwhm_keV'] - amp_data['theoretical_ripple_fwhm_keV']) / amp_data['theoretical_ripple_fwhm_keV'] * 100
        
        # Fill positive error with blue, negative with red
        ax_err.fill_between(amp_data['ripple_frequency_Hz'], rel_err, 0, where=(rel_err >= 0), facecolor='blue', alpha=0.15, interpolate=True, label='Positive Error')
        ax_err.fill_between(amp_data['ripple_frequency_Hz'], rel_err, 0, where=(rel_err < 0), facecolor='red', alpha=0.15, interpolate=True, label='Negative Error')
        
        # Add 0% reference line
        ax_err.axhline(0, color='gray', linestyle='--', linewidth=1)
        
        # Set limits for relative error axis
        ax_err.set_ylim(-60, 20)
        ax_err.set_ylabel('Relative Error [%]')
        ax_err.tick_params(direction='in')
        
        ax.set_title(f'Ripple Amplitude = {amp:.3f} V')
        ax.set_ylabel('Net Ripple FWHM [keV]')
        ax.set_xscale('log')
        ax.set_yscale('linear') # Change Y-axis to linear scale
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.tick_params(direction='in')
        ax.legend(loc='upper center') # Moved to avoid overlapping with right y-axis ticks
        
        # Adjust axis limits
        ax.margins(x=0) # Remove horizontal padding
        ax.set_ylim(bottom=0) # Start Y-axis from 0

    # --- Final Plot Formatting ---
    # Add a single X-axis label to the last subplot
    axes[-1].set_xlabel('Ripple Frequency [Hz]')
    
    fig.suptitle('Simulated vs. Theoretical FWHM of Ripple Noise', fontsize=24, y=1.0)
    # fig.set_dpi(600)
    plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust rect to make space for suptitle
    plt.savefig('./circ_sim_results.png', dpi=600, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    plot_sweep_results()
