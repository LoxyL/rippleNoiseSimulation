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
import sys
import os

# Add the sys_simulate directory to the Python path to find the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'sys_simulate')))

try:
    from analyze_spectrum import SpectrumAnalyzer
except ImportError as e:
    print("Error: Could not import SpectrumAnalyzer.")
    print("Please ensure 'sys_simulate/analyze_spectrum.py' exists and the script is run from the project root.")
    print(f"Original error: {e}")
    sys.exit(1)

def plot_waveforms(file_path, threshold=1.0, time_limit_ms=5.0):
    """
    Plots the input and output waveforms from a simulation file.

    :param file_path: Path to the simulation data CSV file.
    :param threshold: Voltage threshold for peak detection.
    :param time_limit_ms: The time limit in milliseconds for the x-axis.
    """
    if not os.path.exists(file_path):
        print(f"Error: Data file not found at '{file_path}'")
        return

    # Use SpectrumAnalyzer to load data and find peaks
    analyzer = SpectrumAnalyzer(threshold=threshold)
    analyzer.process_file(file_path) # This loads data and finds peaks

    # Create the plot with two subplots, sharing the x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle('Input and Output Waveforms Analysis', fontsize=24)

    time_limit_s = time_limit_ms / 1000.0

    # --- Top Subplot: Output Waveform ---
    ax1.plot(analyzer.time_s, analyzer.vout_V, alpha=0.9, color='royalblue')
    if analyzer.peak_times is not None and len(analyzer.peak_times) > 0:
        # Filter peaks within the time limit for plotting
        mask = analyzer.peak_times <= time_limit_s
        ax1.plot(analyzer.peak_times[mask], analyzer.peak_voltages[mask], "x", markersize=8, color='crimson', label='Detected Peaks')
    ax1.axhline(y=analyzer.threshold, color='red', linestyle=':', label=f'Threshold ({analyzer.threshold}V)')
    ax1.set_title('Detector Output Response (Vout)')
    ax1.set_ylabel('Output Voltage / V')
    ax1.legend()
    ax1.grid(True)
    ax1.tick_params(direction='in')

    # --- Bottom Subplot: Input Current Pulse (Idealized) ---
    # The CSV 'vin' is the voltage at the CSA input, not the current pulse itself.
    # Here, we generate an ideal pulse train based on the known simulation parameters
    # to accurately represent the input current.
    
    # Parameters from my_circuit.cir and main.py
    amplitude_ua = 0.4  # 0.4 uA
    frequency_hz = 1000 # 1 kHz
    period_s = 1.0 / frequency_hz
    pw_s = 1e-6 # 1 us pulse width as described in documentation
    tr_s = 1e-6 # Rise time from .cir file
    delay_s = 0.1e-3 # Delay time from .cir file
    
    # Create the theoretical pulse train
    # This is a simplified way to draw a pulse train with sharp edges for visualization
    t = analyzer.time_s
    pulse = np.zeros_like(t)
    for i in range(int(t[-1] / period_s) + 1):
        start = i * period_s + delay_s
        # Create a trapezoidal pulse for each period
        mask = (t >= start) & (t <= start + pw_s + 2 * tr_s)
        pulse[mask] = np.interp(t[mask], 
                                [start, start + tr_s, start + tr_s + pw_s, start + 2 * tr_s + pw_s], 
                                [0, amplitude_ua, amplitude_ua, 0])

    ax2.plot(t, pulse, label='Ideal Input Current', color='darkorange')
    ax2.set_title('Input Pulse Current (Id)')
    ax2.set_xlabel('Time / s')
    ax2.set_ylabel('Current / μA')
    ax2.legend()
    ax2.grid(True)
    ax2.tick_params(direction='in')

    # Set the shared x-axis limit
    ax2.set_xlim(0, time_limit_s)
    
    # Adjust layout and show the plot
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == '__main__':
    # Assuming the default simulation output file is in the 'sys_simulate' directory
    default_file = './simulation_output.csv'
    plot_waveforms(file_path=default_file, threshold=1.0, time_limit_ms=5.0)
