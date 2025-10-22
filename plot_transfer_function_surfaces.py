import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from ripple_calculation import calculate_transfer_function

# --- Matplotlib 字体与渲染设置 ---
# 全局字体：英文 Times New Roman，中文回退 SimHei；避免负号方块
rcParams['font.family'] = 'Times New Roman'
rcParams['font.sans-serif'] = ['Times New Roman', 'SimHei']
rcParams['axes.unicode_minus'] = False
rcParams['font.size'] = 20
rcParams['axes.titlesize'] = 32
rcParams['axes.labelsize'] = 24
rcParams['xtick.labelsize'] = 20
rcParams['ytick.labelsize'] = 20
rcParams['legend.fontsize'] = 20
rcParams['mathtext.fontset'] = 'custom'
rcParams['mathtext.rm'] = 'Times New Roman'
rcParams['mathtext.it'] = 'Times New Roman:italic'
rcParams['mathtext.bf'] = 'Times New Roman:bold'
rcParams['mathtext.tt'] = 'Times New Roman'


# --- OS 定制的中文回退（在保持 Times New Roman 的前提下） ---
try:
    rcParams['font.sans-serif'] = ['Times New Roman', 'Songti SC']  # macOS
    rcParams['axes.unicode_minus'] = False
except:
    try:
        rcParams['font.sans-serif'] = ['Times New Roman', 'SimHei']  # Windows
        rcParams['axes.unicode_minus'] = False
    except:
        pass

def format_value_with_unit(value, param_name):
    """Formats a parameter value into a human-readable string with units."""
    if param_name == 't':
        return f"{value * 1e6:.1f} μs"
    if param_name == 't1':
        return f"{value * 1e3:.1f} ms"
    if param_name in 't2':
        return f"{value:.1f} s"
    if param_name == 'Cx':
        if value < 1e-6:
            return f"{value * 1e9:.0f} nF"
        return f"{value * 1e6:.2f} μF"
    if param_name in ['Cf', 'Cd']:
        return f"{value * 1e12:.1f} pF"
    if param_name in ['Rb', 'Rf']:
        return f"{value / 1e6:.0f} MΩ"
    if param_name == 'A0':
        return f"{value:.0f}"
    return f"{value:.2e}"

def plot_parameter_sensitivity_2d(param_name, pretty_name, param_values_list, frequencies, default_params):
    """
    Plots the FWHM/V_in ratio vs. frequency for several
    values of a given parameter on a single 2D plot.
    """
    plt.figure(figsize=(10, 6))
    q = 1.6e-19  # Elementary charge

    for p_val in param_values_list:
        y_values = []
        for f in frequencies:
            params = default_params.copy()
            params[param_name] = p_val
            
            H_mag = np.abs(calculate_transfer_function(f, **params))
            
            # Get current Cf for the conversion factor
            current_Cf = params['Cf']
            
            # Convert |H(f)| to FWHM/V_in in keV/V
            # The A0 gain in the transfer function and in the conversion factor cancels out,
            # so the FWHM/V_in ratio is independent of A0. We use the default A0 here.
            conversion_factor = (8.5 * 2.718 / (38 * 2.718) * current_Cf / q / 1000)
            y_value = H_mag * conversion_factor
            y_values.append(y_value)

        # Plot the curve with formatted label
        label_value = format_value_with_unit(p_val, param_name)
        plt.loglog(frequencies, y_values, label=f'{pretty_name} = {label_value}')

    plt.title(f'Sensitivity to Parameter: {pretty_name}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('FWHM / $V_{in}$ (keV/V)')
    plt.grid(True, which="both", ls="--")
    plt.legend(title=f'{pretty_name} values')
    plt.show()

if __name__ == "__main__":
    # Default parameter values from ripple_calculation.py
    default_params = {
        't': 2.8e-6,
        't1': 1.3e-3,
        't2': 0.01,
        'Cx': 0.1e-6,
        'Cf': 3e-12,
        'Cd': 2e-12
    }

    # Map for pretty printing parameter names with LaTeX
    param_pretty_names = {
        't': r'$\tau$',
        't1': r'$R_1C_1$',
        't2': r'$R_2C_2$',
        'Cx': r'$C_x$',
        'Cf': r'$C_f$',
        'Cd': r'$C_d$'
    }

    # Define frequency range
    frequencies = np.logspace(1, 7, 500)  # 10 Hz to 10 MHz

    # Define at least 5 discrete values for each parameter to plot
    params_to_plot = {
        't': np.logspace(-7, -4, 5),
        't1': np.logspace(-4, -2, 5),
        't2': np.logspace(-2, -0, 5),
        'Cx': np.logspace(-7, -5, 5),
        'Cf': np.logspace(-13, -11, 5),
        'Cd': np.logspace(-12, -10, 5)
    }

    # Generate and display a plot for each parameter
    for name, values in params_to_plot.items():
        pretty_name = param_pretty_names.get(name, name)
        plot_parameter_sensitivity_2d(name, pretty_name, values, frequencies, default_params)
