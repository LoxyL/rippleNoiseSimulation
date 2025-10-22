from math import e
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
import matplotlib.gridspec as gridspec
import sys
import os
try:
    from scipy.signal import find_peaks, peak_widths
    from scipy.stats import norm
except ImportError:
    print("错误: 缺少 'scipy' 库。")
    print("这是进行峰值和 FWHM 分析所必需的。")
    print("请在您的终端中运行 'pip install scipy' 来安装它。")
    sys.exit(1)

class SpectrumAnalyzer:
    """
    A class for analyzing pulse-height spectra.
    It can load simulation data from multiple files, find pulses, calculate FWHM, and visualize the results.
    """
    def __init__(self, threshold=1.0, config=None):
        """
        Initializes the analyzer.
        :param threshold: Voltage threshold (V) for event detection.
        :param config: An optional dictionary with physical and circuit parameters.
        """
        self.threshold = threshold
        self._load_config(config)
        self.all_peak_times = []
        self.all_peak_voltages = []
        self.time_s = None
        self.vin_V = None
        self.vout_V = None
        self.peak_times = None
        self.peak_voltages = None
        self.filtered_voltages = None
        self.energies = None
        self.noise_energies = None
        self.noise_std_dev_ev = None
        self.perform_fit = False
        self.last_fwhm_kev = None

    def _load_config(self, user_config):
        """Loads default and user-provided configuration parameters."""
        default_config = {
            't': 2.8e-6,
            't1': 1.3e-2,
            't2': 1.0e-5,
            'Cx': 0.1e-6,
            'Cf': 3e-12,
            'Rb': 100e6,
            'Rf': 200e6,
            'A0': 38 * e,
            'q': 1.602e-19,
            'w': 3.62
        }
        if user_config:
            default_config.update(user_config)
        for key, value in default_config.items():
            setattr(self, key, value)

    def _load_data(self, file_path):
        """Loads simulation data from a CSV file."""
        if not os.path.exists(file_path):
            print(f"Error: Data file not found at '{file_path}'")
            raise FileNotFoundError(f"Data file not found at '{file_path}'")
        data = np.loadtxt(file_path, delimiter=',', skiprows=1)
        self.time_s = data[:, 0]
        self.vin_V = data[:, 1]
        self.vout_V = data[:, 2]

    def _find_peaks_dynamic_window(self):
        """Finds peaks using a dynamic window based on 'Time over Threshold'."""
        up_crossings = np.where((self.vout_V[:-1] < self.threshold) & (self.vout_V[1:] >= self.threshold))[0] + 1
        peak_voltages_list = []
        peak_times_list = []
        last_event_end_index = -1
        for start_index in up_crossings:
            if start_index <= last_event_end_index:
                continue
            subsequent_data = self.vout_V[start_index:]
            down_crossing_relative_indices = np.where((subsequent_data[:-1] >= self.threshold) & (subsequent_data[1:] < self.threshold))[0]
            if len(down_crossing_relative_indices) > 0:
                relative_end_index = down_crossing_relative_indices[0]
                end_index = start_index + relative_end_index
                pulse_window = self.vout_V[start_index : end_index + 1]
                peak_index_in_window = np.argmax(pulse_window)
                peak_index = start_index + peak_index_in_window
                peak_voltages_list.append(self.vout_V[peak_index])
                peak_times_list.append(self.time_s[peak_index])
                last_event_end_index = end_index
        self.peak_voltages = np.array(peak_voltages_list)
        self.peak_times = np.array(peak_times_list)

    def process_file(self, file_path):
        """Loads and finds peaks in a single file, adding them to the aggregate."""
        try:
            self._load_data(file_path)
            self._find_peaks_dynamic_window()
            self.all_peak_times.extend(self.peak_times)
            self.all_peak_voltages.extend(self.peak_voltages)
            print(f"Processed '{file_path}', added {len(self.peak_voltages)} events. Total events: {len(self.all_peak_voltages)}.")
        except FileNotFoundError as e:
            print(e, file=sys.stderr)

    def analyze(self, fit_spectrum=True):
        """
        Analyzes the aggregated data. The actual fitting is deferred to the plot stage.
        :param fit_spectrum: If True, a Gaussian fit will be calculated and plotted.
        """
        self.peak_voltages = np.array(self.all_peak_voltages)
        self.peak_times = np.array(self.all_peak_times)
        if len(self.peak_voltages) == 0:
            print("Error: No events to analyze. Process at least one file first.")
            return
        self.perform_fit = fit_spectrum
        self._filter_outliers()
        self._convert_voltage_to_energy()
        print("Analysis complete.")

    def _filter_outliers(self):
        """Filters outliers from the aggregated peak voltages."""
        self.filtered_voltages = self.peak_voltages
        if self.peak_voltages is None or len(self.peak_voltages) <= 10:
            print("Not enough data points for statistical filtering.")
            return
        Q1 = np.percentile(self.peak_voltages, 25)
        Q3 = np.percentile(self.peak_voltages, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        original_count = len(self.peak_voltages)
        voltages_after_filtering = self.peak_voltages[(self.peak_voltages >= lower_bound) & (self.peak_voltages <= upper_bound)]
        print(f"Filtered out {original_count - len(voltages_after_filtering)} outliers.")
        if len(voltages_after_filtering) > 10:
            print(f"Using {len(voltages_after_filtering)} events for analysis.")
            self.filtered_voltages = voltages_after_filtering
        else:
            print("Not enough data points remaining after filtering. Using original data.")

    def _convert_voltage_to_energy(self):
        """Converts peak voltages to energy (eV)."""
        if self.filtered_voltages is not None and len(self.filtered_voltages) > 0:
            self.energies = 2.355 * self.w * self.filtered_voltages * e / self.A0 * self.Cf / self.q
        else:
            self.energies = np.array([])

    def add_gaussian_noise(self, noise_kev=1.0, enabled=True):
        """Adds Gaussian noise to the energy spectrum."""
        if not enabled:
            return
        if self.energies is None:
            print("Error: Energies have not been calculated yet. Run .analyze() first.", file=sys.stderr)
            return
        if len(self.energies) == 0:
            print("Warning: No energy data to add noise to.")
            return
        noise_ev = noise_kev * 1000.0
        self.noise_std_dev_ev = noise_ev / 2.355
        noise = np.random.normal(loc=0.0, scale=self.noise_std_dev_ev, size=len(self.energies))
        self.noise_energies = noise
        self.energies += noise
        print(f"Added Gaussian noise with FWHM = {noise_kev} keV (std dev = {self.noise_std_dev_ev / 1000:.2f} keV).")

    def plot_results(self, show_waveform=True, show_plot=True):
        """
        Plots the analysis results. Can be run in a 'calculate-only' mode.
        :param show_waveform: If True, displays the waveform plot alongside the histogram.
        :param show_plot: If True, displays the plot window. If False, calculates but does not show.
        """
        if self.energies is None:
            print("No data to plot. Please run .analyze() first.")
            return

        energies_kev = self.energies / 1000.0
        noise_energies_kev = self.noise_energies / 1000.0 if self.noise_energies is not None else None
        noise_std_dev_kev = self.noise_std_dev_ev / 1000.0 if self.noise_std_dev_ev is not None else None

        plot_spec = []
        if show_waveform:
            plot_spec.append('waveform')
        plot_spec.append('spectrum')
        has_noise = noise_energies_kev is not None
        if has_noise:
            plot_spec.append('noise')

        num_plots = len(plot_spec)
        fig = plt.figure(figsize=(12, 5 * num_plots))
        gs = gridspec.GridSpec(num_plots, 1, figure=fig)
        axes_dict = {}
        for i, plot_type in enumerate(plot_spec):
            axes_dict[plot_type] = fig.add_subplot(gs[i, 0])

        if 'waveform' in axes_dict:
            ax = axes_dict['waveform']
            ax.plot(self.time_s, self.vout_V, label='Output Signal (Vout)', alpha=0.7)
            if self.peak_times is not None:
                ax.plot(self.peak_times, self.peak_voltages, "x", markersize=8, label='Detected Peaks')
            ax.axhline(y=self.threshold, color='r', linestyle=':', label=f'Threshold ({self.threshold}V)')
            ax.set_title('Signal with Detected Event Peaks')
            ax.set_xlabel('Time / s')
            ax.set_ylabel('Voltage / V')
            ax.set_xlim(0, 0.01)
            ax.tick_params(direction='in')
            ax.legend()
            ax.grid(True)

        if 'spectrum' in axes_dict:
            ax = axes_dict['spectrum']
            n, bins, hist_patches = [], [], []
            if len(energies_kev) > 0:
                n, bins, hist_patches = ax.hist(energies_kev, bins=100, edgecolor='black')

            if self.perform_fit and len(energies_kev) > 0:
                mean_kev = np.mean(energies_kev)
                std_kev = np.std(energies_kev)
                if std_kev > 0:
                    fwhm_kev = 2.355 * std_kev
                    self.last_fwhm_kev = fwhm_kev # Store the calculated FWHM
                    
                    xmin_spec, xmax_spec = ax.get_xlim()
                    ymin_spec, ymax_spec = ax.get_ylim()

                    bin_width_kev_spec = bins[1] - bins[0]
                    n_samples_spec = len(energies_kev)
                    x_spec = np.linspace(xmin_spec, xmax_spec, 200)
                    pdf_scaling_factor_spec = n_samples_spec * bin_width_kev_spec
                    y_spec = norm.pdf(x_spec, loc=mean_kev, scale=std_kev) * pdf_scaling_factor_spec
                    ax.plot(x_spec, y_spec, 'k-', label='Gaussian Fit')
                    label = f'FWHM: {fwhm_kev:.2f} keV'
                    max_height = norm.pdf(mean_kev, loc=mean_kev, scale=std_kev) * pdf_scaling_factor_spec
                    half_max_level = max_height / 2
                    ax.hlines(half_max_level, xmin_spec-fwhm_kev*10, xmax_spec+fwhm_kev*10, color='k', linestyle=':', label=label)
                    ax.vlines([mean_kev - fwhm_kev / 2, mean_kev + fwhm_kev / 2], ymin_spec, ymax_spec, color='r', linestyle='--')
                    ax.legend()
                    print(f"Fit on final data: Mean={mean_kev:.2f} keV, StdDev={std_kev:.2f} keV, FWHM={fwhm_kev:.2f} keV")
            ax.set_title('Energy Spectrum (Peak Energy Distribution)')
            ax.set_ylabel('Counts')
            ax.set_xlabel('Peak Energy / keV')
            ax.tick_params(direction='in')
            ax.grid(True)

        if 'noise' in axes_dict:
            noise_ax = axes_dict['noise']
            spectrum_ax = axes_dict.get('spectrum')
            n, bins, patches = noise_ax.hist(noise_energies_kev, bins=100, edgecolor='black', color='sandybrown')
            bin_width_kev = bins[1] - bins[0]
            if spectrum_ax and noise_std_dev_kev is not None:
                noise_limit_kev = 10 * noise_std_dev_kev
                noise_ax.set_xlim(-noise_limit_kev, noise_limit_kev)
                width_kev = 2 * noise_limit_kev
                if len(energies_kev) > 0:
                    spectrum_mean_kev = np.mean(energies_kev)
                    spectrum_ax.set_xlim(spectrum_mean_kev - width_kev / 2, spectrum_mean_kev + width_kev / 2)
            xmin, xmax = noise_ax.get_xlim()
            ymin, ymax = noise_ax.get_ylim()
            x = np.linspace(xmin, xmax, 200)
            pdf_scaling_factor = len(noise_energies_kev) * bin_width_kev
            y = norm.pdf(x, loc=0, scale=noise_std_dev_kev) * pdf_scaling_factor
            noise_ax.plot(x, y, 'k-', label='Ideal Gaussian')
            max_height = norm.pdf(0, loc=0, scale=noise_std_dev_kev) * pdf_scaling_factor
            half_max = max_height / 2
            fwhm_noise_kev = 2.355 * noise_std_dev_kev
            noise_ax.hlines(half_max, xmin, xmax, color='k', linestyle=':', label=f'FWHM: {fwhm_noise_kev:.2f} keV')
            noise_ax.vlines([-fwhm_noise_kev / 2, fwhm_noise_kev / 2], ymin, ymax, color='r', linestyle='--')
            noise_ax.set_title('Added Gaussian Noise Distribution')
            noise_ax.set_xlabel('Noise Energy / keV')
            noise_ax.set_ylabel('Counts')
            noise_ax.tick_params(direction='in')
            noise_ax.legend()
            noise_ax.grid(True)
        
        fig.suptitle('Spectrum Analysis Results', fontsize=32)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        if show_plot:
            plt.show()
        else:
            plt.close(fig) # Close the figure to free up memory

if __name__ == '__main__':
    files_to_analyze = [
        'simulation_output.csv',
    ]
    analyzer = SpectrumAnalyzer(threshold=1.0)
    for file_path in files_to_analyze:
        analyzer.process_file(file_path)
    analyzer.analyze(fit_spectrum=True)
    analyzer.add_gaussian_noise(noise_kev=5.0, enabled=True)
    analyzer.plot_results(show_waveform=False)
