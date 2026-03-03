import sys
import numpy as np

sys.path.append('sys_simulate')
from analyze_spectrum import SpectrumAnalyzer

analyzer = SpectrumAnalyzer(threshold=1.0)
analyzer.process_file('/Users/loxyblack/Documents/Lab/rippleNoiseSimulation/sys_simulate/test_output.csv')
analyzer.analyze(fit_spectrum=False)

energies_kev = analyzer.energies / 1000.0
std_kev = np.std(energies_kev)
print(f"Base std dev without noise: {std_kev:.4f} keV")
print(f"Base FWHM without noise: {std_kev * 2.355:.4f} keV")
print(f"Mean energy: {np.mean(energies_kev):.4f} keV")
