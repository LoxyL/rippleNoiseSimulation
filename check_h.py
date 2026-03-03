import sys
sys.path.append('sys_simulate')
from ripple_calculation import RippleCalculator
calc = RippleCalculator()
freq = 100.0
amp = 1.0
H_mag = abs(calc._calculate_transfer_function(freq))
V_out_peak = amp * H_mag
print(f"H_mag at 100Hz: {H_mag}")
print(f"V_out_peak at 100Hz: {V_out_peak}")
print(f"Theory FWHM: {calc.calculate_fwhm_kev(amp, freq)}")
