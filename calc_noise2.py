from sys_simulate.ripple_calculation import RippleCalculator

calculator = RippleCalculator()
V_in_peak = 3.0
f_ripple = 70000

theoretical_fwhm = calculator.calculate_fwhm_kev(V_in_peak, f_ripple)
print(f"FWHM for {V_in_peak}V at {f_ripple}Hz: {theoretical_fwhm} keV")

# Wait, the text mentions "图7为纹波幅值3.0V，纹波频率10kHz...测得总噪声FWHM为6.91keV，即纹波引入的噪声为...4.77keV"
# Let's check 3V 10kHz to see if it matches 4.77keV
fwhm_10k = calculator.calculate_fwhm_kev(3.0, 10000)
print(f"FWHM for 3.0V at 10kHz: {fwhm_10k} keV")
