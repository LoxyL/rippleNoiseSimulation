import numpy as np
import math

# Parameters
t = 2.8e-6 # tau
R1 = 13e3
C1 = 0.1e-6
R2 = 10e6
C2 = 0.01e-6
Cd = 2e-12
Cx = 0.1e-6
Cf = 3e-12
Rb = 100e6
Rf = 200e6
Rs = 0 # Rs is not given in table, but usually 0 or small. Let's assume Rs=0 or look at the formula
A0 = 38 * math.e
e = math.e
q = 1.602e-19
w = 3.62

# Frequencies
f = 70e3
v_in = 3.0

# H_powerFilter
H_pf = (1 / (1 + 1j * 2 * math.pi * f * R1 * C1)) * (1 / (1 + 1j * 2 * math.pi * f * R2 * C2))

# H_csa
# formula: - (j2pi f Cd Rb) / ( (1 + j2pi f Cd Rb + j2pi f Cd Rb) * (j2pi f Cf (Rs + 1/(j2pi f Cf))) )
# wait, the text says 1 + j2pi f Cd Rb + j2pi f Cd Rb. That is 1 + j4pi f Cd Rb
# let's assume Rs = 0
Rs = 0
denom_csa = (1 + 1j * 2 * math.pi * f * Cd * Rb + 1j * 2 * math.pi * f * Cd * Rb) * (1j * 2 * math.pi * f * Cf * (Rs + 1/(1j * 2 * math.pi * f * Cf)))
# the second term is just 1! Because j2pi f Cf * (1 / j2pi f Cf) = 1
H_csa = - (1j * 2 * math.pi * f * Cd * Rb) / (1 + 1j * 4 * math.pi * f * Cd * Rb)

# H_shaper
H_shaper = A0 * (1j * 2 * math.pi * f * t) / ((1 + 1j * 2 * math.pi * f * t)**2)

v_out = v_in * abs(H_pf) * abs(H_csa) * abs(H_shaper)

# ENC
ENC = v_out * (e / A0) * Cf
sigma_ENC = ENC / math.sqrt(2)

FWHM = 2.355 * w * sigma_ENC / q

print(f"v_out: {v_out}")
print(f"FWHM (eV): {FWHM}")
print(f"FWHM (keV): {FWHM / 1000}")
