# -*- coding: utf-8 -*-
import numpy as np
from math import e
import matplotlib.pyplot as plt

class RippleCalculator:
    """
    Calculates the equivalent energy FWHM caused by a sinusoidal voltage ripple
    based on the circuit's transfer function.
    """
    def __init__(self, t=2.8e-6, t1=1.3e-3, t2=0.01, Cx=0.1e-6, Cf=3e-12, Cd=2e-12, Rb=100e6, Rf=200e6, A0=38*e, q=1.6e-19, w=3.62):
        """
        Initializes the calculator with the circuit's physical parameters.
        """
        self.t = t
        self.t1 = t1
        self.t2 = t2
        self.Cx = Cx
        self.Cf = Cf
        self.Cd = Cd
        self.Rb = Rb
        self.Rf = Rf
        self.A0 = A0
        self.q = q
        self.w = w

    def _calculate_transfer_function(self, f):
        """
        Calculates the complex transfer function H(f) for a given frequency.
        """
        if f <= 0:
            return 0j
        
        w = 2 * np.pi * f
        s = 1j * w

        H_1 = 1 / (1 + s * self.t1) * 1 / (1 + s * self.t2)
        H_21 = - s * self.Cd * self.Rb / (1 + s * self.Cx * self.Rb + s * self.Cd * self.Rb)
        H_22 = s * self.Cx / (1 / self.Rf + s * self.Cf)
        H_2 = H_21 * H_22
        H_g = self.A0 / (1 + s * self.t) * s * self.t / (1 + s * self.t)
        
        H_total = H_g * H_1 * H_2
        return H_total

    def calculate_vout_peak(self, amplitude_v, frequency_hz):
        """
        Calculates the peak output voltage for a given ripple.
        """
        H = self._calculate_transfer_function(frequency_hz)
        H_magnitude = np.abs(H)
        V_out_peak = amplitude_v * H_magnitude
        return V_out_peak

    def calculate_fwhm_kev(self, amplitude_v, frequency_hz):
        """
        Calculates the equivalent FWHM in keV for a given ripple.

        :param amplitude_v: Peak amplitude of the input ripple voltage (V).
        :param frequency_hz: Frequency of the input ripple (Hz).
        :return: Equivalent FWHM in keV.
        """
        # 1. Calculate the transfer function magnitude at the ripple frequency
        H = self._calculate_transfer_function(frequency_hz)
        H_magnitude = np.abs(H)

        # 2. Calculate the peak output voltage ripple
        V_out_peak = amplitude_v * H_magnitude

        # 3. The ripple creates a peak-to-peak voltage fluctuation of 2 * V_out_peak.
        #    This is the effective "width" of the energy peak in volts.
        delta_V = 2 * V_out_peak

        # 4. Convert this voltage width to an energy width (FWHM) in eV.
        #    This uses the same energy conversion formula as the analyzer.
        fwhm_ev = 2.355 * self.w * delta_V * e / self.A0 * self.Cf / self.q
        
        # 5. Convert from eV to keV
        fwhm_kev = fwhm_ev / 1000.0
        
        return fwhm_kev

if __name__ == '__main__':
    # --- Example Usage ---
    # Create an instance of the calculator with default parameters
    calculator = RippleCalculator()

    # Input ripple parameters
    V_in_peak = 5  
    f_ripple = 100

    # Calculate the theoretical FWHM
    theoretical_fwhm = calculator.calculate_fwhm_kev(V_in_peak, f_ripple)
    V_out_peak = calculator.calculate_vout_peak(V_in_peak, f_ripple)
    
    print(f"The peak output voltage is: {V_out_peak:.4f} V")
    print(f"For a ripple of {V_in_peak:.2f} V at {f_ripple / 1000:.1f} kHz:")
    print(f"The theoretical equivalent FWHM is: {theoretical_fwhm:.4f} keV")

    # Example of combining with another noise source (e.g., 5 keV detector noise) in quadrature
    detector_noise_fwhm = 5.0 # keV
    total_fwhm = np.sqrt(theoretical_fwhm**2 + detector_noise_fwhm**2)
    print(f"Total FWHM combined with {detector_noise_fwhm} keV detector noise: {total_fwhm:.4f} keV")

    # # --- Bode Plot Generation ---
    # frequencies = np.logspace(0, 6, 500) # From 1 Hz to 1 MHz
    # transfer_function_values = [calculator._calculate_transfer_function(f) for f in frequencies]
    
    # magnitudes_db = 20 * np.log10(np.abs(transfer_function_values))
    # phases_deg = np.angle(transfer_function_values, deg=True)
    
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # # Magnitude Plot
    # ax1.plot(frequencies, magnitudes_db)
    # ax1.set_ylabel('Magnitude [dB]')
    # ax1.set_title('Bode Plot of the System Transfer Function')
    # ax1.grid(which='both', linestyle='--')
    # ax1.set_xscale('log')
    
    # # Phase Plot
    # ax2.plot(frequencies, phases_deg)
    # ax2.set_xlabel('Frequency [Hz]')
    # ax2.set_ylabel('Phase [degrees]')
    # ax2.grid(which='both', linestyle='--')
    # ax2.set_xscale('log')
    
    # plt.tight_layout()
    # plt.show()
