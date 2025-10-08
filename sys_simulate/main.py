import numpy as np
import os
import PySpice.Logging.Logging as Logging
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

from analyze_spectrum import SpectrumAnalyzer
from ripple_calculation import RippleCalculator

def run_simulation_and_analysis(ripple_amplitude_V, ripple_frequency_Hz, noise_kev, calculator, plotFig=False, plotWave=False):
    """
    Configures, runs a single simulation, and analyzes the output to get the FWHM.
    Also calculates the theoretical FWHM for comparison.
    """
    # --- 1. Run PySpice Simulation ---
    # (Simulation logic remains the same)
    circuit = Circuit('Circuit Simulation for FWHM Analysis')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    library_path = os.path.join(script_dir, 'UniversalOpAmp2.lib')
    circuit.include(library_path)
    circuit_path = os.path.join(script_dir, 'my_circuit.cir')
    circuit.include(circuit_path)
    circuit.PulseCurrentSource('charge_injection_pulse', 'N006', 'N003',
                               initial_value=0@u_A, pulsed_value=0.4@u_uA,
                               delay_time=0.1@u_ms, rise_time=1@u_ns, fall_time=1@u_ns,
                               pulse_width=1@u_us, period=0.987@u_ms)
    circuit.SinusoidalVoltageSource('ripple', 'Vin', circuit.gnd, 
                                    amplitude=ripple_amplitude_V@u_V, 
                                    frequency=ripple_frequency_Hz@u_Hz)
    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    analysis = simulator.transient(step_time=1@u_us, end_time=3000@u_ms)
    output_filename = os.path.join(script_dir, 'simulation_output_temp.csv')
    output_data = np.vstack((
        np.array(analysis.time),
        np.array(analysis['Vin']),
        np.array(analysis['Vout'])
    )).T
    np.savetxt(output_filename, output_data,
               delimiter=',', header='time_s,vin_V,vout_V', comments='')

    # --- 2. Analyze Simulation Output ---
    analyzer = SpectrumAnalyzer(threshold=1.0)
    analyzer.process_file(output_filename)
    analyzer.analyze(fit_spectrum=True)
    analyzer.add_gaussian_noise(noise_kev=noise_kev, enabled=True)
    analyzer.plot_results(show_waveform=plotWave, show_plot=plotFig)
    os.remove(output_filename)
    simulated_fwhm = analyzer.last_fwhm_kev

    # --- 3. Calculate Theoretical FWHM ---
    theoretical_ripple_fwhm = calculator.calculate_fwhm_kev(ripple_amplitude_V, ripple_frequency_Hz)
    
    # Combine theoretical ripple FWHM with the added noise FWHM in quadrature
    total_theoretical_fwhm = np.sqrt(theoretical_ripple_fwhm**2 + noise_kev**2)
    
    return simulated_fwhm, theoretical_ripple_fwhm, total_theoretical_fwhm


if __name__ == '__main__':
    
    logger = Logging.setup_logging()

    # --- Define Parameter Sweep Ranges ---
    ripple_amplitudes = np.logspace(np.log10(1), np.log10(10), 3)
    ripple_frequencies = np.logspace(np.log10(100), np.log10(100e3), 30)
    
    # Create a single instance of the calculator to be used for all runs
    calculator = RippleCalculator()

    results = []
    total_sims = len(ripple_amplitudes) * len(ripple_frequencies)
    current_sim = 0

    print(f"Starting parameter sweep for {total_sims} simulations...")

    # --- Run the Sweep ---
    for amp in ripple_amplitudes:
        for freq in ripple_frequencies:
            current_sim += 1
            print(f"\n--- Running Simulation {current_sim}/{total_sims} ---")
            print(f"Parameters: Ripple Amp = {amp:.3f} V, Freq = {freq:.1f} Hz")
            
            # 1. Pre-calculate the theoretical ripple FWHM for the current parameters
            theory_ripple_fwhm = calculator.calculate_fwhm_kev(amp, freq)

            # 2. Set the added noise to be twice the theoretical ripple FWHM
            added_noise_kev = theory_ripple_fwhm * 2
            print(f"Theoretical Ripple FWHM: {theory_ripple_fwhm:.3f} keV. Setting added noise to {added_noise_kev:.3f} keV.")

            # 3. Run the simulation and analysis with the dynamic noise level
            sim_fwhm_total, _, theory_total_fwhm = run_simulation_and_analysis(amp, freq, added_noise_kev, calculator, plotFig=False)
            
            if sim_fwhm_total is not None:
                # 4. Subtract the added noise in quadrature to get the net simulated ripple FWHM
                if sim_fwhm_total**2 >= added_noise_kev**2:
                    net_sim_fwhm = np.sqrt(sim_fwhm_total**2 - added_noise_kev**2)
                else:
                    # This can happen if noise dominates and simulation variance is low
                    net_sim_fwhm = 0 
                
                results.append([amp, freq, net_sim_fwhm, theory_ripple_fwhm, theory_total_fwhm])
                print(f"Result: Net Sim FWHM = {net_sim_fwhm:.3f} keV | Theory FWHM = {theory_ripple_fwhm:.3f} keV")
            else:
                results.append([amp, freq, np.nan, theory_ripple_fwhm, theory_total_fwhm])
                print("Warning: Simulation FWHM calculation failed.")

    # --- Save Final Results ---
    results_filename = 'fwhm_sweep_results.csv'
    results_data = np.array(results)
    np.savetxt(results_filename, results_data,
               delimiter=',',
               header='ripple_amplitude_V,ripple_frequency_Hz,net_simulated_fwhm_keV,theoretical_ripple_fwhm_keV,theoretical_total_fwhm_keV',
               comments='')

    print(f"\nSweep complete. All results saved to {results_filename}")

