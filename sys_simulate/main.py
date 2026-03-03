import argparse
import numpy as np
import os
import PySpice.Logging.Logging as Logging
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

from analyze_spectrum import SpectrumAnalyzer
from ripple_calculation import RippleCalculator
from circuit_simulator import CircuitSimulator

def run_simulation_and_analysis(ripple_amplitude_V, ripple_frequency_Hz, noise_kev, calculator, plotFig=False, plotWave=False, n_samp=1):
    """
    Configures, runs a single simulation, and analyzes the output to get the FWHM.
    Runs the analysis `n_samp` times with different noise seeds and averages the results to reduce statistical fluctuations.
    Also calculates the theoretical FWHM for comparison.
    """
    # --- 1. Run PySpice Simulation ---
    simulator = CircuitSimulator()
    output_filename = simulator.run_simulation(ripple_amplitude_V, ripple_frequency_Hz)

    # --- 2. Analyze Simulation Output ---
    analyzer = SpectrumAnalyzer(threshold=1.0)
    analyzer.process_file(output_filename)
    analyzer.analyze(fit_spectrum=False) # Fit later after adding noise
    
    sim_fwhms = []
    actual_noises = []
    
    for i in range(n_samp):
        # Reset energies to base (without noise)
        analyzer._convert_voltage_to_energy() 
        analyzer.add_gaussian_noise(noise_kev=noise_kev, enabled=True, print_msg=False)
        # Manually calculate stats to avoid plotting overhead during loops
        energies_kev = analyzer.energies / 1000.0
        std_kev = np.std(energies_kev)
        simulated_fwhm = 2.355 * std_kev
        actual_added_noise_fwhm = analyzer.actual_noise_fwhm_kev if hasattr(analyzer, 'actual_noise_fwhm_kev') else noise_kev
        
        sim_fwhms.append(simulated_fwhm)
        actual_noises.append(actual_added_noise_fwhm)
        
    avg_simulated_fwhm = np.mean(sim_fwhms)
    avg_actual_added_noise = np.mean(actual_noises)
    
    if plotFig or plotWave:
        # Plot the last sample's result if requested
        analyzer.perform_fit = True
        analyzer.plot_results(show_waveform=plotWave, show_plot=plotFig)

    simulator.cleanup()

    # --- 3. Calculate Theoretical FWHM ---
    theoretical_ripple_fwhm = calculator.calculate_fwhm_kev(ripple_amplitude_V, ripple_frequency_Hz)
    
    # Combine theoretical ripple FWHM with the added noise FWHM in quadrature
    total_theoretical_fwhm = np.sqrt(theoretical_ripple_fwhm**2 + noise_kev**2)
    
    return avg_simulated_fwhm, theoretical_ripple_fwhm, total_theoretical_fwhm, avg_actual_added_noise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run detector ripple simulation and analysis.")
    parser.add_argument('--nsamp', type=int, default=1, help='Number of noise samples to average for each data point to smooth results.')
    args = parser.parse_args()
    
    logger = Logging.setup_logging()

    # --- Define Parameter Sweep Ranges ---
    ripple_amplitudes = np.logspace(np.log10(0.1), np.log10(10), 3)
    ripple_frequencies = np.logspace(np.log10(100), np.log10(500e3), 40)
    
    # Create a single instance of the calculator to be used for all runs
    calculator = RippleCalculator()

    results_filename = 'fwhm_sweep_results.csv'
    
    # Read existing results to support resuming
    completed_params = set()
    if os.path.exists(results_filename):
        try:
            # We skip the header line
            existing_data = np.loadtxt(results_filename, delimiter=',', skiprows=1)
            if existing_data.ndim == 1 and len(existing_data) > 0:
                existing_data = existing_data.reshape(1, -1)
            for row in existing_data:
                # Store (amp, freq) pairs rounded to avoid floating point matching issues
                completed_params.add((round(row[0], 5), round(row[1], 1)))
            print(f"Found {len(completed_params)} previously completed simulations. Resuming...")
        except Exception as e:
            print(f"Error reading existing results (might be empty or corrupted), starting fresh: {e}")
            with open(results_filename, 'w') as f:
                f.write('ripple_amplitude_V,ripple_frequency_Hz,net_simulated_fwhm_keV,theoretical_ripple_fwhm_keV,theoretical_total_fwhm_keV\n')
    else:
        # Write header if file doesn't exist
        with open(results_filename, 'w') as f:
            f.write('ripple_amplitude_V,ripple_frequency_Hz,net_simulated_fwhm_keV,theoretical_ripple_fwhm_keV,theoretical_total_fwhm_keV\n')

    total_sims = len(ripple_amplitudes) * len(ripple_frequencies)
    current_sim = 0

    print(f"Starting parameter sweep for {total_sims} simulations...")

    # --- Run the Sweep ---
    for amp in ripple_amplitudes:
        for freq in ripple_frequencies:
            current_sim += 1
            
            if (round(amp, 5), round(freq, 1)) in completed_params:
                print(f"--- Skipping Simulation {current_sim}/{total_sims} (already completed) ---")
                continue
                
            print(f"\n--- Running Simulation {current_sim}/{total_sims} ---")
            print(f"Parameters: Ripple Amp = {amp:.3f} V, Freq = {freq:.1f} Hz")
            
            # 1. Pre-calculate the theoretical ripple FWHM for the current parameters
            theory_ripple_fwhm = calculator.calculate_fwhm_kev(amp, freq)

            # 2. Set the added noise to be twice the theoretical ripple FWHM
            added_noise_kev = theory_ripple_fwhm * 2
            print(f"Theoretical Ripple FWHM: {theory_ripple_fwhm:.3f} keV. Setting added noise to {added_noise_kev:.3f} keV.")

            # 3. Run the simulation and analysis with the dynamic noise level
            sim_fwhm_total, _, theory_total_fwhm, actual_added_noise = run_simulation_and_analysis(amp, freq, added_noise_kev, calculator, plotFig=False, n_samp=args.nsamp)
            
            if sim_fwhm_total is not None:
                # 4. Subtract the actual added noise in quadrature to get the net simulated ripple FWHM
                if sim_fwhm_total**2 >= actual_added_noise**2:
                    net_sim_fwhm = np.sqrt(sim_fwhm_total**2 - actual_added_noise**2)
                else:
                    # This can happen if noise dominates and simulation variance is low
                    net_sim_fwhm = 0
                
                print(f"Result: Net Sim FWHM = {net_sim_fwhm:.3f} keV | Theory FWHM = {theory_ripple_fwhm:.3f} keV (from {args.nsamp} samples)")
                
                # Append result immediately
                with open(results_filename, 'a') as f:
                    f.write(f"{amp},{freq},{net_sim_fwhm},{theory_ripple_fwhm},{theory_total_fwhm}\n")
            else:
                print("Warning: Simulation FWHM calculation failed.")
                with open(results_filename, 'a') as f:
                    f.write(f"{amp},{freq},{np.nan},{theory_ripple_fwhm},{theory_total_fwhm}\n")

    print(f"\nSweep complete. All results saved to {results_filename}")

