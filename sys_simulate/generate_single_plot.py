import sys
import os

# Append current directory to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from circuit_simulator import CircuitSimulator
from analyze_spectrum import SpectrumAnalyzer

def main():
    amp = 3.0
    freq = 10000.0
    noise = 5.0
    
    sim = CircuitSimulator()
    # Use a unique filename so we don't interfere with the running sweep
    sim.set_filename('single_sim_plot_temp.csv') 
    print(f"Running simulation with {amp}V, {freq}Hz...")
    out_file = sim.run_simulation(amp, freq)
    
    analyzer = SpectrumAnalyzer(threshold=1.0)
    analyzer.process_file(out_file)
    analyzer.analyze(fit_spectrum=True)
    analyzer.add_gaussian_noise(noise_kev=noise, enabled=True)
    
    # Change dir to sys_simulate so the image saves as sys_simulate/circ_sim_fwhm.png
    os.chdir(script_dir)
    analyzer.plot_results(show_waveform=False, show_plot=False)
    
    sim.cleanup()
    print("Done. Plot saved to sys_simulate/circ_sim_fwhm.png")

if __name__ == '__main__':
    main()