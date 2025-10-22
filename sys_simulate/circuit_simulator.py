import os
import numpy as np
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

class CircuitSimulator:
    """
    Handles the PySpice circuit simulation.
    """
    def __init__(self, step_time=1@u_us, end_time=3000@u_ms):
        """
        Initializes the simulator with timing parameters.
        """
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.step_time = step_time
        self.end_time = end_time
        self.output_filename = os.path.join(self.script_dir, 'simulation_output_temp.csv')

    def _setup_circuit(self, ripple_amplitude_V, ripple_frequency_Hz):
        """
        Sets up the circuit with specific ripple parameters.
        """
        circuit = Circuit('Circuit Simulation for FWHM Analysis')
        library_path = os.path.join(self.script_dir, 'UniversalOpAmp2.lib')
        circuit.include(library_path)
        circuit_path = os.path.join(self.script_dir, 'my_circuit.cir')
        circuit.include(circuit_path)
        
        # Add sources
        circuit.PulseCurrentSource('charge_injection_pulse', 'N006', 'N003',
                                   initial_value=0@u_A, pulsed_value=0.4@u_uA,
                                   delay_time=0.1@u_ms, rise_time=1@u_ns, fall_time=1@u_ns,
                                   pulse_width=1@u_us, period=0.987@u_ms)
        circuit.SinusoidalVoltageSource('ripple', 'Vin', circuit.gnd,
                                        amplitude=ripple_amplitude_V@u_V,
                                        frequency=ripple_frequency_Hz@u_Hz)
        return circuit
    
    def set_filename(self, filename):
        """
        Sets the filename for the simulation output.
        """
        self.output_filename = os.path.join(self.script_dir, filename)

    def run_simulation(self, ripple_amplitude_V, ripple_frequency_Hz):
        """
        Runs the simulation and saves the output to a CSV file.
        
        Args:
            ripple_amplitude_V (float): The amplitude of the ripple voltage in Volts.
            ripple_frequency_Hz (float): The frequency of the ripple in Hertz.
            
        Returns:
            str: The path to the output CSV file.
        """
        circuit = self._setup_circuit(ripple_amplitude_V, ripple_frequency_Hz)
        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        analysis = simulator.transient(step_time=self.step_time, end_time=self.end_time)

        output_data = np.vstack((
            np.array(analysis.time),
            np.array(analysis['Vin']),
            np.array(analysis['Vout'])
        )).T
        
        np.savetxt(self.output_filename, output_data,
                   delimiter=',', header='time_s,vin_V,vout_V', comments='')
        
        return self.output_filename

    def cleanup(self):
        """
        Removes the temporary simulation output file.
        """
        if os.path.exists(self.output_filename):
            os.remove(self.output_filename)

if __name__ == '__main__':
    simulator = CircuitSimulator()
    simulator.set_filename('simulation_output.csv')
    simulator.run_simulation(3@u_V, 10000@u_Hz)