import sys
sys.path.append('sys_simulate')
from circuit_simulator import CircuitSimulator
from PySpice.Unit import *

sim = CircuitSimulator(step_time=0.1@u_us, end_time=500@u_ms)
sim.set_filename('test_output.csv')
print("Starting simulation...")
sim.run_simulation(1.0, 100)
print("Simulation finished.")
