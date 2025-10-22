from tkinter import Scale
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 全局字体设置：英文 Times New Roman，中文回退 SimHei；避免负号变方块
rcParams['font.family'] = 'Times New Roman'
rcParams['font.sans-serif'] = ['Times New Roman', 'SimHei']
rcParams['axes.unicode_minus'] = False
rcParams['font.size'] = 20
rcParams['axes.titlesize'] = 32
rcParams['axes.labelsize'] = 24
rcParams['xtick.labelsize'] = 20
rcParams['ytick.labelsize'] = 20
rcParams['legend.fontsize'] = 20
import os
import numpy as np

def plot_ripple_waveform(file_path):
    """
    Reads data from the specified CSV file and plots the ripple waveform.

    The CSV file should be formatted as follows:
    - The first line is the title and will be ignored.
    - The second line contains metadata, where the third column is the start time 
      and the fourth column is the time increment.
    - Starting from the third line is the actual waveform data, with the first 
      column being the sequence number and the second column being the voltage value.
    """
    try:
        # Read the start time and time increment from the second line
        with open(file_path, 'r', encoding='utf-8') as f:
            f.readline()  # Skip the first line
            meta_line = f.readline()
            meta_parts = meta_line.strip().split(',')
            start_time = float(meta_parts[2])
            increment = float(meta_parts[3])

        # Use pandas to read the waveform data, skipping the first two rows
        # Specify column names and use only the first two columns
        df = pd.read_csv(file_path, skiprows=2, usecols=[0, 1], names=['Sequence', 'Voltage'])

        # Drop any empty rows that may exist
        df.dropna(inplace=True)

        # Calculate the time axis
        df['Time'] = start_time + df['Sequence'] * increment
        
        # Prepare for FFT calculation
        voltage = df['Voltage'].values
        n = len(voltage)
        sample_rate = 1 / increment
        
        # Calculate FFT
        yf = np.fft.fft(voltage)
        xf = np.fft.fftfreq(n, 1 / sample_rate)[:n//2]
        
        # Prepare for plotting
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 15))
        
        # Plot the waveform
        ax1.plot(df['Time'], df['Voltage'])
        plot_title = os.path.basename(file_path).replace('.csv', '')
        ax1.set_title(f'{plot_title} - Time Domain Waveform', fontsize=32)
        ax1.set_xlabel("Time (s)", fontsize=24)
        ax1.set_ylabel("Voltage (V)", fontsize=24)
        ax1.grid(True)

        # Plot the frequency spectrum
        ax2.plot(xf, 2.0/n * np.abs(yf[0:n//2]))
        ax2.set_title(f'{plot_title} - Frequency Spectrum', fontsize=32)
        ax2.set_xlabel("Frequency (Hz)", fontsize=24)
        ax2.set_ylabel("Amplitude", fontsize=24)
        ax2.grid(True)

        # Plot the voltage amplitude distribution histogram
        ax3.hist(df['Voltage'], bins=50, density=True, alpha=0.75, color='g')
        ax3.set_title(f'{plot_title} - Voltage Amplitude Distribution', fontsize=32)
        ax3.set_xlabel("Voltage (V)", fontsize=24)
        ax3.set_ylabel("Probability Density", fontsize=24)
        ax3.grid(True)

        # Adjust layout and show the chart
        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print(f"Error: File not found '{file_path}'")
    except Exception as e:
        print(f"An error occurred while processing the file: {e}")

def plot_ripple_with_sine(file_path):
    """
    Reads data from a specified CSV file and compares it with a 
    400mV 2100Hz sine wave.

    The CSV file should be formatted as follows:
    - The first line is a header and will be ignored.
    - The second line contains metadata, with the third column being the start time
      and the fourth column being the time increment.
    - Data logging starts from the third line, with the first column being the 
      sequence number and the second column being the voltage.
    """
    try:
        # From the second line, read the start time and time increment
        with open(file_path, 'r', encoding='utf-8') as f:
            f.readline()  # Skip first line
            meta_line = f.readline()
            meta_parts = meta_line.strip().split(',')
            start_time = float(meta_parts[2])
            increment = float(meta_parts[3])

        # Read waveform data using pandas, skipping the first two rows
        # Specify column names and use only the first two columns
        df = pd.read_csv(file_path, skiprows=2, usecols=[0, 1], names=['Sequence', 'Voltage'])

        # Remove any empty rows that may exist
        df.dropna(inplace=True)

        # Calculate time axis for measured data
        df['Time'] = start_time + df['Sequence'] * increment
        time_values = df['Time'].values
        
        # Generate a high-resolution sine wave
        amplitude = 0.3  # 300mV
        frequency = 2100  # 2100Hz
        time_min, time_max = time_values.min(), time_values.max()
        # Use 10x sampling for a smoother sine wave plot
        time_values_sine = np.linspace(time_min, time_max, len(time_values) * 10)
        sine_wave = amplitude * np.sin(2 * np.pi * frequency * time_values_sine)
        
        # Prepare for plotting
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Plot the imported ripple waveform
        ax.plot(time_values, df['Voltage'].values, label='Measured Ripple Data')
        
        # Plot the sine wave
        ax.plot(time_values_sine, sine_wave, label=f'{int(amplitude*1000)}mV {frequency}Hz Sinusoidal Ripple Assumption', linestyle='--')
        
        # Mark the amplitude with red horizontal lines
        ax.axhline(y=amplitude, color='r', linestyle='--', label=f'Sine Wave Amplitude ({amplitude}V)')
        ax.axhline(y=-amplitude, color='r', linestyle='--')

        # Set chart title and labels
        ax.set_title(f'Ripple Waveform vs. Sinusoidal Ripple', fontsize=32)
        ax.set_xlabel("Time (s)", fontsize=24)
        ax.set_ylabel("Voltage (V)", fontsize=24)
        ax.grid(True)
        ax.legend(loc='upper right')
        
        # Set x-axis range to the middle 1/10
        SCALE_RATE = 1
        time_range = time_max - time_min
        center_time = time_min + time_range / 2.0
        display_width = time_range / SCALE_RATE
        ax.set_xlim(center_time - display_width / 2.0, center_time + display_width / 2.0)

        # Adjust layout and display chart
        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print(f"Error: File not found '{file_path}'")
    except Exception as e:
        print(f"An error occurred while processing the file: {e}")

if __name__ == '__main__':
    # Specify the path to the CSV file to be plotted
    # csv_file_path = 'data/5.7/旧高压源接电阻2500Hz纹波波形.csv'
    csv_file_path = 'data/6.24/旧纹波波形源.csv'
    plot_ripple_waveform(csv_file_path)
    plot_ripple_with_sine(csv_file_path)
