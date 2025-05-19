import os
import sys
from pathlib import Path
from datetime import datetime

# Set matplotlib backend before importing matplotlib
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend instead of default

import matplotlib.pyplot as plt

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Modules.DataVis import plot_tcd_comparison

# Configuration
YEAR = 2022  # Single variable to control the year
SCENARIO_RUNS = {
    'A': '2022_Analog_Obs',  # Will use regular weighted temperature
    'B': '2022_Madera_Analog'  # Will use TCD weighted temperature
}

def test_temperature_comparison():
    """Test the plot_tcd_comparison function with sample data"""
    # Set up file paths
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                           'CEQUAL_outputs', str(YEAR))
    
    print("\nBase directory:", base_dir)
    
    # Define the model run paths using scenario configuration
    file_a = os.path.join(base_dir, SCENARIO_RUNS['A'], f'two_31_{YEAR}_{SCENARIO_RUNS["A"]}.csv')
    file_b = os.path.join(base_dir, SCENARIO_RUNS['B'], f'two_31_{YEAR}_{SCENARIO_RUNS["B"]}.csv')
    
    print("\nFull file paths:")
    print(f"File A: {file_a}")
    print(f"File B: {file_b}")
    
    # Check if files exist
    print("\nChecking for required files:")
    print(f"File A exists: {os.path.exists(file_a)}")
    print(f"File B exists: {os.path.exists(file_b)}")
    
    if not all(os.path.exists(f) for f in [file_a, file_b]):
        print("Error: One or more required files are missing")
        return
    
    # Read the first file to get Julian days
    import pandas as pd
    df = pd.read_csv(file_a, skiprows=3, header=None)
    jdays = df[0]  # First column contains Julian days
    
    print(f"\nCreating temperature comparison plot for year {YEAR}")
    print(f"Using files:")
    print(f"A: {file_a} (Regular weighted temperature)")
    print(f"B: {file_b} (TCD weighted temperature)")
    
    # Create the plot with scenario names from configuration
    plt = plot_tcd_comparison(
        file_a, 
        file_b, 
        year=YEAR, 
        jdays=jdays,
        scenario_a_name=SCENARIO_RUNS['A'],
        scenario_b_name=SCENARIO_RUNS['B']
    )
    plt.show()

if __name__ == '__main__':
    test_temperature_comparison() 