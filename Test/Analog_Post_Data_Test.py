import pandas as pd
import os

def test_data_copy():
    """
    Test script to verify data is being copied correctly from raw files to combined output
    """
    # Define file paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    year = "2024"  # Test with one year
    scenario = "2018_Analog_Scenario_A"  # Test with one scenario
    
    # Raw file paths
    raw_temp_file = os.path.join(base_dir, 'CEQUAL_outputs', year, scenario, 
                                f'two_31_{year}_{scenario}.csv')
    raw_flow_file = os.path.join(base_dir, 'CEQUAL_outputs', year, scenario,
                                f'qwo_31_{year}_{scenario}.csv')
    
    # Combined output file
    combo_file = os.path.join(base_dir, 'CEQUAL_outputs', '2018_Analogs_Scenario_ABC',
                             f'{scenario}_combo.csv')
    
    print(f"Testing data copy for {year} {scenario}")
    print("-" * 50)
    
    # Read raw data
    raw_temp = pd.read_csv(raw_temp_file, skiprows=3, header=None)
    raw_flow = pd.read_csv(raw_flow_file, skiprows=3, header=None)
    
    # Read combined data
    combo_df = pd.read_csv(combo_file)
    
    # Compare all four data columns
    print("\nSpillway Temperature Comparison (Column 2):")
    print("Raw data first 5 values:")
    print(raw_temp[2].head())
    print("\nCombined data first 5 values:")
    print(combo_df[f'Spillway_Temp_{year}'].head())
    
    print("\nSpillway Flow Comparison (Column 2):")
    print("Raw data first 5 values:")
    print(raw_flow[2].head())
    print("\nCombined data first 5 values:")
    print(combo_df[f'Spillway_Flow_{year}'].head())
    
    print("\nRiver Release Temperature Comparison (Column 5):")
    print("Raw data first 5 values:")
    print(raw_temp[5].head())
    print("\nCombined data first 5 values:")
    print(combo_df[f'River_Release_Temp_{year}'].head())
    
    print("\nRiver Release Flow Comparison (Column 5):")
    print("Raw data first 5 values:")
    print(raw_flow[5].head())
    print("\nCombined data first 5 values:")
    print(combo_df[f'River_Release_Flow_{year}'].head())
    
    # Check if values match
    temp_spillway_match = raw_temp[2].equals(combo_df[f'Spillway_Temp_{year}'])
    flow_spillway_match = raw_flow[2].equals(combo_df[f'Spillway_Flow_{year}'])
    temp_river_match = raw_temp[5].equals(combo_df[f'River_Release_Temp_{year}'])
    flow_river_match = raw_flow[5].equals(combo_df[f'River_Release_Flow_{year}'])
    
    print("\nResults:")
    print(f"Spillway Temperature data matches: {temp_spillway_match}")
    print(f"Spillway Flow data matches: {flow_spillway_match}")
    print(f"River Release Temperature data matches: {temp_river_match}")
    print(f"River Release Flow data matches: {flow_river_match}")

if __name__ == '__main__':
    test_data_copy()