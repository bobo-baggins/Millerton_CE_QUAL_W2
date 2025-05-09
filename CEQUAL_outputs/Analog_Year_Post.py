# Post-Processing of CEQUAL Model Output for a given analog year

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

# Constants
analog_years = [2024, 2023, 2022, 2021, 2020, 2019, 2018, 2017]
model_runs = ["2018_Analog_Scenario_A", "2018_Analog_Scenario_B", "2018_Analog_Scenario_C"]

def verify_data_match(raw_data, processed_data, year, model_run):
    """Verify that processed data matches raw data, ignoring -99 values"""
    # print(f"\nChecking {year} {model_run}:")
    
    # Get temperature column (index 5)
    raw_temp = raw_data[5]
    processed_temp = processed_data[5]
    
    # Create mask for valid values (not -99)
    valid_mask = (raw_temp != -99)
    
    # Compare only valid values
    if not raw_temp[valid_mask].equals(processed_temp[valid_mask]):
        # print(f"WARNING: Temperature mismatch in {year}")
        # Find first difference
        diff_mask = (raw_temp != processed_temp) & valid_mask
        if diff_mask.any():
            diff_idx = diff_mask.idxmax()
            # print(f"First difference at index {diff_idx}:")
            # print(f"Raw: {raw_temp[diff_idx]}")
            # print(f"Processed: {processed_temp[diff_idx]}")
        return False
    
    # print(f"✓ {year} temperature data verified")
    return True

def verify_file_structure(model_run):
    """Verify that all required files exist and are readable"""
    # print(f"\nVerifying file structure for {model_run}")
    
    for year in analog_years:
        # Check temperature file
        temp_file = os.path.join(f"{year}", model_run, 
                                f"two_31_{year}_{model_run}.csv")
        if not os.path.exists(temp_file):
            # print(f"ERROR: Missing temperature file: {temp_file}")
            return False
            
        # Check flow file
        flow_file = os.path.join(f"{year}", model_run,
                                f"qwo_31_{year}_{model_run}.csv")
        if not os.path.exists(flow_file):
            # print(f"ERROR: Missing flow file: {flow_file}")
            return False
    
    # print("✓ All required files exist")
    return True

def verify_data_content(combo_df, model_run):
    """Verify the content of the combined DataFrame"""
    print(f"\nVerifying data content for {model_run}")
    
    # Check for any NaN values
    if combo_df.isna().any().any():
        print("WARNING: NaN values found in data")
        return False
    
    # Check row counts
    expected_rows = 5831  # or whatever your expected row count is
    if len(combo_df) != expected_rows:
        print(f"WARNING: Unexpected number of rows: {len(combo_df)}")
        return False
    
    # Check column names
    expected_columns = []
    for year in analog_years:
        expected_columns.extend([
            f'Spillway_Temp_{year}',
            f'Spillway_Flow_{year}',
            f'River_Release_Temp_{year}',
            f'River_Release_Flow_{year}'
        ])
    
    if not all(col in combo_df.columns for col in expected_columns):
        print("WARNING: Missing expected columns")
        print("Expected columns:", expected_columns)
        print("Actual columns:", combo_df.columns.tolist())
        return False
    
    print("✓ Data content verified")
    return True

def verify_value_ranges(combo_df, model_run):
    """Verify that temperature values are within expected ranges (0 to 40)"""
    print(f"\nVerifying temperature ranges for {model_run}")
    
    # Check temperature ranges, but ignore -99 values
    temp_cols = [col for col in combo_df.columns if 'Temp' in col]
    for col in temp_cols:
        valid_temps = combo_df[col][combo_df[col] != -99]
        if len(valid_temps) > 0:  # Only check if there are valid temperatures
            if valid_temps.max() > 40 or valid_temps.min() < 0:
                print(f"WARNING: Temperature out of range in {col}")
                print(f"Range: {valid_temps.min()} to {valid_temps.max()}")
                return False
    
    print("✓ Temperature ranges verified (all between 0 and 40°C)")
    return True

def process_model_run(model_run):
    """Process and save year-specific data files"""
    print(f"\nProcessing {model_run}")
    
    # Verify file structure first
    if not verify_file_structure(model_run):
        return
    
    # Get current working directory and create output path
    current_dir = os.getcwd()
    output_dir = os.path.join(current_dir, "2018_Analogs_Scenario_ABC")
    os.makedirs(output_dir, exist_ok=True)

    # Read first year's data to get the time series
    first_year = analog_years[0]
    temp_file = os.path.join(current_dir, f"{first_year}", model_run, 
                            f"two_31_{first_year}_{model_run}.csv")
    first_data = pd.read_csv(temp_file, skiprows=3, header=None)
    
    # Convert Julian days to DateTime
    jday_start = first_data[0]  # First column contains Julian days
    start_date = pd.Timestamp('1921-01-01')
    
    # Create hourly timestamps
    datetime_index = []
    for i, jday in enumerate(jday_start):
        # Get the integer part for the date
        day = int(jday)
        # Get the decimal part for the hour
        hour = int((jday - day) * 24)  # Convert fraction of day to hours
        
        # Create the date
        date = start_date + pd.Timedelta(days=day-1)
        # Add the hours
        datetime_index.append(date + pd.Timedelta(hours=hour))
    
    # Initialize DataFrame with DateTime as first column
    combo_df = pd.DataFrame({'DateTime': datetime_index})

    # Process each year
    for year in analog_years:
        # Read temperature data
        temp_file = os.path.join(current_dir, f"{year}", model_run, 
                                f"two_31_{year}_{model_run}.csv")
        temp_data = pd.read_csv(temp_file, skiprows=3, header=None)
        
        # Read flow data
        flow_file = os.path.join(current_dir, f"{year}", model_run,
                                f"qwo_31_{year}_{model_run}.csv")
        flow_data = pd.read_csv(flow_file, skiprows=3, header=None)
        
        # Add temp and flow data to combo_df
        combo_df[f'Spillway_Temp_{year}'] = temp_data[2]
        combo_df[f'Spillway_Flow_{year}'] = flow_data[2]
        combo_df[f'River_Release_Temp_{year}'] = temp_data[5]
        combo_df[f'River_Release_Flow_{year}'] = flow_data[5]

    # Verify data content and ranges
    print("\nRunning verification checks...")
    if not verify_data_content(combo_df, model_run):
        print("Data content verification failed - exiting")
        return
    if not verify_value_ranges(combo_df, model_run):
        print("Value ranges verification failed - exiting")
        return
    
    print("\nAll verifications passed, proceeding with weighted temperature calculation...")

    # Print first few rows of data
    print("\nFirst few rows of data:")
    print(combo_df.head())

    # Create weighted temperature DataFrame
    weighted_df = pd.DataFrame({'DateTime': combo_df['DateTime']})
    
    # Process each year
    for year in analog_years:
        # Get the column names for this year
        temp_river = f'River_Release_Temp_{year}'
        flow_river = f'River_Release_Flow_{year}'
        temp_spill = f'Spillway_Temp_{year}'
        flow_spill = f'Spillway_Flow_{year}'
        
        # Initialize weighted temperature column with river temperature
        weighted_df[f'Weighted_Temp_{year}'] = combo_df[temp_river]
        
        # Calculate weighted temperature only where spillway values are valid
        spillway_valid = (combo_df[temp_spill] != -99) & (combo_df[flow_spill] != -99) & \
                        (~combo_df[temp_spill].isna()) & (~combo_df[flow_spill].isna()) & \
                        (combo_df[flow_river] + combo_df[flow_spill] > 0)  # Ensure denominator is not zero
        
        # Calculate weighted temperature where spillway values are valid
        weighted_df.loc[spillway_valid, f'Weighted_Temp_{year}'] = (
            (combo_df[temp_river] * combo_df[flow_river] + 
             combo_df[temp_spill] * combo_df[flow_spill]) / 
            (combo_df[flow_river] + combo_df[flow_spill])
        )[spillway_valid]
        
        # Any remaining NaN values should default to river temperature
        weighted_df[f'Weighted_Temp_{year}'].fillna(combo_df[temp_river], inplace=True)
    
    # Save both DataFrames
    combo_output = os.path.join(output_dir, f"{model_run}_combo.csv")
    weighted_output = os.path.join(output_dir, f"{model_run}_weighted.csv")
    
    combo_df.to_csv(combo_output, index=False, date_format='%Y-%m-%d %H:%M:%S')
    weighted_df.to_csv(weighted_output, index=False, date_format='%Y-%m-%d %H:%M:%S')
    
    print(f"Saved combined data to: {combo_output}")
    print(f"Saved weighted temperatures to: {weighted_output}")
    
    # Print first few rows of weighted temperatures
    print("\nFirst few rows of weighted temperatures:")
    print(weighted_df.head())

# Process each model run
for model_run in model_runs:
    process_model_run(model_run)

final_dir = os.path.join(os.getcwd(), "2018_Analogs_Scenario_ABC")
print("\nAll files have been saved to:", final_dir)
print("Directory contents:", os.listdir(final_dir))

