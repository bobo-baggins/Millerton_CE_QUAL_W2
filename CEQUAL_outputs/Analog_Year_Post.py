# Post-Processing of CEQUAL Model Output for a given analog year

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

# Script Parameters
analog_years = [2024,2023,2022,2021,2020,2019,2018,2017]
start_date = "2018-04-18"  # Start date for the time series
time_step = "1H"           # Time step (1 hour)

model_runs = ["2018_Analog_Scenario_A",
              "2018_Analog_Scenario_B",
              "2018_Analog_Scenario_C"
              ]

def process_model_run(model_run):
    """Process all analog years for a given model run"""
    all_combined_data = []
    all_individual_data = []
    
    # First, find the minimum length across all files
    min_length = float('inf')
    for year in analog_years:
        flow_file = f"{year}/{model_run}/qwo_31_{year}_{model_run}.csv"
        flow_df = pd.read_csv(flow_file, skiprows=3, header=None, usecols=[5], names=['qwo_31'])
        min_length = min(min_length, len(flow_df))
    
    print(f"Using minimum length: {min_length}")
    
    # Create hourly time series with exact length
    hourly_index = pd.date_range(start=start_date, periods=min_length, freq=time_step)
    
    # Get SJR flow from first year (same for all years in a model run)
    base_flow_file = f"{analog_years[0]}/{model_run}/qwo_31_{analog_years[0]}_{model_run}.csv"
    base_flow_data = pd.read_csv(base_flow_file, skiprows=3, header=None, usecols=[5], names=['SJR_Flow']).head(min_length)
    
    # Create base DataFrame with DateTime and SJR flow
    combined_base_df = pd.DataFrame({
        'DateTime': hourly_index,
        'SJR_Flow': base_flow_data['SJR_Flow']
    })
    
    for year in analog_years:
        # Construct file paths
        flow_file = f"{year}/{model_run}/qwo_31_{year}_{model_run}.csv"
        wse_file = f"{year}/{model_run}/tsr_1_seg31_{year}_{model_run}.csv"
        temp_file = f"{year}/{model_run}/two_31_{year}_{model_run}.csv"
        
        # Read flow and spill flow data - skip 3 rows and get 3rd and 6th columns
        flow_data = pd.read_csv(flow_file, skiprows=3, header=None, usecols=[2,5], names=['spill_flow', 'SJR_Flow']).head(min_length)
        
        # Create DataFrame with DateTime and individual flows
        individual_flow_df = pd.DataFrame({
            'DateTime': hourly_index,
            f'Spill_Flow_{year}': flow_data['spill_flow'],
            f'SJR_Flow_{year}': flow_data['SJR_Flow']
        })
        
        # Create DataFrame with DateTime and spill flow for combined data
        combined_flow_df = pd.DataFrame({
            'DateTime': hourly_index,
            f'Spill_Flow_{year}': flow_data['spill_flow']
        })
        
        # Read WSE data - get third column (ELWS(m))
        wse_df = pd.read_csv(wse_file, skiprows=1, header=None, usecols=[2], names=['ELWS(m)'])
        # Create daily time series for WSE
        daily_index = pd.date_range(start=start_date, periods=len(wse_df), freq='1D')
        wse_df.index = daily_index
        wse_df = wse_df.reset_index()
        wse_df.columns = ['DateTime', 'ELWS(m)']
        # Resample to hourly and forward fill
        wse_df = wse_df.set_index('DateTime').resample('1H').ffill().reset_index()
        wse_df = wse_df.head(min_length)  # Trim to match other data
        wse_df.columns = ['DateTime', f'WSE_{year}']
        
        # Read temperature and spill temperature data - skip 3 rows and get 3rd and 6th columns
        temp_data = pd.read_csv(temp_file, skiprows=3, header=None, usecols=[2,5], names=['spill_temp', 'SJR_Temp']).head(min_length)
        
        # Calculate weighted temperature
        weighted_temp = (flow_data['spill_flow'] * temp_data['spill_temp'] + 
                        flow_data['SJR_Flow'] * temp_data['SJR_Temp']) / (flow_data['spill_flow'] + flow_data['SJR_Flow'])
        
        # Create DataFrame with DateTime and weighted temperature for combined data
        combined_temp_df = pd.DataFrame({
            'DateTime': hourly_index,
            f'Weighted_Temp_{year}': weighted_temp
        })
        
        # Create DataFrame with DateTime and individual temperatures
        individual_temp_df = pd.DataFrame({
            'DateTime': hourly_index,
            f'Spill_Temp_{year}': temp_data['spill_temp'],
            f'SJR_Temp_{year}': temp_data['SJR_Temp']
        })
        
        # Merge all data for combined output (without WSE)
        combined_year_data = pd.merge(combined_base_df, combined_flow_df, on='DateTime')
        combined_year_data = pd.merge(combined_year_data, combined_temp_df, on='DateTime')
        all_combined_data.append(combined_year_data)
        
        # Merge all data for individual output (with WSE)
        individual_year_data = pd.merge(individual_flow_df, wse_df, on='DateTime')
        individual_year_data = pd.merge(individual_year_data, individual_temp_df, on='DateTime')
        all_individual_data.append(individual_year_data)
    
    # Combine all years for combined data
    combined_final_df = all_combined_data[0]
    for df in all_combined_data[1:]:
        # Drop SJR_Flow from the right DataFrame before merge to avoid duplication
        df = df.drop(columns=['SJR_Flow'])
        combined_final_df = pd.merge(combined_final_df, df, on='DateTime', how='outer')
    
    # Combine all years for individual data
    individual_final_df = all_individual_data[0]
    for df in all_individual_data[1:]:
        individual_final_df = pd.merge(individual_final_df, df, on='DateTime', how='outer')
    
    # Sort by DateTime
    combined_final_df = combined_final_df.sort_values('DateTime')
    individual_final_df = individual_final_df.sort_values('DateTime')
    
    # Convert all numeric columns to float and remove any whitespace
    for df in [combined_final_df, individual_final_df]:
        for col in df.columns:
            if col != 'DateTime':  # Skip DateTime column
                df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Round only the weighted temperature columns in combined data right before saving
    for col in combined_final_df.columns:
        if 'Weighted_Temp' in col:
            combined_final_df[col] = combined_final_df[col].round(2)
    
    # Save to CSV with appropriate formatting
    combined_output_file = f"{model_run}_combined_data.csv"
    individual_output_file = f"{model_run}_individual_data.csv"
    
    combined_final_df.to_csv(combined_output_file, index=False)
    individual_final_df.to_csv(individual_output_file, index=False)
    
    print(f"Saved combined data for {model_run} to {combined_output_file}")
    print(f"Saved individual data for {model_run} to {individual_output_file}")

# Process each model run
for model_run in model_runs:
    process_model_run(model_run)

