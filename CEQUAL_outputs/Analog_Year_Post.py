# Post-Processing of CEQUAL Model Output for a given analog year

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

# Constants
PRIMARY_YEAR = 2018  # Primary year for the analog scenarios
analog_years = [2024, 2023,2022,2021, 2020, 2019, 2018, 2017]
model_runs = [f"{PRIMARY_YEAR}_Analog_A_May", 
             f"{PRIMARY_YEAR}_Analog_B_May",  # Add both model runs
             f"{PRIMARY_YEAR}_Analog_C_May"]  # Add both model runs

# Define which model runs should use TCD weighted temperatures
TCD_MODEL_RUNS = []  # Add model runs that should use TCD here

# Define output directory
final_dir = os.path.join(os.getcwd(), f"{PRIMARY_YEAR}_May_Analogs_ABC")

def verify_data_match(raw_data, processed_data, year, model_run):
    """Verify that processed data matches raw data, ignoring -99 values"""
    print(f"\nChecking {year} {model_run}:")
    
    # Get temperature column (index 5)
    raw_temp = raw_data[5].values  # Convert to numpy array to ignore index
    processed_temp = processed_data[f'River_Release_Temp_{year}'].values  # Convert to numpy array to ignore index
    
    # Print heads for debugging
    print("\nRaw temperature data (first 5 rows):")
    print(raw_temp[:5])
    print("\nProcessed temperature data (first 5 rows):")
    print(processed_temp[:5])
    
    # Create mask for valid values (not -99)
    valid_mask = (raw_temp != -99)
    
    # Compare only valid values
    if not np.array_equal(raw_temp[valid_mask], processed_temp[valid_mask]):
        print(f"WARNING: Temperature mismatch in {year}")
        # Find first difference
        diff_mask = (raw_temp != processed_temp) & valid_mask
        if np.any(diff_mask):
            diff_idx = np.where(diff_mask)[0][0]
            print(f"First difference at index {diff_idx}:")
            print(f"Raw: {raw_temp[diff_idx]}")
            print(f"Processed: {processed_temp[diff_idx]}")
        return False
    
    print(f"✓ {year} temperature data verified")
    return True

def verify_file_structure(model_run):
    """Verify that all required files exist and are readable"""
    print(f"\nVerifying file structure for {model_run}")
    
    # Get current working directory
    current_dir = os.getcwd()
    
    for year in analog_years:
        # Check temperature file
        temp_file = os.path.join(current_dir, f"{year}", model_run, 
                                f"two_31_{year}_{model_run}.csv")
        print(f"Checking temperature file: {temp_file}")
        if not os.path.exists(temp_file):
            print(f"ERROR: Missing temperature file: {temp_file}")
            return False
            
        # Check flow file
        flow_file = os.path.join(current_dir, f"{year}", model_run,
                                f"qwo_31_{year}_{model_run}.csv")
        print(f"Checking flow file: {flow_file}")
        if not os.path.exists(flow_file):
            print(f"ERROR: Missing flow file: {flow_file}")
            return False
    
    print("✓ All required files exist")
    return True

def verify_data_content(combo_df, model_run):
    """Verify the content of the combined DataFrame"""
    print(f"\nVerifying data content for {model_run}")
    
    # Check for data consistency
    if len(combo_df) == 0:
        print("ERROR: No data found in the combined dataframe")
        return False
        
    # Check for any gaps in the DateTime index
    date_range = pd.date_range(start=combo_df['DateTime'].min(), 
                             end=combo_df['DateTime'].max(), 
                             freq='h')
    missing_dates = date_range.difference(combo_df['DateTime'])
    
    if len(missing_dates) > 0:
        print(f"WARNING: Found {len(missing_dates)} missing hours in the data")
        print("First few missing dates:", missing_dates[:5].tolist())
        print("This might be expected for some scenarios - proceeding with verification")
    
    # Check for duplicate timestamps
    duplicates = combo_df[combo_df['DateTime'].duplicated()]
    if len(duplicates) > 0:
        print(f"WARNING: Found {len(duplicates)} duplicate timestamps")
        print("First few duplicates:", duplicates['DateTime'].head().tolist())
        print("This might be expected for some scenarios - proceeding with verification")
    
    # Check for any NaN values with detailed reporting
    if combo_df.isna().any().any():
        print("WARNING: NaN values found in data")
        print("\nNaN values by column:")
        for col in combo_df.columns:
            nan_count = combo_df[col].isna().sum()
            if nan_count > 0:
                print(f"{col}: {nan_count} NaN values")
                # Print the first few rows where NaN occurs
                nan_rows = combo_df[combo_df[col].isna()].head()
                print("First few rows with NaN values:")
                print(nan_rows[['DateTime', col]])
                print()
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
    
    # Check if this model run should use TCD weighted temperatures
    use_tcd = model_run in TCD_MODEL_RUNS
    if use_tcd:
        print(f"Using TCD weighted temperatures for {model_run}")
    
    # Verify file structure first
    if not verify_file_structure(model_run):
        print(f"File structure verification failed for {model_run}")
        return
    
    # Get current working directory and create output path
    current_dir = os.getcwd()
    output_dir = final_dir  # Use the final_dir defined at the top
    print(f"Creating output directory: {output_dir}")
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        print(f"Error creating output directory: {str(e)}")
        return

    # Read first year's data to get the time series
    first_year = analog_years[0]
    print(f"\nUsing {first_year} as reference year for time series")
    temp_file = os.path.join(current_dir, f"{first_year}", model_run, 
                            f"two_31_{first_year}_{model_run}.csv")
    print(f"Reading temperature file: {temp_file}")
    
    if not os.path.exists(temp_file):
        print(f"Error: Temperature file not found: {temp_file}")
        return
        
    try:
        first_data = pd.read_csv(temp_file, skiprows=3, header=None)
        print(f"First year data shape for {model_run}: {first_data.shape}")
    except Exception as e:
        print(f"Error reading temperature file: {str(e)}")
        return
    
    # Get the first Julian day and convert to datetime
    first_jday = first_data[0].iloc[0]
    start_date = pd.Timestamp('1921-01-01') + pd.Timedelta(days=int(first_jday)-1)
    
    # Get the minimum length of all data files
    min_length = float('inf')
    for year in analog_years:
        temp_file = os.path.join(current_dir, f"{year}", model_run, 
                                f"two_31_{year}_{model_run}.csv")
        if not os.path.exists(temp_file):
            print(f"Warning: Temperature file not found for year {year}: {temp_file}")
            continue
            
        try:
            temp_data = pd.read_csv(temp_file, skiprows=3, header=None)
            print(f"Year {year} data shape for {model_run}: {temp_data.shape}")
            min_length = min(min_length, len(temp_data))
        except Exception as e:
            print(f"Error reading temperature file for year {year}: {str(e)}")
            continue
    
    if min_length == float('inf'):
        print("Error: No valid data files found")
        return
        
    print(f"\nUsing minimum length of {min_length} rows for {model_run}")
    
    # Create datetime index with the minimum length
    datetime_index = pd.date_range(start=start_date, periods=min_length, freq='h')
    
    # Initialize DataFrame with DateTime as first column
    combo_df = pd.DataFrame({'DateTime': datetime_index})
    
    # Process each year
    for year in analog_years:
        # Read temperature data
        temp_file = os.path.join(current_dir, f"{year}", model_run, 
                                f"two_31_{year}_{model_run}.csv")
        flow_file = os.path.join(current_dir, f"{year}", model_run,
                                f"qwo_31_{year}_{model_run}.csv")
        
        if not os.path.exists(temp_file) or not os.path.exists(flow_file):
            print(f"Warning: Missing files for year {year}")
            continue
            
        try:
            temp_data = pd.read_csv(temp_file, skiprows=3, header=None)
            flow_data = pd.read_csv(flow_file, skiprows=3, header=None)
            
            print(f"\nProcessing {year} for {model_run}:")
            print(f"Temperature data shape: {temp_data.shape}")
            print(f"Flow data shape: {flow_data.shape}")
            
            # Truncate both raw and processed data to min_length
            temp_data = temp_data.iloc[:min_length]
            flow_data = flow_data.iloc[:min_length]
            
            # Add temp and flow data to combo_df
            combo_df[f'Spillway_Temp_{year}'] = temp_data[2]
            combo_df[f'Spillway_Flow_{year}'] = flow_data[2]
            combo_df[f'River_Release_Temp_{year}'] = temp_data[5]
            combo_df[f'River_Release_Flow_{year}'] = flow_data[5]
            combo_df[f'Madera_Temp_{year}'] = temp_data[4]  # Add Madera temperature
            
            # Verify data match for this year
            if not verify_data_match(temp_data, combo_df, year, model_run):
                print(f"Data match verification failed for year {year}")
                continue
                
        except Exception as e:
            print(f"Error processing year {year}: {str(e)}")
            continue

    # Verify data content and ranges
    if not verify_data_content(combo_df, model_run):
        print("Data content verification failed")
        return
    if not verify_value_ranges(combo_df, model_run):
        print("Value ranges verification failed")
        return

    # Create weighted temperature DataFrame
    weighted_df = pd.DataFrame({'DateTime': combo_df['DateTime']})
    
    # Process each year
    for year in analog_years:
        # Get the column names for this year
        temp_river = f'River_Release_Temp_{year}'
        flow_river = f'River_Release_Flow_{year}'
        temp_spill = f'Spillway_Temp_{year}'
        flow_spill = f'Spillway_Flow_{year}'
        temp_madera = f'Madera_Temp_{year}'
        
        if not all(col in combo_df.columns for col in [temp_river, flow_river, temp_spill, flow_spill, temp_madera]):
            print(f"Warning: Missing columns for year {year}")
            continue
            
        if use_tcd:
            # Calculate TCD weighted temperature
            weighted_df[f'Weighted_TCD_Temp_{year}'] = combo_df[temp_river]
            
            # Calculate TCD weighted temperature only where spillway values are valid
            spillway_valid = (combo_df[temp_spill] != -99) & (combo_df[flow_spill] != -99) & \
                           (~combo_df[temp_spill].isna()) & (~combo_df[flow_spill].isna()) & \
                           (combo_df[flow_river] + combo_df[flow_spill] > 0)
            
            # Get the month for each timestamp
            months = combo_df['DateTime'].dt.month
            
            # Initialize flow adjustments
            flow_adjustment = np.zeros_like(combo_df[flow_river])
            
            # Apply flow adjustments based on month
            flow_adjustment[(months >= 3) & (months <= 6)] = 9.91  # March to June
            flow_adjustment[months == 7] = 4.25  # July
            flow_adjustment[months == 8] = 2.83  # August
            
            # Calculate adjusted release flow
            adjusted_release_flow = combo_df[flow_river] + flow_adjustment
            
            # Calculate total flow for weighting
            total_flow = combo_df[flow_spill] + adjusted_release_flow
            
            # Calculate weighted average temperature including all components
            weighted_sum = (combo_df[temp_river] * combo_df[flow_river]) + \
                          (combo_df[temp_madera] * flow_adjustment)
            
            # Add spillway component if there's spillway flow
            weighted_sum[spillway_valid] += (combo_df[temp_spill] * combo_df[flow_spill])[spillway_valid]
            
            # Calculate final TCD temperature
            weighted_df[f'Weighted_TCD_Temp_{year}'] = weighted_sum / total_flow
            
            # Any remaining NaN values should default to river temperature
            weighted_df[f'Weighted_TCD_Temp_{year}'] = weighted_df[f'Weighted_TCD_Temp_{year}'].fillna(combo_df[temp_river])
        else:
            # Calculate regular weighted temperature
            weighted_df[f'Weighted_Temp_{year}'] = combo_df[temp_river]
            
            # Calculate weighted temperature only where spillway values are valid
            spillway_valid = (combo_df[temp_spill] != -99) & (combo_df[flow_spill] != -99) & \
                           (~combo_df[temp_spill].isna()) & (~combo_df[flow_spill].isna()) & \
                           (combo_df[flow_river] + combo_df[flow_spill] > 0)
            
            # Calculate weighted temperature where spillway values are valid
            weighted_df.loc[spillway_valid, f'Weighted_Temp_{year}'] = (
                (combo_df[temp_river] * combo_df[flow_river] + 
                 combo_df[temp_spill] * combo_df[flow_spill]) / 
                (combo_df[flow_river] + combo_df[flow_spill])
            )[spillway_valid]
            
            # Any remaining NaN values should default to river temperature
            weighted_df[f'Weighted_Temp_{year}'] = weighted_df[f'Weighted_Temp_{year}'].fillna(combo_df[temp_river])
    
    # Save both DataFrames
    combo_output = os.path.join(output_dir, f"{model_run}_combo.csv")
    weighted_output = os.path.join(output_dir, f"{model_run}_weighted.csv")
    
    try:
        combo_df.to_csv(combo_output, index=False, date_format='%Y-%m-%d %H:%M:%S')
        weighted_df.to_csv(weighted_output, index=False, date_format='%Y-%m-%d %H:%M:%S')
        print(f"Saved combined data to: {combo_output}")
        print(f"Saved weighted temperatures to: {weighted_output}")
    except Exception as e:
        print(f"Error saving output files: {str(e)}")
        return
    
    # Print final combo_df information
    print(f"\nFinal combo_df for {model_run}:")
    print(f"Shape: {combo_df.shape}")
    print("\nFirst 5 rows:")
    print(combo_df.head())
    print("\nLast 5 rows:")
    print(combo_df.tail())

# Process each model run
print("\nProcessing model runs...")
for model_run in model_runs:
    try:
        process_model_run(model_run)
    except Exception as e:
        print(f"Error processing {model_run}: {str(e)}")
        continue

# Add your stats function here, before the final directory print
def calculate_stats(weighted_df, model_run):
    """Calculate statistics for each year in the model run"""
    # Convert DateTime to datetime if it's not already
    weighted_df['DateTime'] = pd.to_datetime(weighted_df['DateTime'])
    
    # Print data range for debugging
    print(f"\nData range for {model_run}:")
    print(f"Start: {weighted_df['DateTime'].min()}")
    print(f"End: {weighted_df['DateTime'].max()}")
    print(f"Total rows: {len(weighted_df)}")
    
    # Create date range mask for September 5th to Deceember 5th
    def get_date_range(df):
        year = df['DateTime'].dt.year.iloc[0]  # Get the year from the data
        start_date = pd.Timestamp(f"{year}-09-05")
        end_date = pd.Timestamp(f"{year}-12-05")
        return (df['DateTime'] >= start_date) & (df['DateTime'] < end_date)
    
    # Apply the date range filter
    date_mask = get_date_range(weighted_df)
    filtered_df = weighted_df[date_mask]
    
    # Determine if this is a TCD model run
    is_tcd = model_run in TCD_MODEL_RUNS
    temp_prefix = 'Weighted_TCD_Temp_' if is_tcd else 'Weighted_Temp_'
    
    # Initialize list to store stats
    stats_list = []
    
    # Process each year
    for year in analog_years:
        weighted_temp = f'{temp_prefix}{year}'
        
        if weighted_temp not in filtered_df.columns:
            print(f"Warning: Column {weighted_temp} not found in data for {model_run}")
            continue
            
        if len(filtered_df) > 0:
            # Get timestamps for min and max temperatures
            min_temp = filtered_df[weighted_temp].min()
            max_temp = filtered_df[weighted_temp].max()
            min_time = filtered_df[filtered_df[weighted_temp] == min_temp]['DateTime'].iloc[0]
            max_time = filtered_df[filtered_df[weighted_temp] == max_temp]['DateTime'].iloc[0]
            
            # Calculate hours above 58F (14.44C)
            above_threshold = filtered_df[filtered_df[weighted_temp] > 14.44]
            hours_above = len(above_threshold)
            days_above = hours_above / 24
            
            stats = {
                'Model_Run': model_run,
                'Year': year,
                'Mean_Temp': round(filtered_df[weighted_temp].mean(), 2),
                'Max_Temp': round(max_temp, 2),
                'Min_Temp': round(min_temp, 2),
                'Days_Above_58F': round(days_above, 2),
                'Max_Temp_Time': max_time,
                'Min_Temp_Time': min_time,
                'First_Hour_Above_58F': above_threshold['DateTime'].min() if hours_above > 0 else None,
                'Last_Hour_Above_58F': above_threshold['DateTime'].max() if hours_above > 0 else None
            }
            stats_list.append(stats)
            
            # Print detailed stats for debugging
            print(f"\nDetailed statistics for {model_run} - Year {year}:")
            print(f"Temperature range: {min_temp:.2f} to {max_temp:.2f}")
            print(f"Min temperature occurred at: {min_time}")
            print(f"Max temperature occurred at: {max_time}")
            print(f"Hours above 58F: {hours_above}")
            print(f"Days above 58F: {days_above:.2f}")
            if hours_above > 0:
                print(f"First hour above 58F: {above_threshold['DateTime'].min()}")
                print(f"Last hour above 58F: {above_threshold['DateTime'].max()}")
            else:
                print("No hours above 58F")
    
    return pd.DataFrame(stats_list)

# Process stats for all model runs
print("\nCalculating statistics for all model runs...")
all_stats = []
for model_run in model_runs:
    # Read the weighted temperature file
    weighted_file = os.path.join(final_dir, f"{model_run}_weighted.csv")
    if not os.path.exists(weighted_file):
        print(f"Warning: Weighted file not found for {model_run}, skipping statistics")
        continue
        
    try:
        weighted_df = pd.read_csv(weighted_file)
        weighted_df['DateTime'] = pd.to_datetime(weighted_df['DateTime'])
        
        # Calculate and append stats
        stats_df = calculate_stats(weighted_df, model_run)
        if not stats_df.empty:
            all_stats.append(stats_df)
            print(f"\nStatistics calculated for {model_run}")
        else:
            print(f"\nNo statistics calculated for {model_run} - check data availability")
    except Exception as e:
        print(f"Error calculating stats for {model_run}: {str(e)}")
        continue

# Combine all stats into one DataFrame
if all_stats:
    combined_stats = pd.concat(all_stats, ignore_index=True)
    
    # Save combined stats
    combined_stats_output = os.path.join(final_dir, "all_model_runs_stats.csv")
    combined_stats.to_csv(combined_stats_output, index=False)
    print(f"\nSaved combined statistics to: {combined_stats_output}")
    
    # Print summary of statistics
    print("\nSummary of statistics:")
    print(combined_stats.groupby('Model_Run')[['Mean_Temp', 'Max_Temp', 'Min_Temp', 'Days_Above_58F']].mean())
else:
    print("\nNo statistics were calculated - check if model runs were processed successfully")

print("\nAll files have been saved to:", final_dir)
if os.path.exists(final_dir):
    print("Directory contents:", os.listdir(final_dir))
else:
    print("Warning: Output directory was not created")

