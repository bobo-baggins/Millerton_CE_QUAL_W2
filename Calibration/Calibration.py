# This scripts post-processes model results and raw water temperature data from CDEC so that model performance can be evaluated
#FWQ is a sensor immeadately downstream of river release outlet

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configuration
MODEL_PATHS = {
    'RUN1': 'Runs/two_31_2022_Observed.csv',
    'RUN2': 'Runs/two_31_2022_K2P_Temp_In.csv',
    'RUN3': 'Runs/two_31_2022_Weighted_Temp_In.csv',
}

SIM_START_DATE = '2022-01-13'

# Read model results
model_results = {}
for run_id, path in MODEL_PATHS.items():
    print(f"\nProcessing {run_id} from {path}")
    # Read the data file - only 1st and 6th columns, skip header line
    df = pd.read_csv(path, skiprows=3, delim_whitespace=True, 
                    usecols=[0, 5], names=['JDAY', 'T(C)'])
    
    # Clean up the data - remove commas and convert to numeric
    df['JDAY'] = df['JDAY'].str.rstrip(',').astype(float)
    df['T(C)'] = df['T(C)'].str.rstrip(',').astype(float)
    
    print(f"Raw data shape: {df.shape}")
    print(f"First few values:\n{df.head()}")
    
    # Convert Julian day to datetime and round to nearest hour
    df['DATE TIME'] = pd.to_datetime(SIM_START_DATE) + pd.to_timedelta(df['JDAY'] - 36903, unit='D')
    df['DATE TIME'] = df['DATE TIME'].dt.round('H')
    print(f"After adding datetime, shape: {df.shape}")
    
    # Remove -99 values
    df = df[df['T(C)'] != -99]
    print(f"After removing -99 values, shape: {df.shape}")
    model_results[run_id] = df.set_index('DATE TIME')['T(C)']
    print(f"Model results shape: {model_results[run_id].shape}")
    print(f"First few values:\n{model_results[run_id].head()}")

# Read and process sensor data
print("\nProcessing FWQ data")
fwq = pd.read_csv('FWQ_2022.csv')
print(f"Raw FWQ data shape: {fwq.shape}")
print(f"FWQ columns: {fwq.columns.tolist()}")
print(f"First few rows of FWQ data:\n{fwq.head()}")

# Clean up column names by stripping whitespace
fwq.columns = fwq.columns.str.strip()
print(f"Cleaned FWQ columns: {fwq.columns.tolist()}")

# Convert datetime and temperature
fwq['DATE TIME'] = pd.to_datetime(fwq['DATE TIME'])
print(f"Date range in FWQ data: {fwq['DATE TIME'].min()} to {fwq['DATE TIME'].max()}")

# Convert temperature to float and handle any invalid values
fwq['T(C)'] = pd.to_numeric(fwq['T(C)'], errors='coerce')
print(f"Temperature range in FWQ data: {fwq['T(C)'].min()} to {fwq['T(C)'].max()}")

# Remove invalid values (below 0, -99, -18, and NaN)
fwq = fwq[fwq['T(C)'] > 0]  # Remove any values below 0
fwq = fwq[fwq['T(C)'] != -99]
fwq = fwq[fwq['T(C)'] != -18]
fwq = fwq.dropna(subset=['T(C)'])
print(f"After removing invalid values, shape: {fwq.shape}")

# Set datetime as index and convert to float
fwq_hourly = fwq.set_index('DATE TIME')['T(C)'].astype(float)
print(f"FWQ hourly data shape: {fwq_hourly.shape}")
print(f"First few values:\n{fwq_hourly.head()}")
print(f"Date range in hourly data: {fwq_hourly.index.min()} to {fwq_hourly.index.max()}")

# Create combined dataframe
combined_data = pd.DataFrame()
for run_id in MODEL_PATHS.keys():
    # Align model data with FWQ data
    model_data = model_results[run_id].copy()
    # Round to nearest hour and handle duplicates by taking the mean
    model_data = model_data.groupby(model_data.index.floor('H')).mean()
    combined_data[f'{run_id}_T(C)'] = model_data

# Align FWQ data with model data
fwq_aligned = fwq_hourly.copy()
# Round to nearest hour and handle duplicates by taking the mean
fwq_aligned = fwq_aligned.groupby(fwq_aligned.index.floor('H')).mean()
combined_data['FWQ_T(C)'] = fwq_aligned

print(f"\nCombined data shape: {combined_data.shape}")
print(f"First few rows of combined data:\n{combined_data.head()}")
print(f"Date range in combined data: {combined_data.index.min()} to {combined_data.index.max()}")

# Drop any rows with NaN values
combined_data = combined_data.dropna()
print(f"After dropping NaN values, shape: {combined_data.shape}")
combined_data.to_csv('Combined_Hourly_Temps.csv')

# Plotting
plt.figure(figsize=(12, 8))
plt.rcParams.update({'font.size': 14})  # Set global font size to 14
colors = {'RUN1': 'green', 'RUN2': 'red', 'RUN3': 'purple', 'FWQ': 'blue'}
labels = {
    'RUN1': 'River Release Temperature for SJA Inflow Temperature',
    'RUN2': 'River Release Temperature for K2P Inflow Temperature',
    'RUN3': 'River Release Temperature for Weighted Inflow Temperature',
    'FWQ': 'FWQ Sensor Temperature'
}

print("\nPlotting data...")
for run_id in MODEL_PATHS.keys():
    print(f"Plotting {run_id} with {len(model_results[run_id])} points")
    print(f"Date range for {run_id}: {model_results[run_id].index.min()} to {model_results[run_id].index.max()}")
    plt.plot(model_results[run_id].index, 
             model_results[run_id] * 9/5 + 32,
             color=colors[run_id],
             label=labels[run_id])

print(f"Plotting FWQ with {len(fwq_hourly)} points")
print(f"Date range for FWQ: {fwq_hourly.index.min()} to {fwq_hourly.index.max()}")
plt.plot(fwq_hourly.index, 
         fwq_hourly * 9/5 + 32,
         color=colors['FWQ'],
         label=labels['FWQ'])

# Add vertical line
vline_date = pd.to_datetime('2022-09-30')
plt.axvline(x=vline_date, color='black', linestyle='--', alpha=0.7)
plt.annotate('K2P Data Available\n until September 30th',
             xy=(vline_date, 48),
             xytext=(vline_date + pd.Timedelta(days=10), 45),
             bbox=dict(facecolor='white', edgecolor='black', alpha=0.8),
             fontsize=14,
             arrowprops=dict(arrowstyle='->', color='black', alpha=0.7))

# Calculate and display statistics
stats_text = ""
for run_id in MODEL_PATHS.keys():
    # Use the aligned data for statistics
    model_temp = combined_data[f'{run_id}_T(C)'] * 9/5 + 32
    fwq_temp = combined_data['FWQ_T(C)'] * 9/5 + 32
    
    r2 = np.corrcoef(model_temp, fwq_temp)[0,1]**2
    rmse = np.sqrt(np.mean((model_temp - fwq_temp)**2))
    nse = 1 - (np.sum((model_temp - fwq_temp)**2) / 
              np.sum((fwq_temp - fwq_temp.mean())**2))
    
    stats_text += f"{labels[run_id]} Statistics:\n"
    stats_text += f"R² : {r2:.2f}\n"
    stats_text += f"NSE: {nse:.2f}\n"
    stats_text += f"RMSE: {rmse:.2f}°F\n\n"

plt.text(0.02, 0.98, stats_text.strip(),
         transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
         verticalalignment='top',
         fontsize=14)

# Format plot
plt.ylim(41, 68)
plt.yticks(np.arange(41, 69, 1))
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Water Temp. (°F)', fontsize=14)
plt.legend(fontsize=14, loc='upper right')
plt.tight_layout()
plt.savefig('temperature_validation.png', dpi=300)
plt.show()
