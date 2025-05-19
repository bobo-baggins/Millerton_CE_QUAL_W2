# This scripts post-processes model results and raw water temperature data from CDEC so that model performance can be evaluated
#FWQ is a sensor immeadately downstream of river release outlet

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configuration
MODEL_PATHS = {
    'February Profile': '../Cequal_Outputs/2024/Feb_Start_Obs/two_31_2024_Feb_Start_Obs.csv',
    'March Profile': '../Cequal_Outputs/2024/Mar_Start_Obs/two_31_2024_Mar_Start_Obs.csv',
    'April Profile': '../Cequal_Outputs/2024/Apr_Start_Obs/two_31_2024_Apr_Start_Obs.csv',
    #'RUN3': 'Runs/two_31_2022_Weighted_Temp_In.csv',
}

SIM_START_DATE = '2024-02-07'
SECOND_SIM_START_DATE = '2024-04-15'  # New variable for April start date
THIRD_SIM_START_DATE = '2024-04-18'  # New variable for April start date

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
    
    # Convert Julian day to datetime - adjust base date to match expected dates
    base_date = pd.to_datetime('1921-01-01')
    df['DATE TIME'] = base_date + pd.to_timedelta(df['JDAY'], unit='D')
    
    # Adjust the dates to match expected start dates
    if 'Feb_Start_Obs' in path:
        # For February data, adjust to start on SIM_START_DATE
        date_offset = pd.to_datetime(SIM_START_DATE) - df['DATE TIME'].min()
        df['DATE TIME'] = df['DATE TIME'] + date_offset
    elif 'Mar_Start_Obs' in path:
        # For March data, adjust to start on THIRD_SIM_START_DATE
        date_offset = pd.to_datetime(THIRD_SIM_START_DATE) - df['DATE TIME'].min()
        df['DATE TIME'] = df['DATE TIME'] + date_offset
    elif 'Apr_Start_Obs' in path:
        # For April data, adjust to start on SECOND_SIM_START_DATE
        date_offset = pd.to_datetime(SECOND_SIM_START_DATE) - df['DATE TIME'].min()
        df['DATE TIME'] = df['DATE TIME'] + date_offset
    
    # Round to nearest hour
    df['DATE TIME'] = df['DATE TIME'].dt.round('h')  # Note: using 'h' instead of 'H' to address deprecation warning
    
    # Remove -99 values
    df = df[df['T(C)'] != -99]
    print(f"After removing -99 values, shape: {df.shape}")
    model_results[run_id] = df.set_index('DATE TIME')['T(C)']
    print(f"Model results shape: {model_results[run_id].shape}")
    print(f"First few values:\n{model_results[run_id].head()}")

# Extract year from SIM_START_DATE
year = pd.to_datetime(SIM_START_DATE).year

# Read and process sensor data
print("\nProcessing FWQ data")
fwq = pd.read_csv(f'FWQ_{year}.csv')  # Use the extracted year in the filename
fwq.columns = fwq.columns.str.strip()
fwq['DATE TIME'] = pd.to_datetime(fwq['DATE TIME'])
fwq['T(C)'] = pd.to_numeric(fwq['T(C)'], errors='coerce')

# Replace invalid values with NaN instead of removing rows
fwq.loc[fwq['T(C)'] <= 0, 'T(C)'] = np.nan

# Keep original temporal resolution for FWQ data
fwq_hourly = fwq.set_index('DATE TIME')['T(C)'].astype(float)

# Create dataframe for statistics without alignment
combined_data = pd.DataFrame()
for run_id in MODEL_PATHS.keys():
    # Use model data directly
    combined_data[f'{run_id}_T(C)'] = model_results[run_id]

# Add FWQ data directly
combined_data['FWQ_T(C)'] = fwq_hourly

# Save the combined data with year in filename
combined_data.to_csv(f'Combined_Hourly_Temps_{year}.csv')
print(f"\nFirst few rows of Combined_Hourly_Temps_{year}.csv:")
print(combined_data.head())

# Define colors and labels before statistics calculation
colors = {
    'February Profile': 'green',
    'March Profile': 'purple',
    'April Profile': 'red',
    'FWQ': 'blue'
}
labels = {
    'February Profile': 'February Start Profile',
    'March Profile': 'March Start Profile',
    'April Profile': 'April Start Profile',
    'FWQ': 'FWQ Sensor Temperature'
}

# Calculate statistics using the original data
stats_text = ""  # Initialize empty string for stats
for run_id in MODEL_PATHS.keys():
    # Get the overlapping date range
    model_temp = model_results[run_id]  # Keep in Celsius
    fwq_temp = fwq_hourly  # Keep in Celsius
    
    # Find common dates
    common_dates = model_temp.index.intersection(fwq_temp.index)
    model_temp = model_temp[common_dates]
    fwq_temp = fwq_temp[common_dates]
    
    # Debug prints
    print(f"\nRMSE calculation for {run_id}:")
    print(f"Number of points used: {len(model_temp)}")
    print(f"Date range: {model_temp.index.min()} to {model_temp.index.max()}")
    
    # Print last 5 values with timestamps
    print("\nLast 5 values:")
    print("Timestamp | Model Temp | FWQ Temp | Difference | Squared Diff")
    print("------------------------------------------------------------")
    for i in range(-5, 0):
        if i >= -len(model_temp):  # Check if we have enough data points
            diff = model_temp.iloc[i] - fwq_temp.iloc[i]
            print(f"{model_temp.index[i]} | {model_temp.iloc[i]:.3f}°C | {fwq_temp.iloc[i]:.3f}°C | {diff:.3f} | {diff**2:.3f}")
    
    # Calculate statistics in Celsius
    r2 = model_temp.corr(fwq_temp)**2
    rmse_c = np.sqrt(np.mean((model_temp - fwq_temp)**2))
    nse = 1 - (np.sum((model_temp - fwq_temp)**2) / 
              np.sum((fwq_temp - fwq_temp.mean())**2))
    
    # Add stats to text string
    stats_text += f"{labels[run_id]}:\n"
    stats_text += f"R² = {r2:.2f}\n"
    stats_text += f"NSE = {nse:.2f}\n"
    stats_text += f"RMSE = {rmse_c:.3f}°C\n\n"

# Plotting with original temporal resolution
plt.figure(figsize=(12, 8))
plt.rcParams.update({'font.size': 14})

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

# After plotting all data, add the stats text box
plt.text(0.02, 0.98, stats_text.strip(),
         transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
         verticalalignment='top',
         fontsize=12)

# Format plot
plt.ylim(41, 68)
plt.yticks(np.arange(41, 69, 1))
plt.grid(True, linestyle='--', alpha=0.7)

# Add monthly x-axis ticks
plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %Y'))

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

plt.xlabel('Date', fontsize=14)
plt.ylabel('River Release Water Temperature (°F)', fontsize=14)
plt.legend(fontsize=14, loc='upper right')
plt.tight_layout()  # This will ensure the rotated labels don't get cut off
plt.savefig(f'temperature_validation_{year}.png', dpi=300)
plt.show()
