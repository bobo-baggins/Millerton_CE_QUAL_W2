# This scripts post-processes model results and raw water temperature data from CDEC so that model performance can be evaluated
#FWQ is a sensor immeadately downstream of river release outlet

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configuration
YEAR = 2024 # Single variable to control the year
SINGLE_RUN = None  # Set to None to analyze all runs, or specify a single run (e.g., 'February Profile')
SHOW_DIFFERENCE_PLOT = False  # Set to False to hide the difference subplot

# Figure size configuration (width, height) in inches
FIGURE_WIDTH = 16  # Width of the figure in inches
FIGURE_HEIGHT = 15  # Height of the figure in inches
FIGURE_DPI = 300  # Dots per inch for the output image

MODEL_PATHS = {
    'February Profile': f'../Cequal_Outputs/{YEAR}/Feb_Start_Obs/two_31_{YEAR}_Feb_Start_Obs.csv',
    'March Profile': f'../Cequal_Outputs/{YEAR}/Mar_Start_Obs/two_31_{YEAR}_Mar_Start_Obs.csv',
    'April Profile': f'../Cequal_Outputs/{YEAR}/Apr_Start_Obs/two_31_{YEAR}_Apr_Start_Obs.csv',
}

# If SINGLE_RUN is specified, filter MODEL_PATHS to only include that run
if SINGLE_RUN is not None:
    MODEL_PATHS = {SINGLE_RUN: MODEL_PATHS[SINGLE_RUN]}

FLOW_PATHS = {
    'February Profile': f'../Cequal_Outputs/{YEAR}/Feb_Start_Obs/qwo_31_{YEAR}_Feb_Start_Obs.csv',
    'March Profile': f'../Cequal_Outputs/{YEAR}/Mar_Start_Obs/qwo_31_{YEAR}_Mar_Start_Obs.csv',
    'April Profile': f'../Cequal_Outputs/{YEAR}/Apr_Start_Obs/qwo_31_{YEAR}_Apr_Start_Obs.csv',
}

# If SINGLE_RUN is specified, filter FLOW_PATHS to only include that run
if SINGLE_RUN is not None:
    FLOW_PATHS = {SINGLE_RUN: FLOW_PATHS[SINGLE_RUN]}

# Read model results
model_results = {}
flow_results = {}  # New dictionary to store flow data
for run_id, path in MODEL_PATHS.items():
    print(f"\nProcessing {run_id} from {path}")
    # Read the temperature data file - columns 3 and 6
    temp_df = pd.read_csv(path, skiprows=3, delim_whitespace=True, 
                         usecols=[0, 2, 5], names=['JDAY', 'Spill_T(C)', 'Release_T(C)'])
    
    # Read the flow data file - columns 3 and 6
    flow_df = pd.read_csv(FLOW_PATHS[run_id], skiprows=3, delim_whitespace=True,
                         usecols=[0, 2, 5], names=['JDAY', 'Spill_Flow', 'Release_Flow'])
    
    # Clean up the data - remove commas and convert to numeric
    for df in [temp_df, flow_df]:
        for col in df.columns:
            df[col] = df[col].str.rstrip(',').astype(float)
        df['JDAY'] = df['JDAY'].astype(float)  # JDAY doesn't need comma removal
    
    print(f"Raw data shape: {temp_df.shape}")
    print(f"First few values:\n{temp_df.head()}")
    
    # Convert Julian day to datetime - adjust base date to account for Julian day offset
    base_date = pd.to_datetime('1920-12-31')  # Changed from 1921-01-01 to account for Julian day offset
    temp_df['DATE TIME'] = base_date + pd.to_timedelta(temp_df['JDAY'], unit='D')
    flow_df['DATE TIME'] = base_date + pd.to_timedelta(flow_df['JDAY'], unit='D')
    
    # Round to nearest hour
    temp_df['DATE TIME'] = temp_df['DATE TIME'].dt.round('h')
    flow_df['DATE TIME'] = flow_df['DATE TIME'].dt.round('h')
    
    # Convert -99 values to NaN before weighted calculation
    temp_df.loc[temp_df['Spill_T(C)'] == -99, 'Spill_T(C)'] = np.nan
    temp_df.loc[temp_df['Release_T(C)'] == -99, 'Release_T(C)'] = np.nan
    flow_df.loc[flow_df['Spill_Flow'] == -99, 'Spill_Flow'] = np.nan
    flow_df.loc[flow_df['Release_Flow'] == -99, 'Release_Flow'] = np.nan
    
    # Calculate weighted temperature
    def calculate_weighted_temp(row):
        spill_flow = row['Spill_Flow']
        release_flow = row['Release_Flow']
        spill_temp = row['Spill_T(C)']
        release_temp = row['Release_T(C)']
        
        if spill_flow == 0:
            return release_temp
        else:
            return ((spill_temp * spill_flow) + (release_temp * release_flow)) / (spill_flow + release_flow)
    
    # Merge temperature and flow data
    merged_df = pd.merge(temp_df, flow_df, on='DATE TIME', suffixes=('_temp', '_flow'))
    
    # Calculate weighted temperature
    merged_df['Weighted_T(C)'] = merged_df.apply(calculate_weighted_temp, axis=1)
    
    # No need for these conversions anymore since we did them before
    # merged_df.loc[merged_df['Weighted_T(C)'] == -99, 'Weighted_T(C)'] = np.nan
    # merged_df.loc[merged_df['Spill_T(C)'] == -99, 'Spill_T(C)'] = np.nan
    # etc...
    
    print(f"After removing -99 values, shape: {merged_df.shape}")
    model_results[run_id] = merged_df.set_index('DATE TIME')['Weighted_T(C)']
    print(f"Model results shape: {model_results[run_id].shape}")
    print(f"First few values:\n{model_results[run_id].head()}")
    
    # Store the flow data
    flow_results[run_id] = merged_df.set_index('DATE TIME')[['Spill_Flow', 'Release_Flow']]

# Read and process sensor data
print("\nProcessing FWQ data")
fwq = pd.read_csv(f'FWQ_{YEAR}.csv')
fwq.columns = fwq.columns.str.strip()
fwq['DATE TIME'] = pd.to_datetime(fwq['DATE TIME'])
fwq['T(C)'] = pd.to_numeric(fwq['T(C)'], errors='coerce')  # This converts blanks to NaN

# Replace invalid values with NaN instead of removing rows
fwq.loc[fwq['T(C)'] <= 0, 'T(C)'] = np.nan
fwq.loc[fwq['T(C)'] == -99, 'T(C)'] = np.nan  # Add explicit conversion of -99 to NaN

# Keep original temporal resolution for FWQ data
fwq_hourly = fwq.set_index('DATE TIME')['T(C)'].astype(float)

# Create dataframe for statistics without alignment
combined_data = pd.DataFrame()
for run_id in MODEL_PATHS.keys():
    combined_data[f'{run_id}_T(C)'] = model_results[run_id]

# Add FWQ data directly
combined_data['FWQ_T(C)'] = fwq_hourly

# Save the combined data with year in filename
combined_data.to_csv(f'Combined_Hourly_Temps_{YEAR}.csv')
print(f"\nFirst few rows of Combined_Hourly_Temps_{YEAR}.csv:")
print(combined_data.head())

# Define colors and labels before statistics calculation
colors = {
    'February Profile': 'green',
    'March Profile': 'purple',
    'April Profile': 'red',
    'FWQ': 'blue'
}
labels = {
    'February Profile': 'February Start Profile (Weighted)',
    'March Profile': 'March Start Profile (Weighted)',
    'April Profile': 'April Start Profile (Weighted)',
    'FWQ': 'FWQ Sensor Temperature'
}

# Check for spillway flow using February run data
flow_data = flow_results['February Profile']
has_spillway_flow = (flow_data['Spill_Flow'] > 0).any()

# Calculate statistics using the original data
stats_text = ""  # Initialize empty string for stats
stats_data = []  # List to store stats for CSV

# Define date range for CSV statistics
start_date = f'{YEAR}-09-05'
end_date = f'{YEAR}-12-05'

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
    
    # Calculate statistics in Celsius for plot
    r2 = model_temp.corr(fwq_temp)**2
    rmse_c = np.sqrt(np.mean((model_temp - fwq_temp)**2))
    nse = 1 - (np.sum((model_temp - fwq_temp)**2) / 
              np.sum((fwq_temp - fwq_temp.mean())**2))
    
    # Add stats to text string
    label = labels[run_id]
    if not has_spillway_flow:
        label = label.replace(" (Weighted)", "")
    stats_text += f"{label}:\n"
    stats_text += f"R² = {r2:.2f}\n"
    stats_text += f"NSE = {nse:.2f}\n"
    stats_text += f"RMSE = {rmse_c:.2f}°C\n\n"
    
    # Calculate filtered statistics for CSV
    date_mask = (model_temp.index >= start_date) & (model_temp.index <= end_date)
    filtered_model_temp = model_temp[date_mask]
    filtered_fwq_temp = fwq_temp[date_mask]
    
    # Calculate differences
    differences = filtered_model_temp - filtered_fwq_temp
    abs_differences = abs(differences)
    max_diff = differences.max()
    min_diff = differences.min()
    mean_diff = differences.mean()
    min_abs_diff = abs_differences.min()
    max_abs_diff = abs_differences.max()
    mean_abs_diff = abs_differences.mean()
    
    filtered_r2 = filtered_model_temp.corr(filtered_fwq_temp)**2
    filtered_rmse_c = np.sqrt(np.mean((filtered_model_temp - filtered_fwq_temp)**2))
    filtered_nse = 1 - (np.sum((filtered_model_temp - filtered_fwq_temp)**2) / 
                       np.sum((filtered_fwq_temp - filtered_fwq_temp.mean())**2))
    
    # Store filtered stats for CSV
    stats_data.append({
        'Profile': run_id,
        'R2': filtered_r2,
        'NSE': filtered_nse,
        'RMSE': filtered_rmse_c,
        'Max_Difference': max_diff,
        'Min_Difference': min_diff,
        'Mean_Difference': mean_diff,
        'Max_Abs_Difference': max_abs_diff,
        'Min_Abs_Difference': min_abs_diff,
        'Mean_Abs_Difference': mean_abs_diff,
        'Number_of_Points': len(filtered_model_temp),
        'Start_Date': filtered_model_temp.index.min(),
        'End_Date': filtered_model_temp.index.max()
    })

# Create and save stats DataFrame
stats_df = pd.DataFrame(stats_data)

# Format numeric columns to 2 decimal places
numeric_columns = ['R2', 'NSE', 'RMSE', 'Max_Difference', 'Min_Difference', 'Mean_Difference', 
                  'Max_Abs_Difference', 'Mean_Abs_Difference']
stats_df[numeric_columns] = stats_df[numeric_columns].round(2)

stats_df.to_csv(f'temperature_validation_stats_{YEAR}_Sept5_Dec5.csv', index=False)
print(f"\nSaved statistics to temperature_validation_stats_{YEAR}_Sept5_Dec5.csv")

# Plotting with original temporal resolution
if SHOW_DIFFERENCE_PLOT:
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), height_ratios=[2, 1])
else:
    fig, ax1 = plt.subplots(1, 1, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT/2))
plt.rcParams.update({'font.size': 14})

# Create second y-axis for flows
ax2 = ax1.twinx()

print("\nPlotting data...")
# Determine if we should use "Weighted" in labels
temp_label = "Weighted River Release Water Temperature (°C)" if has_spillway_flow else "River Release Water Temperature (°C)"

# Plot temperatures on primary axis
for run_id in MODEL_PATHS.keys():
    print(f"Plotting {run_id} with {len(model_results[run_id])} points")
    print(f"Date range for {run_id}: {model_results[run_id].index.min()} to {model_results[run_id].index.max()}")
    label = labels[run_id]
    if not has_spillway_flow:
        label = label.replace(" (Weighted)", "")
    ax1.plot(model_results[run_id].index, 
             model_results[run_id],
             color=colors[run_id],
             label=label)
    
    # Plot differences on the subplot if enabled
    if SHOW_DIFFERENCE_PLOT:
        common_dates = model_results[run_id].index.intersection(fwq_hourly.index)
        differences = model_results[run_id][common_dates] - fwq_hourly[common_dates]
        ax3.plot(common_dates, differences,
                 color=colors[run_id],
                 label=f"{label} - FWQ")

print(f"Plotting FWQ with {len(fwq_hourly)} points")
print(f"Date range for FWQ: {fwq_hourly.index.min()} to {fwq_hourly.index.max()}")
ax1.plot(fwq_hourly.index, 
         fwq_hourly,
         color=colors['FWQ'],
         label=labels['FWQ'])

# Plot flows on secondary axis
ax2.plot(flow_data.index, flow_data['Spill_Flow'] * 35.3147, 
         color='purple', linestyle='--', 
         label='Spillway Flow',
         zorder=9)
ax2.plot(flow_data.index, flow_data['Release_Flow'] * 35.3147, 
         color='orange', linestyle='--', 
         label='River Release Flow',
         zorder=9)

# After plotting all data, add the stats text box
plt.text(0.02, 0.98, stats_text.strip(),
         transform=ax1.transAxes,
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
         verticalalignment='top',
         fontsize=12)

# Format primary axis (temperature)
ax1.set_ylim(5, 20)  # Adjusted for Celsius
ax1.set_yticks(np.arange(5, 21, 1))  # Adjusted for Celsius
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.set_xlabel('Date', fontsize=14)
ax1.set_ylabel(temp_label, fontsize=14)  # Use conditional label

# Format secondary axis (flow)
ax2.set_ylabel('Flow (cfs)', fontsize=14)

# Format difference subplot if enabled
if SHOW_DIFFERENCE_PLOT:
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.set_xlabel('Date', fontsize=14)
    ax3.set_ylabel('Temperature Difference (°C)', fontsize=14)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)  # Add zero line

# Add monthly x-axis ticks to both plots
for ax in [ax1] + ([ax3] if SHOW_DIFFERENCE_PLOT else []):
    ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
legend = ax2.legend(lines1 + lines2, labels1 + labels2, 
           fontsize=14, loc='lower right',
           bbox_to_anchor=(1.0, 0.0),
           bbox_transform=ax1.transAxes,
           framealpha=0.85,
           edgecolor='gray',
           facecolor='white')
legend.set_frame_on(True)
legend.set_zorder(10)  # Set zorder after legend creation

# Add legend for difference subplot if enabled
if SHOW_DIFFERENCE_PLOT:
    ax3.legend(fontsize=14, loc='lower right')

# Adjust the plot margins to make room for the legend
plt.subplots_adjust(right=0.85, hspace=0.3 if SHOW_DIFFERENCE_PLOT else 0.1)

plt.tight_layout()  # This will ensure the rotated labels don't get cut off
plt.savefig(f'temperature_validation_{YEAR}.png', dpi=FIGURE_DPI, bbox_inches='tight')
plt.show() 