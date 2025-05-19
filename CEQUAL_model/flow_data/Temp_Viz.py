import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Set global font size
plt.rcParams.update({'font.size': 14})

# Define colors for each temperature series
PLOT_COLORS = {
    'SJA': 'green',
    'K2P': 'red',
    'Weighted': 'purple'
}

# Read the CSV files
k2p_temp = pd.read_csv('K2P_2024_Temp.csv')
k2p_weighted = pd.read_csv('2024_W_Temp.csv')
sja_temp = pd.read_csv('SJA_2024_Temp.csv')

# Convert first column to datetime and handle NaT values
for df in [k2p_temp, k2p_weighted, sja_temp]:
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
    # Remove any rows with NaT values
    df.dropna(subset=[df.columns[0]], inplace=True)

# After reading and processing the CSV files, add a filter for dates after July 1st
start_date = pd.to_datetime('2024-07-01')

# Filter each dataframe to only include dates from July 1st onwards
k2p_temp = k2p_temp[k2p_temp.iloc[:, 0] >= start_date]
k2p_weighted = k2p_weighted[k2p_weighted.iloc[:, 0] >= start_date]
sja_temp = sja_temp[sja_temp.iloc[:, 0] >= start_date]

# Create the plot
plt.figure(figsize=(12, 6))

# Plot each temperature series
plt.plot(sja_temp.iloc[:, 0].values, 
         sja_temp.iloc[:, 1].values, 
         label='SJA Water Temperature',
         color=PLOT_COLORS['SJA'])
plt.plot(k2p_temp.iloc[:, 0].values, 
         k2p_temp.iloc[:, 1].values, 
         label='K2P Water Temperature',
         color=PLOT_COLORS['K2P'])
plt.plot(k2p_weighted.iloc[:, 0].values, 
         k2p_weighted.iloc[:, 1].values, 
         label='Weighted Water Temperature',
         color=PLOT_COLORS['Weighted'])

# Format x-axis to show dates nicely
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

# Set the x-axis limits explicitly
plt.xlim(start_date, pd.to_datetime('2024-12-31'))

# Add vertical dashed line at July 30th with label
july_30 = pd.to_datetime('2024-07-30').date()
plt.axvline(x=july_30, color='black', linestyle='--', alpha=0.7)

# Convert the annotation coordinates (keeping in Celsius)
plt.annotate('K2P Data Available\nbeginning July 30th', 
            xy=(july_30, 14),  # Using Celsius value
            xytext=(july_30 + pd.Timedelta(days=25), 12),  # Using Celsius value
            bbox=dict(facecolor='white', edgecolor='black', alpha=0.8),
            horizontalalignment='left',
            verticalalignment='top',
            arrowprops=dict(arrowstyle='-', color='black', linewidth=1),
            fontsize=14)

# Customize the plot
plt.title('2024 Inflow Temperature Comparison', fontsize=14)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Daily Average Water Temperature (°C)', fontsize=14)
plt.legend(fontsize=14)
plt.grid(True)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, fontsize=14)

# After the plotting commands but before plt.show(), add:
# Get the current y-axis limits
y_min, y_max = plt.ylim()
# Create ticks at 1-degree intervals, rounding to nearest degree for min/max
y_ticks = range(int(y_min), int(y_max) + 1)
plt.yticks(y_ticks, fontsize=14)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Show the plot
plt.show()
