import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Read both CSV files
hourly_df = pd.read_csv('SJA_Temp_2024_hourly.csv')
daily_df = pd.read_csv('SJA_2024_temp.csv')

# Print column names to debug
print("Hourly DataFrame columns:", hourly_df.columns.tolist())
print("Daily DataFrame columns:", daily_df.columns.tolist())

# Convert date columns to datetime
hourly_df['DateTime'] = pd.to_datetime(hourly_df['DateTime'])
daily_df['Date'] = pd.to_datetime(daily_df['Date'])

# Create a date column without time for hourly data
hourly_df['date_only'] = hourly_df['DateTime'].dt.date
daily_df['date_only'] = daily_df['Date'].dt.date

# Merge the daily average with hourly data
merged_df = pd.merge(
    hourly_df,
    daily_df[['date_only', '2024_Temp']],
    on='date_only',
    how='left'
)

# Rename columns for clarity
merged_df = merged_df.rename(columns={
    '2024_Temp_x': 'Hourly_Temp',
    '2024_Temp_y': 'Daily_Avg'
})

# Calculate the difference
merged_df['Difference'] = merged_df['Hourly_Temp'] - merged_df['Daily_Avg']

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), height_ratios=[2, 1])

# Plot 1: Temperatures
ax1.plot(merged_df['DateTime'], merged_df['Hourly_Temp'], 
         label='Hourly Temperature', color='blue', alpha=0.5)
ax1.plot(merged_df['DateTime'], merged_df['Daily_Avg'], 
         label='Daily Average', color='red', linewidth=2)
ax1.set_title('Hourly vs Daily Average Temperatures - 2024', fontsize=14)
ax1.set_ylabel('Temperature (°C)', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)

# Plot 2: Differences
ax2.plot(merged_df['DateTime'], merged_df['Difference'], 
         label='Temperature Difference', color='green')
ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
ax2.set_title('Difference Between Hourly and Daily Temperatures', fontsize=14)
ax2.set_xlabel('Date', fontsize=12)
ax2.set_ylabel('Difference (°C)', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
plt.savefig('temperature_comparison.png', dpi=300, bbox_inches='tight')

# Print statistics
print("\nTemperature Comparison Statistics:")
print(f"Mean difference: {merged_df['Difference'].mean():.2f}°C")
print(f"Max difference: {merged_df['Difference'].max():.2f}°C")
print(f"Min difference: {merged_df['Difference'].min():.2f}°C")
print(f"Standard deviation: {merged_df['Difference'].std():.2f}°C")

# Print the first few rows
print("\nFirst few rows of the comparison:")
print(merged_df[['DateTime', 'Hourly_Temp', 'Daily_Avg', 'Difference']].head()) 