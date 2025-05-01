import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv('SJA_Temp_2024.csv')

# Convert the date column to datetime
df['OBS DATE'] = pd.to_datetime(df['OBS DATE'])

# Set the datetime as index
df.set_index('OBS DATE', inplace=True)

# Resample to hourly frequency and calculate mean
hourly_df = df['VALUE'].resample('H').mean()

# Reset index to make datetime a column again
hourly_df = hourly_df.reset_index()

# Save to new CSV file
hourly_df.to_csv('SJA_Temp_2024_hourly.csv', index=False)

print("Conversion complete! Data saved to 'SJA_Temp_2024_hourly.csv'") 