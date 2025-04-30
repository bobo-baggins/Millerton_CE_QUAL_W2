# Conerts CDEC SJA 146 Air Temp CSV from 15 min to daily averages 

import pandas as pd

# Read in xlsx file
df = pd.read_excel('SJA_146.xlsx', usecols=['DATE TIME', 'VALUE'], header=0, parse_dates=True)

# Convert to datetime
df['DATE TIME'] = pd.to_datetime(df['DATE TIME'])

# Set index to datetime
df.set_index('DATE TIME', inplace=True)

# Resample to daily averages
df = df.resample('D').mean()

# Write to csv
df.to_csv('SJA_146_Daily.csv', index=True, header=True)