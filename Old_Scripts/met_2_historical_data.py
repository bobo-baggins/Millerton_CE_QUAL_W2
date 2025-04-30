# November 27th 2023
# J.B. Hawkins
# Justin@McBaneAssociates.com

#This script pull values from the QAQC csvs and merge these rows into met_2003_2022.xlsx 
# while extending the date coloumn in met_2003_2022.xlsx
# This is the order of data rows in met_2003_2022.xlsx;
# Air Temperature (TAIR)
# Wind Speed (WIND)
# Atmospheric Pressure (PHI)
# Cloud Cover (CLOUD)
# Solar Radiation (SRO)

import pandas as pd
import numpy as np
import datetime as dt


#For testing 
# Start date of met_2003_2022.xlsx dn met_data 
start_date_hist = dt.datetime(2003, 1, 1, 12, 0)
start_jday = dt.timedelta(days=29951)
start_date_met = dt.datetime(2022, 7, 12, 12, 0)

# Calculate the difference between two dates
diff = start_date_met - start_date_hist
merge_Jday = diff + start_jday
# print(merge_Jday)

# Dictionary containing Met_data file paths and target coloumns
qaqc_files = {
    'met_data_update/QAQC/air_temp_avg_hourly_QAQC.csv': ['DATE TIME', 'Temp_C'],
    'met_data_update/QAQC/dewpoint_temp_hourly_QAQC.csv': [ 'DP_temp_C'],
    'met_data_update/QAQC/wind_speed_hourly_QAQC.csv': ['speed_ms'],
    'met_data_update/QAQC/atmospheric_pressure_hourly_QAQC.csv': ['Pressure_In'],
    'met_data_update/QAQC/cloud_cover_hourly_QAQC.csv': ['cloud_cover_10'],
    'met_data_update/QAQC/solar_rad_avg_hourly_QAQC.csv': ['Radiation_Wm2']
}

# List of coloumn names for met_2003_2022.xlsx used to rename met_data dfs
columns = [ 'DATE TIME', 'TAIR', 'TDEW', 'WIND', 'PHI', 'CLOUD', 'SRO'] 
# Step 1: Read QAQC data
met_dfs = [pd.read_csv(f, usecols=cols) for f, cols in qaqc_files.items()]

# Step 2: Read existing Excel file
excel_df = pd.read_excel('historical_data/met_2003_2022.xlsx')
# print(excel_df.head(5))

# for met_df in met_dfs:
#     print(met_df.tail(5))

# Step 3: Merge QAQC data into a single DataFrame
met_df = pd.concat(met_dfs, axis=1)

# Create a dictionary mapping old column names to new column names
column_mapping = dict(zip(met_df.columns, columns))

# Rename the columns
met_df = met_df.rename(columns=column_mapping)

# create  and inserting Julian Day Series
jday_met_series = np.arange(merge_Jday.days,(merge_Jday.days+0.04*met_df.shape[0]), step=0.04)
# print(len(jday_met_series))
met_df.insert(1, 'JDAY', jday_met_series)


# Drop the last row of met_df
met_df = met_df.drop(met_df.index[-1])
met_df['PHI'] = met_df['PHI'].interpolate(method='linear')

# Chopping off bottom of met_2003_2022.xlsx to match met_df
chop_to_row = excel_df[excel_df['DATE TIME'] == '2022-07-11 11:00:00'].index[0]
chopped_df = excel_df.loc[:chop_to_row]

merged_df = pd.concat([chopped_df, met_df], axis=0)
# print(merged_df.tail(10))

merged_df.to_csv('historical_data/met_2003_2023.csv', index=False)