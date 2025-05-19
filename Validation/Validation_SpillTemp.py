# This script calculates weighted temperatures using downstream and lateral flows
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def jday_to_datetime(jday):
    # For JDAY = 37293.042 to be Feb 7, 2023 1:00 AM
    base_date = datetime(1921, 1, 1)  # Start from Jan 1, 1921
    days_since_base = float(jday)
    fractional_day = days_since_base % 1  # Get just the decimal part for hours
    hours = fractional_day * 24  # Convert decimal days to hours
    
    # Calculate the date and time
    date_time = base_date + timedelta(days=int(days_since_base), hours=hours)
    return date_time

    # To verify:
    # 37293.042 days from Dec 31, 1900 should be:
    # 37293.042/24 ≈ 1553.876 days
    # 1900 + 1553.876 ≈ 2023 (early February)

def read_and_process_files(temp_file, flow_file):
    # Read temperature file with all columns
    temp_df = pd.read_csv(temp_file, 
                         skiprows=3,
                         names=['JDAY', 'T(C)', 'Spillway_T', 'FKC_T', 'Madera_T', 'SJR_Out_T'],
                         skipinitialspace=True,
                         delim_whitespace=False,
                         index_col=False,
                         thousands=',',
                         on_bad_lines='skip',
                         dtype={'JDAY': str})  # Read JDAY as string initially

    
    # Convert JDAY to float first
    temp_df['JDAY'] = pd.to_numeric(temp_df['JDAY'].str.strip(), errors='coerce')
    

    
    # Then convert to datetime
    temp_df['datetime'] = temp_df['JDAY'].apply(jday_to_datetime)
    
    # Print resulting dates
    print("\nConverted dates:")
    print(temp_df['datetime'].head().dt.strftime('%Y-%m-%d %H:%M:%S'))
    
    # Read flow file with all columns
    flow_df = pd.read_csv(flow_file, skiprows=3,  # Skip 3 rows now to include the header
                         names=['JDAY', 'QWD(m3s-1)', 'Spillway_Q', 'FKC_Q', 'Madera_Q', 'SJR_Out_Q'],
                         skipinitialspace=True,
                         delim_whitespace=False,
                         on_bad_lines='skip',
                         index_col=False,
                         thousands=',',
                         dtype={'JDAY': str})
    print("Flow data:")
    print(flow_df.head())
    # Convert JDAY to float first
    flow_df['JDAY'] = pd.to_numeric(flow_df['JDAY'], errors='coerce')
    
    # Then convert to datetime
    flow_df['datetime'] = flow_df['JDAY'].apply(jday_to_datetime)
    
    # Print first few dates to verify
    print("First few dates in temperature data:")
    print(temp_df['datetime'].head())
    
    # Keep only needed columns
    temp_df = temp_df[['datetime', 'Spillway_T', 'SJR_Out_T']]
    flow_df = flow_df[['datetime', 'Spillway_Q', 'SJR_Out_Q']]
    
    # Handle -99 values in temperature data
    temp_df = temp_df.replace(-99.00, np.nan)
    
    # Merge dataframes
    merged_df = pd.merge(temp_df, flow_df, on='datetime')
    
    # Calculate weighted temperatures (intermediate columns)
    merged_df['weighted_temp_spillway'] = merged_df['Spillway_T'] * merged_df['Spillway_Q']
    merged_df['weighted_temp_sjr_out'] = merged_df['SJR_Out_T'] * merged_df['SJR_Out_Q']
    
    # Calculate final weighted average
    merged_df['weighted_avg'] = np.where(
        merged_df['Spillway_Q'] == 0,
        merged_df['SJR_Out_T'],
        (merged_df['weighted_temp_spillway'] + merged_df['weighted_temp_sjr_out']) / 
        (merged_df['Spillway_Q'] + merged_df['SJR_Out_Q'])
    )
    
    # Remove intermediate calculation columns
    final_columns = ['datetime', 'Spillway_T', 'SJR_Out_T', 'Spillway_Q', 'SJR_Out_Q', 'weighted_avg']
    merged_df = merged_df[final_columns]
    
    return merged_df

if __name__ == "__main__":
    # Get run name from user
    run_name = input("Enter the name for this run: ")
    
    # File paths
    temp_file = '../CEQUAL_model/two_31.csv'
    flow_file = '../CEQUAL_model/qwo_31.csv'
    
    # Process files
    result_df = read_and_process_files(temp_file, flow_file)
    
    # Save full results with run name
    full_results_filename = f'weighted_temperature_results_{run_name}.csv'
    result_df.to_csv(full_results_filename, index=False)
    print(f"Full results saved to: {full_results_filename}")
    
    # Create daily averages of weighted temperature
    # Convert datetime to date for grouping
    result_df['date'] = result_df['datetime'].dt.date
    daily_avg = result_df.groupby('date')['weighted_avg'].mean().reset_index()
    
    # Save daily averages with run name in filename
    daily_avg_filename = f'T_Weighted_{run_name}.csv'
    daily_avg.to_csv(daily_avg_filename, index=False)
    print(f"Daily averages saved to: {daily_avg_filename}")
