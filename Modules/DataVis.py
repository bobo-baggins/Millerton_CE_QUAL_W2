import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib import patheffects  # Add this import at the top of the file

def julian_to_date(julian_day, year):
    """Convert Julian day to datetime using 1921 as reference year"""
    start_date = datetime(1921, 1, 1)
    return start_date + timedelta(days=float(julian_day) - 1)

def calculate_weighted_temp(spill_temp, release_temp, spill_flow, release_flow):
    """
    Calculate weighted temperature based on spill and release flows
    
    Parameters:
    -----------
    spill_temp : array-like
        Spill temperature values
    release_temp : array-like
        Release temperature values
    spill_flow : array-like
        Spill flow values
    release_flow : array-like
        Release flow values
        
    Returns:
    --------
    array-like
        Weighted temperature values
    """
    weighted_temp = np.zeros_like(spill_temp)
    for i in range(len(spill_temp)):
        if spill_flow[i] == 0:
            weighted_temp[i] = release_temp[i]
        else:
            weighted_temp[i] = ((spill_temp[i] * spill_flow[i]) + 
                              (release_temp[i] * release_flow[i])) / (spill_flow[i] + release_flow[i])
    return weighted_temp

def calculate_tcd_weighted_temp(spill_temp, release_temp, madera_temp, spill_flow, release_flow, jdays, year):
    """
    Calculate TCD weighted temperature based on spill and release flows with date-based adjustments
    
    Parameters:
    -----------
    spill_temp : array-like
        Spill temperature values
    release_temp : array-like
        Release temperature values
    madera_temp : array-like
        Madera temperature values
    spill_flow : array-like
        Spill flow values
    release_flow : array-like
        Release flow values
    jdays : array-like
        Julian days for each data point
    year : int
        Year for the simulation
        
    Returns:
    --------
    array-like
        TCD weighted temperature values
    """
    # Convert Julian days to datetime objects
    dates = [julian_to_date(jday, year) for jday in jdays]
    
    # Initialize TCD temperature array
    tcd_temp = np.zeros_like(spill_temp)
    
    # Process each time step
    for i in range(len(spill_temp)):
        current_date = dates[i]
        current_month = current_date.month
        
        # Apply flow adjustments based on month
        if 3 <= current_month <= 6:  # March to June
            flow_adjustment = 9.91
            adjustment_temp = madera_temp[i]  # Use Madera temp for 350 cfs adjustment
        elif current_month == 7:  # July
            flow_adjustment = 4.25
            adjustment_temp = madera_temp[i]  # Use Madera temp for 150 cfs adjustment
        elif current_month in [8]:  # August
            flow_adjustment = 2.83
            adjustment_temp = madera_temp[i]  # Use Madera temp for 100 cfs adjustment
        else:
            flow_adjustment = 0
            adjustment_temp = 0
            
        # Add flow adjustment to release flow
        adjusted_release_flow = release_flow[i] + flow_adjustment
        
        # Calculate total flow for weighting
        total_flow = spill_flow[i] + adjusted_release_flow
        
        # Calculate weighted average temperature including all components
        # Only include spillway component if there's spillway flow
        weighted_sum = (release_temp[i] * release_flow[i]) + \
                      (adjustment_temp * flow_adjustment)
        
        if spill_flow[i] > 0:
            weighted_sum += (spill_temp[i] * spill_flow[i])
        
        tcd_temp[i] = weighted_sum / total_flow if total_flow > 0 else release_temp[i]
        
        # Print debug info only for August
        if current_month == 8:
            print(f"\nTimestamp: {current_date}")
            print(f"Flow adjustment: {flow_adjustment}")
            print(f"Adjusted release flow: {adjusted_release_flow}")
            print(f"Total flow: {total_flow}")
            print(f"Release temp: {release_temp[i]}")
            print(f"Madera temp: {madera_temp[i]}")
            print(f"Spill flow: {spill_flow[i]}")
            print(f"Spill temp: {spill_temp[i]}")
            print(f"Weighted sum: {weighted_sum}")
            print(f"TCD temp: {tcd_temp[i]}")
    
    return tcd_temp

def count_days_above_threshold(df, threshold):
    """
    Count the number of days where temperature is above the threshold
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'Temperature' column
    threshold : float
        Temperature threshold in Celsius
        
    Returns:
    --------
    int
        Number of days above threshold
    """
    return (df['Temperature'] > threshold).sum()

def plot_temperature_comparison(file_a, file_b, file_c=None, year=None, jdays=None, 
                              scenario_a_name="Scenario A", scenario_b_name="Scenario B", 
                              scenario_c_name="Scenario C"):
    """
    Plot temperature comparisons for different scenarios using provided Julian days
    
    Parameters:
    -----------
    file_a, file_b : str
        Paths to the CSV files containing temperature data for scenarios A and B
    file_c : str, optional
        Path to the CSV file containing temperature data for scenario C
    year : int
        Year for the simulation
    jdays : array-like
        Array of Julian days from the simulation
    scenario_a_name : str, optional
        Name for scenario A (default: "Scenario A")
    scenario_b_name : str, optional
        Name for scenario B (default: "Scenario B")
    scenario_c_name : str, optional
        Name for scenario C (default: "Scenario C")
    """
    # DPI configuration for output image
    FIGURE_DPI = 300  # Dots per inch for the output image
    
    # Read temperature CSV files, skipping first 3 rows
    df_a = pd.read_csv(file_a, skiprows=3, header=None)
    df_b = pd.read_csv(file_b, skiprows=3, header=None)
    
    # Read flow CSV files
    flow_file_a = file_a.replace('two_31', 'qwo_31')
    flow_file_b = file_b.replace('two_31', 'qwo_31')
    
    flow_df_a = pd.read_csv(flow_file_a, skiprows=3, header=None)
    flow_df_b = pd.read_csv(flow_file_b, skiprows=3, header=None)
    
    # Initialize lists for data
    dfs = [df_a, df_b]
    flow_dfs = [flow_df_a, flow_df_b]
    scenario_names = [scenario_a_name, scenario_b_name]
    colors = ['blue', 'red']
    
    # Add scenario C if provided
    if file_c is not None:
        df_c = pd.read_csv(file_c, skiprows=3, header=None)
        flow_file_c = file_c.replace('two_31', 'qwo_31')
        flow_df_c = pd.read_csv(flow_file_c, skiprows=3, header=None)
        dfs.append(df_c)
        flow_dfs.append(flow_df_c)
        scenario_names.append(scenario_c_name)
        colors.append('green')
    
    # Clean up the data - remove commas and convert to numeric
    for df in dfs + flow_dfs:
        for col in df.columns:
            # Check if the column is string type before trying to strip commas
            if df[col].dtype == 'object':
                df[col] = df[col].str.rstrip(',').astype(float)
            else:
                df[col] = df[col].astype(float)
        df[0] = df[0].astype(float)  # JDAY doesn't need comma removal
    
    # Extract data for each scenario
    weighted_temps = []
    spill_flows = []
    dates = []
    
    for i, (df, flow_df) in enumerate(zip(dfs, flow_dfs)):
        time = jdays
        spill_temp = df[2]  # 3rd column (spill temperature)
        release_temp = df[5]  # 6th column (release temperature)
        spill_flow = flow_df[2]  # 3rd column (spill flow)
        release_flow = flow_df[5]  # 6th column (release flow)
        
        # Convert -99 values to NaN
        spill_temp[spill_temp == -99] = np.nan
        release_temp[release_temp == -99] = np.nan
        
        # Calculate weighted temperature
        weighted_temp = calculate_weighted_temp(spill_temp, release_temp, spill_flow, release_flow)
        weighted_temps.append(weighted_temp)
        spill_flows.append(spill_flow)
        dates.append([julian_to_date(jday, year) for jday in time])
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    # Set font to Arial
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 14
    
    # Plot weighted temperatures on first subplot
    for i, (temp, date, name, color) in enumerate(zip(weighted_temps, dates, scenario_names, colors)):
        ax1.plot(date, temp, label=name, color=color, linestyle='-')
    
    # Check if there's any spillway flow
    has_spillway_flow = any((flow > 0).any() for flow in spill_flows)
    
    # Format first subplot
    ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %d'))
    ax1.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
    ax1.set_ylabel('River Release Temperature (°C)' if not has_spillway_flow else 'Weighted Temperature (°C)', fontsize=14)
    ax1.set_title(f'River Release Temperature Comparison for {year}' if not has_spillway_flow else f'Weighted Temperature Comparison for {year}', fontsize=14)
    ax1.grid(True)
    ax1.legend(fontsize=14)
    
    # Plot differences on second subplot
    for i in range(1, len(weighted_temps)):
        ax2.plot(dates[0], weighted_temps[i] - weighted_temps[0], 
                label=f'{scenario_names[i]} - {scenario_names[0]}', 
                color=colors[i])
    
    # Format second subplot
    ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %d'))
    ax2.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
    ax2.set_xlabel('Date', fontsize=14)
    ax2.set_ylabel('Temperature Difference (°C)', fontsize=14)
    ax2.grid(True)
    ax2.legend(fontsize=14)
    
    # Rotate and align the tick labels
    plt.gcf().autofmt_xdate()
    
    # Calculate statistics for September 5th to November 5th
    start_date = datetime(year, 9, 5)
    end_date = datetime(year, 12, 5)
    
    # Create DataFrames for each scenario
    dfs_stats = []
    for date, temp in zip(dates, weighted_temps):
        df = pd.DataFrame({'Date': date, 'Temperature': temp})
        df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
        dfs_stats.append(df)
    
    # Calculate statistics
    stats = {
        'Scenario': scenario_names,
        'Max_Temperature': [df['Temperature'].max() for df in dfs_stats],
        'Min_Temperature': [df['Temperature'].min() for df in dfs_stats],
        'Mean_Temperature': [df['Temperature'].mean() for df in dfs_stats],
        'Days_Above_14.44C': [count_days_above_threshold(df, 14.44)/24 for df in dfs_stats]
    }
    
    # Create DataFrame from statistics
    stats_df = pd.DataFrame(stats)
    
    # Create text box content
    text_lines = [f'Statistics (9/5 - 12/5):', '']  # Add blank line
    for i, name in enumerate(scenario_names):
        text_lines.extend([
            f'{name}:',
            f'  Max: {stats["Max_Temperature"][i]:.2f}°C',
            f'  Min: {stats["Min_Temperature"][i]:.2f}°C',
            f'  Mean: {stats["Mean_Temperature"][i]:.2f}°C',
            f'  Days > 14.44°C: {stats["Days_Above_14.44C"][i]:.2f}',
            ''  # Add blank line
        ])
    textstr = '\n'.join(text_lines)
    
    # Add text box to the plot
    props = dict(boxstyle='round', facecolor='white', alpha=0.95)
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=16,
             verticalalignment='top', bbox=props)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save the plot
    output_dir = f'CEQUAL_outputs/{year}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f'temperature_comparison_{timestamp}.png')
    
    # Save with high DPI for better quality
    plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
    
    # Create DataFrame with weighted temperatures and dates
    weighted_data = pd.DataFrame({
        'DateTime': dates[0],
        **{f'{name}_Temp': temp for name, temp in zip(scenario_names, weighted_temps)}
    })
    
    # Save weighted temperatures to CSV
    weighted_csv = os.path.join(output_dir, f'weighted_temperatures_{timestamp}.csv')
    weighted_data.to_csv(weighted_csv, index=False)
    print(f"\nWeighted temperatures saved to: {weighted_csv}")
    
    # Save statistics to CSV
    stats_file = os.path.join(output_dir, f'temperature_stats_{timestamp}.csv')
    stats_df.to_csv(stats_file, index=False)
    print(f"\nStatistics saved to: {stats_file}")
    
    return plt

def plot_tcd_comparison(file_a, file_b, file_c=None, year=None, jdays=None, 
                       scenario_a_name="Scenario A", scenario_b_name="Scenario B", 
                       scenario_c_name="Scenario C"):
    """
    Plot temperature comparisons where one scenario uses TCD weighted temperature and the other uses regular weighted temperature
    
    Parameters:
    -----------
    file_a, file_b : str
        Paths to the CSV files containing temperature data for scenarios A and B
    file_c : str, optional
        Path to the CSV file containing temperature data for scenario C
    year : int
        Year for the simulation
    jdays : array-like
        Array of Julian days from the simulation
    scenario_a_name : str, optional
        Name for scenario A (default: "Scenario A")
    scenario_b_name : str, optional
        Name for scenario B (default: "Scenario B")
    scenario_c_name : str, optional
        Name for scenario C (default: "Scenario C")
    """
    if year is None or jdays is None:
        raise ValueError("year and jdays parameters are required for TCD calculations")
        
    # DPI configuration for output image
    FIGURE_DPI = 300  # Dots per inch for the output image
    
    # Read temperature CSV files, skipping first 3 rows
    df_a = pd.read_csv(file_a, skiprows=3, header=None)
    df_b = pd.read_csv(file_b, skiprows=3, header=None)
    
    # Read flow CSV files
    flow_file_a = file_a.replace('two_31', 'qwo_31')
    flow_file_b = file_b.replace('two_31', 'qwo_31')
    
    flow_df_a = pd.read_csv(flow_file_a, skiprows=3, header=None)
    flow_df_b = pd.read_csv(flow_file_b, skiprows=3, header=None)
    
    # Initialize lists for data
    dfs = [df_a, df_b]
    flow_dfs = [flow_df_a, flow_df_b]
    scenario_names = [scenario_a_name, scenario_b_name]
    colors = ['blue', 'red']
    
    # Add scenario C if provided
    if file_c is not None:
        df_c = pd.read_csv(file_c, skiprows=3, header=None)
        flow_file_c = file_c.replace('two_31', 'qwo_31')
        flow_df_c = pd.read_csv(flow_file_c, skiprows=3, header=None)
        dfs.append(df_c)
        flow_dfs.append(flow_df_c)
        scenario_names.append(scenario_c_name)
        colors.append('green')
    
    # Clean up the data - remove commas and convert to numeric
    for df in dfs + flow_dfs:
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.rstrip(',').astype(float)
            else:
                df[col] = df[col].astype(float)
        df[0] = df[0].astype(float)  # JDAY doesn't need comma removal
    
    # Extract data for each scenario
    weighted_temps = []
    spill_flows = []
    dates = []
    
    # Get the length of the shorter array (Madera scenario)
    min_length = min(len(df[0]) for df in dfs)
    
    print("\nCalculating weighted temperatures:")
    for i, (df, flow_df) in enumerate(zip(dfs, flow_dfs)):
        # Use the Julian days from each file instead of the input jdays
        time = df[0][:min_length]  # Trim to shorter length
        spill_temp = df[2][:min_length]  # Trim to shorter length
        release_temp = df[5][:min_length]  # Trim to shorter length
        madera_temp = df[4][:min_length]  # Trim to shorter length
        spill_flow = flow_df[2][:min_length]  # Trim to shorter length
        release_flow = flow_df[5][:min_length]  # Trim to shorter length
        
        # Convert -99 values to NaN
        spill_temp[spill_temp == -99] = np.nan
        release_temp[release_temp == -99] = np.nan
        madera_temp[madera_temp == -99] = np.nan
        
        # Calculate weighted temperature - use TCD for scenario B, regular for scenario A
        if i == 1:  # Scenario B (Madera)
            print(f"\nCalculating TCD weighted temperature for {scenario_names[i]}")
            weighted_temp = calculate_tcd_weighted_temp(spill_temp, release_temp, madera_temp, 
                                                     spill_flow, release_flow, time, year)
            print(f"TCD weighted temperature array length: {len(weighted_temp)}")
            print(f"First few TCD weighted temperatures: {weighted_temp[:5]}")
        else:  # Scenario A (Obs)
            print(f"\nCalculating regular weighted temperature for {scenario_names[i]}")
            weighted_temp = calculate_weighted_temp(spill_temp, release_temp, spill_flow, release_flow)
            print(f"Regular weighted temperature array length: {len(weighted_temp)}")
            print(f"First few regular weighted temperatures: {weighted_temp[:5]}")
            
        weighted_temps.append(weighted_temp)
        spill_flows.append(spill_flow)
        dates.append([julian_to_date(jday, year) for jday in time])
    
    # Create DataFrame with both weighted temperatures and datetime
    comparison_df = pd.DataFrame({
        'DateTime': dates[0],
        f'{scenario_names[0]}_Temp': weighted_temps[0],
        f'{scenario_names[1]}_Temp': weighted_temps[1],
        'Spill_Flow': spill_flows[0],
        'Release_Flow': release_flow,
        'Madera_Temp': madera_temp
    })
    
    # Save the comparison data to CSV
    output_dir = f'CEQUAL_outputs/{year}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    comparison_csv = os.path.join(output_dir, f'temperature_comparison_detailed_{timestamp}.csv')
    comparison_df.to_csv(comparison_csv, index=False)
    print(f"\nDetailed temperature comparison saved to: {comparison_csv}")
    
    # Continue with the rest of the plotting code...
    print("\nPlotting temperatures:")
    for i, (temp, date, name) in enumerate(zip(weighted_temps, dates, scenario_names)):
        print(f"{name} temperature array length: {len(temp)}")
        print(f"{name} dates array length: {len(date)}")
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    # Set font to Arial
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 14
    
    # Plot weighted temperatures on first subplot
    for i, (temp, date, name, color) in enumerate(zip(weighted_temps, dates, scenario_names, colors)):
        print(f"Plotting {name} with color {color}")
        ax1.plot(date, temp, label=name, color=color, linestyle='-')
    
    # Format first subplot
    ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %d'))
    ax1.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
    ax1.set_ylabel('Temperature (°C)', fontsize=14)
    ax1.set_title(f'Temperature Comparison for {year} (Obs vs TCD)', fontsize=14)
    ax1.grid(True)
    ax1.legend(fontsize=14)
    
    # Plot differences on second subplot
    for i in range(1, len(weighted_temps)):
        ax2.plot(dates[0], weighted_temps[i] - weighted_temps[0], 
                label=f'{scenario_names[i]} - {scenario_names[0]}', 
                color=colors[i])
    
    # Format second subplot
    ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %d'))
    ax2.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
    ax2.set_xlabel('Date', fontsize=14)
    ax2.set_ylabel('Temperature Difference (°C)', fontsize=14)
    ax2.grid(True)
    ax2.legend(fontsize=14)
    
    # Rotate and align the tick labels
    plt.gcf().autofmt_xdate()
    
    # Calculate statistics for September 5th to November 5th
    start_date = datetime(year, 9, 5)
    end_date = datetime(year, 12, 5)
    
    # Create DataFrames for each scenario
    dfs_stats = []
    for date, temp in zip(dates, weighted_temps):
        df = pd.DataFrame({'Date': date, 'Temperature': temp})
        df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
        dfs_stats.append(df)
    
    # Calculate statistics
    stats = {
        'Scenario': scenario_names,
        'Max_Temperature': [df['Temperature'].max() for df in dfs_stats],
        'Min_Temperature': [df['Temperature'].min() for df in dfs_stats],
        'Mean_Temperature': [df['Temperature'].mean() for df in dfs_stats],
        'Days_Above_14.44C': [count_days_above_threshold(df, 14.44)/24 for df in dfs_stats]
    }
    
    # Create DataFrame from statistics
    stats_df = pd.DataFrame(stats)
    
    # Create text box content
    text_lines = [f'Statistics (9/5 - 12/5):', '']  # Add blank line
    for i, name in enumerate(scenario_names):
        text_lines.extend([
            f'{name}:',
            f'  Max: {stats["Max_Temperature"][i]:.2f}°C',
            f'  Min: {stats["Min_Temperature"][i]:.2f}°C',
            f'  Mean: {stats["Mean_Temperature"][i]:.2f}°C',
            f'  Days > 14.44°C: {stats["Days_Above_14.44C"][i]:.2f}',
            ''  # Add blank line
        ])
    textstr = '\n'.join(text_lines)
    
    # Add text box to the plot
    props = dict(boxstyle='round', facecolor='white', alpha=0.95)
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=16,
             verticalalignment='top', bbox=props)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save the plot
    output_file = os.path.join(output_dir, f'temperature_comparison_obs_vs_tcd_{timestamp}.png')
    
    # Save with high DPI for better quality
    plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
    
    # Save statistics to CSV
    stats_file = os.path.join(output_dir, f'temperature_stats_obs_vs_tcd_{timestamp}.csv')
    stats_df.to_csv(stats_file, index=False)
    print(f"\nStatistics saved to: {stats_file}")
    
    return plt

def QAQC_plot(csv_file, year, run_name):
    """
    Create a QAQC plot of water temperature, flow, and WSE over time for a single run
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file containing temperature data (two_31_[Year]_[Run].csv)
    year : int
        Year for the simulation
    run_name : str
        Name of the run/scenario
    """
    # Read temperature CSV file, skipping first 3 rows
    df_temp = pd.read_csv(csv_file, skiprows=3, header=None)
    
    # Read flow CSV file, skipping first 3 rows
    flow_file = csv_file.replace('two_31', 'qwo_31')
    df_flow = pd.read_csv(flow_file, skiprows=3, header=None)
    
    # Read WSE CSV file with headers
    wse_file = csv_file.replace('two_31', 'tsr_1_seg31')
    print(f"\nReading WSE file: {wse_file}")
    print("First few rows of WSE file:")
    print(pd.read_csv(wse_file, nrows=5))  # Print first 5 rows
    df_wse = pd.read_csv(wse_file)  # Read with headers
    
    # Extract JDAY (1st column) and temperature data (6th column)
    jdays = df_temp[0]  # First column (JDAY)
    temperature = df_temp[5]  # 6th column (temperature)
    
    # Extract flow data (3rd and 6th columns) and convert from cms to cfs
    flow1 = df_flow[2] * 35.3147  # 3rd column (first flow) converted to cfs
    flow2 = df_flow[5] * 35.3147  # 6th column (second flow) converted to cfs
    
    # # Interpolate WSE data to match hourly resolution and convert from meters to feet
    # wse_interp = np.interp(jdays, df_wse['JDAY'], df_wse['ELWS(m)']) * 3.28084
    
    # Debug prints for array shapes
    print(f"\nArray shapes:")
    print(f"Temperature array shape: {temperature.shape}")
    print(f"Flow1 array shape: {flow1.shape}")
    print(f"Flow2 array shape: {flow2.shape}")
    # print(f"WSE array shape: {wse_interp.shape}")
    print(f"Jdays array shape: {jdays.shape}")
    
    # Convert Julian days to dates
    dates = [julian_to_date(jday, year) for jday in jdays]
    
    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(15, 8))
    
    # Create secondary axes
    ax2 = ax1.twinx()    # Flow axis
    # ax3 = ax1.twinx()    # WSE axis
    
    # Position the axes
    ax2.spines['right'].set_position(('axes', 1.0))      # Flow on right
    # ax3.spines['right'].set_position(('axes', 1.1))      # WSE on far right
    
    # Convert temperature to Fahrenheit
    temperature_f = temperature * 9/5 + 32
    
    # Plot temperature on primary y-axis (Fahrenheit)
    temp_line = ax1.plot(dates, temperature_f, label='River Release Outflow Temperature', color='blue')
    
    # Get the current Fahrenheit limits and round them
    f_min, f_max = ax1.get_ylim()
    f_min = np.floor(f_min)
    f_max = np.ceil(f_max)
    
    # Set the Fahrenheit ticks and labels
    f_ticks = np.arange(f_min, f_max + 1, 2)  # Increment by 2 for cleaner ticks
    ax1.set_yticks(f_ticks)
    ax1.set_yticklabels([f'{tick:.0f}°F' for tick in f_ticks])
    
    # Set the labels
    ax1.set_ylabel('Temperature (°F)', color='blue', labelpad=15)
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Plot flow on secondary y-axis
    flow1_line = ax2.plot(dates, flow1, label='Spillway Outflow', color='red', alpha=0.7)
    flow2_line = ax2.plot(dates, flow2, label='River Release Outflow', color='green', alpha=0.7)
    ax2.set_ylabel('Flow (cfs)', color='red', labelpad=15)
    ax2.tick_params(axis='y', labelcolor='red')
    
    # # Plot WSE on tertiary y-axis
    # wse_line = ax3.plot(dates, wse_interp, label='WSE at Dam', color='purple', alpha=0.7)
    # ax3.set_ylabel('WSE (ft)', color='purple', labelpad=15)
    # ax3.tick_params(axis='y', labelcolor='purple')
    
    # Format x-axis
    ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %d'))
    ax1.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
    ax1.set_xlabel('Date')
    
    # Add title and grid
    plt.title(f'Water Temperature and Flow QAQC Plot - {year} {run_name}')
    ax1.grid(True)
    
    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    # lines3, labels3 = ax3.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Rotate and align the tick labels
    plt.gcf().autofmt_xdate()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    output_dir = f'../CEQUAL_outputs/{year}/{run_name}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create filename with year and run name
    output_file = os.path.join(output_dir, f'{year}_{run_name}_QAQC_plot.png')
    
    # Save with high DPI for better quality
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"QAQC plot saved to: {output_file}")
    
    return plt

def Analog_Post_Plot(model_run_a, model_run_b, input_dir, scenario_a_name="Scenario A", scenario_b_name="Scenario B"):
    """
    Create comparison plots for weighted temperatures between two model runs using weighted data files
    
    Parameters:
    -----------
    model_run_a : str
        Name of the first model run (e.g., '2018_Analog_Scenario_A')
    model_run_b : str
        Name of the second model run (e.g., '2018_Analog_Scenario_B')
    input_dir : str
        Directory containing the input weighted data files
    scenario_a_name : str, optional
        Display name for the first scenario (default: "Scenario A")
    scenario_b_name : str, optional
        Display name for the second scenario (default: "Scenario B")
    """
    # Define the path to the weighted data files
    file_a = os.path.join(input_dir, f'{model_run_a}_weighted.csv')
    file_b = os.path.join(input_dir, f'{model_run_b}_weighted.csv')
    
    # Print debug information
    print(f"\nLooking for files in:")
    print(f"Input directory: {input_dir}")
    print(f"File A: {file_a}")
    print(f"File B: {file_b}")
    
    # Check if files exist
    if not os.path.exists(file_a) or not os.path.exists(file_b):
        raise ValueError(f"Could not find weighted data files for {model_run_a} and/or {model_run_b}")
    
    # Load the data
    df_a = pd.read_csv(file_a, parse_dates=['DateTime'])
    df_b = pd.read_csv(file_b, parse_dates=['DateTime'])
    
    # Find weighted temperature columns
    temp_cols_a = [col for col in df_a.columns if col.startswith('Weighted_Temp')]
    temp_cols_b = [col for col in df_b.columns if col.startswith('Weighted_Temp')]
    
    # Get the years from the column names
    years = [col.split('_')[-1] for col in temp_cols_a if col in temp_cols_b]
    
    if not years:
        raise ValueError("No matching weighted temperature columns found in the data files")
    
    # Create the plots
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Use a color map for consistent coloring
    colors = plt.cm.tab10.colors  # Up to 10 distinct colors
    
    # First plot: Model Run A and B
    for idx, year in enumerate(years):
        col_a = f'Weighted_Temp_{year}'
        col_b = f'Weighted_Temp_{year}'
        color = colors[idx % len(colors)]
        
        axs[0].plot(df_a['DateTime'], df_a[col_a], 
                   label=f'{scenario_a_name} {year}', color=color, linestyle='-')
        axs[0].plot(df_b['DateTime'], df_b[col_b], 
                   label=f'{scenario_b_name} {year}', color=color, linestyle='--')
    
    axs[0].set_ylabel('Weighted River Release Temperature (°C)')
    axs[0].set_title('Weighted Outflow Temperature by Scenario and Year')
    axs[0].legend(title='Scenario')
    axs[0].grid(True)
    
    # Second plot: Difference (B - A) for each year
    for idx, year in enumerate(years):
        col_a = f'Weighted_Temp_{year}'
        col_b = f'Weighted_Temp_{year}'
        color = colors[idx % len(colors)]
        # Merge on DateTime to ensure alignment
        merged = pd.merge(
            df_a[['DateTime', col_a]],
            df_b[['DateTime', col_b]],
            on='DateTime',
            how='inner',
            suffixes=('_a', '_b')
        )
        diff = merged[f'{col_b}_b'] - merged[f'{col_a}_a']
        axs[1].plot(merged['DateTime'], diff, label=f'{year}', color=color)
    
    axs[1].set_xlabel('DateTime')
    axs[1].set_ylabel('Temperature Difference (°C)')
    axs[1].set_title('Outflow Temperature Difference by Year')
    axs[1].legend(title='Year')
    axs[1].grid(True)
    
    plt.tight_layout()
    
    # Create output directory within input directory
    output_dir = os.path.join(input_dir, 'comparison_plots')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save the plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f'{model_run_a}_vs_{model_run_b}_{timestamp}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    return plt

def Analog_Post_Plot_Average(model_run_a, model_run_b, input_dir):
    """
    Create comparison plots for average weighted temperatures between two model runs using weighted data files
    
    Parameters:
    -----------
    model_run_a : str
        Name of the first model run (e.g., '2018_Analog_Scenario_A')
    model_run_b : str
        Name of the second model run (e.g., '2018_Analog_Scenario_B')
    input_dir : str
        Directory containing the input weighted data files
    """
    # Define the path to the weighted data files
    file_a = os.path.join(input_dir, f'{model_run_a}_weighted.csv')
    file_b = os.path.join(input_dir, f'{model_run_b}_weighted.csv')
    
    # Print debug information
    print(f"\nLooking for files in:")
    print(f"Input directory: {input_dir}")
    print(f"File A: {file_a}")
    print(f"File B: {file_b}")
    
    # Check if files exist
    if not os.path.exists(file_a) or not os.path.exists(file_b):
        raise ValueError(f"Could not find weighted data files for {model_run_a} and/or {model_run_b}")
    
    # Load the data
    df_a = pd.read_csv(file_a, parse_dates=['DateTime'])
    df_b = pd.read_csv(file_b, parse_dates=['DateTime'])
    
    # Find weighted temperature columns
    temp_cols_a = [col for col in df_a.columns if col.startswith('Weighted_Temp')]
    temp_cols_b = [col for col in df_b.columns if col.startswith('Weighted_Temp')]
    
    # Get the years from the column names
    years = [col.split('_')[-1] for col in temp_cols_a if col in temp_cols_b]
    
    if not years:
        raise ValueError("No matching weighted temperature columns found in the data files")
    
    # Create the plots
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Calculate average temperatures and ranges for each scenario
    avg_temp_a = df_a[temp_cols_a].mean(axis=1)
    avg_temp_b = df_b[temp_cols_b].mean(axis=1)
    max_temp_a = df_a[temp_cols_a].max(axis=1)
    min_temp_a = df_a[temp_cols_a].min(axis=1)
    max_temp_b = df_b[temp_cols_b].max(axis=1)
    min_temp_b = df_b[temp_cols_b].min(axis=1)
    
    # First plot: Average temperatures with shaded ranges
    axs[0].fill_between(df_a['DateTime'], 
                       min_temp_a, max_temp_a,
                       color='blue', alpha=0.2, label='Scenario A Range')
    axs[0].fill_between(df_b['DateTime'], 
                       min_temp_b, max_temp_b,
                       color='red', alpha=0.2, label='Scenario B Range')
    
    axs[0].plot(df_a['DateTime'], avg_temp_a, 
               label='Scenario A Average', color='blue', linestyle='-')
    axs[0].plot(df_b['DateTime'], avg_temp_b, 
               label='Scenario B Average', color='red', linestyle='--')
    
    axs[0].set_ylabel('Weighted River Release Temperature (°C)')
    axs[0].set_title('Average Outflow Temperature by Scenario')
    axs[0].legend(title='Scenario')
    axs[0].grid(True)
    
    # Second plot: Average difference
    # Calculate differences for each year
    differences = []
    for year in years:
        col_a = f'Weighted_Temp_{year}'
        col_b = f'Weighted_Temp_{year}'
        
        # Merge on DateTime to ensure alignment
        merged = pd.merge(
            df_a[['DateTime', col_a]],
            df_b[['DateTime', col_b]],
            on='DateTime',
            how='inner',
            suffixes=('_a', '_b')
        )
        diff = merged[f'{col_b}_b'] - merged[f'{col_a}_a']
        differences.append(diff)
    
    # Convert list of differences to DataFrame
    diff_df = pd.concat(differences, axis=1)
    
    # Calculate average difference and range
    avg_diff = diff_df.mean(axis=1)
    max_diff = diff_df.max(axis=1)
    min_diff = diff_df.min(axis=1)
    
    # Plot the difference range and average
    axs[1].fill_between(merged['DateTime'], 
                       min_diff, max_diff,
                       color='purple', alpha=0.2, label='Difference Range')
    axs[1].plot(merged['DateTime'], avg_diff, 
               color='purple', label='Average Difference')

    axs[1].set_xlabel('Date')
    axs[1].set_ylabel('Temperature Difference (°C)')
    axs[1].set_title('Average Outflow Temperature Difference')
    axs[1].legend()
    axs[1].grid(True)
    
    plt.tight_layout()
    
    # Create output directory within input directory
    output_dir = os.path.join(input_dir, 'comparison_plots')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save the plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f'{model_run_a}_vs_{model_run_b}_average_{timestamp}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Average plot saved to: {output_file}")
    
    return plt

def Analog_Post_Plot_Interactive(model_run_a, model_run_b, input_dir, scenario_a_name="Scenario A", scenario_b_name="Scenario B"):
    """
    Create interactive comparison plots for weighted temperatures between two model runs using weighted data files
    
    Parameters:
    -----------
    model_run_a : str
        Name of the first model run (e.g., '2018_Analog_Scenario_A')
    model_run_b : str
        Name of the second model run (e.g., '2018_Analog_Scenario_B')
    input_dir : str
        Directory containing the input weighted data files
    scenario_a_name : str, optional
        Display name for the first scenario (default: "Scenario A")
    scenario_b_name : str, optional
        Display name for the second scenario (default: "Scenario B")
    """
    # Define the path to the weighted data files
    file_a = os.path.join(input_dir, f'{model_run_a}_weighted.csv')
    file_b = os.path.join(input_dir, f'{model_run_b}_weighted.csv')
    
    # Print debug information
    print(f"\nLooking for files in:")
    print(f"Input directory: {input_dir}")
    print(f"File A: {file_a}")
    print(f"File B: {file_b}")
    
    # Check if files exist
    if not os.path.exists(file_a) or not os.path.exists(file_b):
        raise ValueError(f"Could not find weighted data files for {model_run_a} and/or {model_run_b}")
    
    # Load the data
    df_a = pd.read_csv(file_a, parse_dates=['DateTime'])
    df_b = pd.read_csv(file_b, parse_dates=['DateTime'])
    
    # Find weighted temperature columns
    temp_cols_a = [col for col in df_a.columns if col.startswith('Weighted_Temp')]
    temp_cols_b = [col for col in df_b.columns if col.startswith('Weighted_Temp')]
    
    # Get the years from the column names
    years = [col.split('_')[-1] for col in temp_cols_a if col in temp_cols_b]
    
    if not years:
        raise ValueError("No matching weighted temperature columns found in the data files")
    
    # Create subplot figure with increased vertical spacing
    fig = make_subplots(rows=2, cols=1, 
                       subplot_titles=('Weighted Outflow Temperature by Scenario and Year',
                                     'Outflow Temperature Difference by Year'),
                       vertical_spacing=0.15)  # Increased spacing
    
    # Define a color palette for the years
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Add traces for each year in the first subplot
    for idx, year in enumerate(years):
        col_a = f'Weighted_Temp_{year}'
        col_b = f'Weighted_Temp_{year}'
        color = colors[idx % len(colors)]
        
        # Add Scenario A trace
        fig.add_trace(
            go.Scatter(x=df_a['DateTime'], 
                      y=df_a[col_a],
                      name=f'{scenario_a_name} {year}',
                      line=dict(width=2, color=color),
                      showlegend=True,
                      legendgroup=f'year_{year}',
                      legendgrouptitle_text=f'Year {year}'),
            row=1, col=1
        )
        
        # Add Scenario B trace
        fig.add_trace(
            go.Scatter(x=df_b['DateTime'], 
                      y=df_b[col_b],
                      name=f'{scenario_b_name} {year}',
                      line=dict(width=2, color=color, dash='dash'),
                      showlegend=True,
                      legendgroup=f'year_{year}'),
            row=1, col=1
        )
        
        # Calculate and add difference trace with same color
        merged = pd.merge(
            df_a[['DateTime', col_a]],
            df_b[['DateTime', col_b]],
            on='DateTime',
            how='inner',
            suffixes=('_a', '_b')
        )
        diff = merged[f'{col_b}_b'] - merged[f'{col_a}_a']
        
        fig.add_trace(
            go.Scatter(x=merged['DateTime'],
                      y=diff,
                      name=f'Difference {year}',
                      line=dict(width=2, color=color),
                      showlegend=True,
                      legendgroup=f'year_{year}'),
            row=2, col=1
        )
    
    # Update layout with improved legend and single range slider
    fig.update_layout(
        height=800,
        width=1200,
        title_text=f'{year} Met Analog Year Comparison for {scenario_a_name} and {scenario_b_name}',
        title_x=0.5,
        hovermode='x unified',
        template='plotly_white',
        legend=dict(
            groupclick="toggleitem",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.05
        ),
        xaxis=dict(
            rangeslider=dict(
                visible=True,
                thickness=0.05
            ),
            type="date"
        ),
        margin=dict(b=100)
    )
    
    # Update y-axes labels
    fig.update_yaxes(title_text='Weighted River Release Temperature (°C)', row=1, col=1)
    fig.update_yaxes(title_text='Temperature Difference (°C)', row=2, col=1)
    
    # Update x-axes labels
    fig.update_xaxes(title_text='Date', row=2, col=1)
    
    # Link x-axes between subplots
    fig.update_xaxes(matches='x')
    
    # Create output directory within input directory
    output_dir = os.path.join(input_dir, 'comparison_plots')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save the interactive plot as HTML
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f'{model_run_a}_vs_{model_run_b}_interactive_{timestamp}.html')
    fig.write_html(output_file)
    print(f"Interactive plot saved to: {output_file}")
    
    return fig

def Analog_Post_3Plot_Avg(model_run_a, model_run_b, model_run_c, input_dir):
    """
    Create comparison plots for average weighted temperatures between three model runs using weighted data files
    
    Parameters:
    -----------
    model_run_a : str
        Name of the first model run (e.g., '2018_Analog_Scenario_A')
    model_run_b : str
        Name of the second model run (e.g., '2018_Analog_Scenario_B')
    model_run_c : str
        Name of the third model run (e.g., '2018_Analog_Scenario_C')
    input_dir : str
        Directory containing the input weighted data files
    """
    # Define the path to the weighted data files
    file_a = os.path.join(input_dir, f'{model_run_a}_weighted.csv')
    file_b = os.path.join(input_dir, f'{model_run_b}_weighted.csv')
    file_c = os.path.join(input_dir, f'{model_run_c}_weighted.csv')
    
    # Print debug information
    print(f"\nLooking for files in:")
    print(f"Input directory: {input_dir}")
    print(f"File A: {file_a}")
    print(f"File B: {file_b}")
    print(f"File C: {file_c}")
    
    # Check if files exist
    if not all(os.path.exists(f) for f in [file_a, file_b, file_c]):
        raise ValueError(f"Could not find weighted data files for one or more scenarios")
    
    # Load the data
    df_a = pd.read_csv(file_a, parse_dates=['DateTime'])
    df_b = pd.read_csv(file_b, parse_dates=['DateTime'])
    df_c = pd.read_csv(file_c, parse_dates=['DateTime'])
    
    # Find weighted temperature columns
    temp_cols_a = [col for col in df_a.columns if col.startswith('Weighted_Temp')]
    temp_cols_b = [col for col in df_b.columns if col.startswith('Weighted_Temp')]
    temp_cols_c = [col for col in df_c.columns if col.startswith('Weighted_Temp')]
    
    # Get the years from the column names
    years = [col.split('_')[-1] for col in temp_cols_a if col in temp_cols_b and col in temp_cols_c]
    
    if not years:
        raise ValueError("No matching weighted temperature columns found in the data files")
    
    # Create the plots
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Calculate average temperatures and ranges for each scenario
    avg_temp_a = df_a[temp_cols_a].mean(axis=1)
    avg_temp_b = df_b[temp_cols_b].mean(axis=1)
    avg_temp_c = df_c[temp_cols_c].mean(axis=1)
    max_temp_a = df_a[temp_cols_a].max(axis=1)
    min_temp_a = df_a[temp_cols_a].min(axis=1)
    max_temp_b = df_b[temp_cols_b].max(axis=1)
    min_temp_b = df_b[temp_cols_b].min(axis=1)
    max_temp_c = df_c[temp_cols_c].max(axis=1)
    min_temp_c = df_c[temp_cols_c].min(axis=1)
    
    # First plot: Average temperatures with shaded ranges
    axs[0].fill_between(df_a['DateTime'], 
                       min_temp_a, max_temp_a,
                       color='blue', alpha=0.2, label='Scenario A Range')
    axs[0].fill_between(df_b['DateTime'], 
                       min_temp_b, max_temp_b,
                       color='red', alpha=0.2, label='Scenario B Range')
    axs[0].fill_between(df_c['DateTime'], 
                       min_temp_c, max_temp_c,
                       color='green', alpha=0.2, label='Scenario C Range')
    
    axs[0].plot(df_a['DateTime'], avg_temp_a, 
               label='Scenario A Average', color='blue', linestyle='-')
    axs[0].plot(df_b['DateTime'], avg_temp_b, 
               label='Scenario B Average', color='red', linestyle='--')
    axs[0].plot(df_c['DateTime'], avg_temp_c, 
               label='Scenario C Average', color='green', linestyle=':')
    
    axs[0].set_ylabel('Weighted River Release Temperature (°C)')
    axs[0].set_title('Average Outflow Temperature by Scenario')
    axs[0].legend(title='Scenario')
    axs[0].grid(True)
    
    # Set y-axis ticks to increment by 1°C
    y_min = int(axs[0].get_ylim()[0])  # Get current y-axis minimum and round down
    y_max = 20  # Set maximum to 20°C
    axs[0].set_yticks(np.arange(y_min, y_max + 1, 1))  # Create ticks at 1°C intervals
    axs[0].set_ylim(y_min, y_max)  # Set the y-axis limits
    
    # Second plot: Average differences
    # Calculate differences for each year
    differences_ab = []
    differences_ac = []
    for year in years:
        col_a = f'Weighted_Temp_{year}'
        col_b = f'Weighted_Temp_{year}'
        col_c = f'Weighted_Temp_{year}'
        
        # Merge on DateTime to ensure alignment
        merged = pd.merge(
            pd.merge(
                df_a[['DateTime', col_a]],
                df_b[['DateTime', col_b]],
                on='DateTime',
                how='inner',
                suffixes=('_a', '_b')
            ),
            df_c[['DateTime', col_c]],
            on='DateTime',
            how='inner'
        )
        diff_ab = merged[f'{col_b}_b'] - merged[f'{col_a}_a']
        diff_ac = merged[col_c] - merged[f'{col_a}_a']
        differences_ab.append(diff_ab)
        differences_ac.append(diff_ac)
    
    # Convert list of differences to DataFrame
    diff_df_ab = pd.concat(differences_ab, axis=1)
    diff_df_ac = pd.concat(differences_ac, axis=1)
    
    # Calculate average difference and range
    avg_diff_ab = diff_df_ab.mean(axis=1)
    max_diff_ab = diff_df_ab.max(axis=1)
    min_diff_ab = diff_df_ab.min(axis=1)
    avg_diff_ac = diff_df_ac.mean(axis=1)
    max_diff_ac = diff_df_ac.max(axis=1)
    min_diff_ac = diff_df_ac.min(axis=1)
    
    # Plot the difference ranges and averages
    axs[1].fill_between(merged['DateTime'], 
                       min_diff_ab, max_diff_ab,
                       color='red', alpha=0.2, label='B-A Difference Range')
    axs[1].plot(merged['DateTime'], avg_diff_ab, 
               color='red', label='B-A Average Difference')
    
    axs[1].fill_between(merged['DateTime'], 
                       min_diff_ac, max_diff_ac,
                       color='green', alpha=0.2, label='C-A Difference Range')
    axs[1].plot(merged['DateTime'], avg_diff_ac, 
               color='green', label='C-A Average Difference')

    axs[1].set_xlabel('Date')
    axs[1].set_ylabel('Temperature Difference (°C)')
    axs[1].set_title('Average Outflow Temperature Difference by Scenario')
    axs[1].legend()
    axs[1].grid(True)
    
    plt.tight_layout()
    
    # Create output directory within input directory
    output_dir = os.path.join(input_dir, 'comparison_plots')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save the plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f'{model_run_a}_vs_{model_run_b}_vs_{model_run_c}_average_{timestamp}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Average plot saved to: {output_file}")
    
    return plt

def Analog_Post_Plot_TCD_Average(model_run_a, model_run_b, input_dir):
    """
    Create comparison plots for average weighted temperatures between two model runs using weighted data files,
    where scenario B uses TCD weighted temperatures
    
    Parameters:
    -----------
    model_run_a : str
        Name of the first model run (e.g., '2018_Analog_Scenario_A')
    model_run_b : str
        Name of the second model run (e.g., '2018_Analog_Scenario_B')
    input_dir : str
        Directory containing the input weighted data files
    """
    # Define the path to the weighted data files
    file_a = os.path.join(input_dir, f'{model_run_a}_weighted.csv')
    file_b = os.path.join(input_dir, f'{model_run_b}_weighted.csv')
    
    # Print debug information
    print(f"\nLooking for files in:")
    print(f"Input directory: {input_dir}")
    print(f"File A: {file_a}")
    print(f"File B: {file_b}")
    
    # Check if files exist
    if not os.path.exists(file_a) or not os.path.exists(file_b):
        raise ValueError(f"Could not find weighted data files for {model_run_a} and/or {model_run_b}")
    
    # Load the data
    df_a = pd.read_csv(file_a, parse_dates=['DateTime'])
    df_b = pd.read_csv(file_b, parse_dates=['DateTime'])
    
    # Find weighted temperature columns
    temp_cols_a = [col for col in df_a.columns if col.startswith('Weighted_Temp_')]
    temp_cols_b = [col for col in df_b.columns if col.startswith('Weighted_TCD_Temp_')]
    
    # Get the years from the column names
    years_a = [col.split('_')[-1] for col in temp_cols_a]
    years_b = [col.split('_')[-1] for col in temp_cols_b]
    
    # Find common years
    years = list(set(years_a).intersection(set(years_b)))
    
    if not years:
        raise ValueError("No matching years found between the two files")
    
    print(f"\nFound matching years: {years}")
    
    # Create the plots
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Calculate average temperatures and ranges for each scenario
    avg_temp_a = df_a[temp_cols_a].mean(axis=1)
    avg_temp_b = df_b[temp_cols_b].mean(axis=1)
    max_temp_a = df_a[temp_cols_a].max(axis=1)
    min_temp_a = df_a[temp_cols_a].min(axis=1)
    max_temp_b = df_b[temp_cols_b].max(axis=1)
    min_temp_b = df_b[temp_cols_b].min(axis=1)
    
    # First plot: Average temperatures with shaded ranges
    axs[0].fill_between(df_a['DateTime'], 
                       min_temp_a, max_temp_a,
                       color='blue', alpha=0.2, label='Regular Weighted Range')
    axs[0].fill_between(df_b['DateTime'], 
                       min_temp_b, max_temp_b,
                       color='red', alpha=0.2, label='TCD Weighted Range')
    
    axs[0].plot(df_a['DateTime'], avg_temp_a, 
               label='Regular Weighted Average', color='blue', linestyle='-')
    axs[0].plot(df_b['DateTime'], avg_temp_b, 
               label='TCD Weighted Average', color='red', linestyle='--')

    # Add three horizontal lines with hard-coded dates and values
    axs[0].hlines(y=14.44, xmin=datetime(2022, 8, 1), xmax=datetime(2022, 12, 31), 
                 color='black', linestyle='--', label='Egg Incubation (14.44°C)')
    axs[0].hlines(y=15.5, xmin=datetime(2022, 8, 1), xmax=datetime(2022, 11, 1), 
                  color='black', linestyle=':', label='Adult Spawning (15.5°C)')
    axs[0].hlines(y=17.0, xmin=datetime(2022, 3, 1), xmax=datetime(2022, 10, 1), 
                 color='black', linestyle='-.', label='Adult Holding (17.0°C)')

    # Add vertical lines at the ends to create brackets for each horizontal line
    bracket_height = 0.35  # Height of the bracket in y-axis units
    
    # Brackets for Egg Incubation line
    axs[0].vlines(x=datetime(2022, 8, 1), ymin=14.44-bracket_height/2, ymax=14.44+bracket_height/2,
                 color='black', linestyle='-', linewidth=1.5)
    axs[0].vlines(x=datetime(2022, 12, 31), ymin=14.44-bracket_height/2, ymax=14.44+bracket_height/2,
                 color='black', linestyle='-', linewidth=1.5)
    
    # Brackets for Adult Spawning line  
    axs[0].vlines(x=datetime(2022, 8, 1), ymin=15.5-bracket_height/2, ymax=15.5+bracket_height/2,
                 color='black', linestyle='-', linewidth=1.5)
    axs[0].vlines(x=datetime(2022, 11, 1), ymin=15.5-bracket_height/2, ymax=15.5+bracket_height/2,
                 color='black', linestyle='-', linewidth=1.5)
    
    # Brackets for Adult Holding line
    axs[0].vlines(x=datetime(2022, 3, 1), ymin=17.0-bracket_height/2, ymax=17.0+bracket_height/2,
                 color='black', linestyle='-', linewidth=1.5) 
    axs[0].vlines(x=datetime(2022, 10, 1), ymin=17.0-bracket_height/2, ymax=17.0+bracket_height/2,
                 color='black', linestyle='-', linewidth=1.5)

    # Add text annotations above each line
    # For Egg Incubation
    # axs[0].text(datetime(2022, 10, 15), 14.44 + 0.00, 'Egg Incubation\n(14.44°C)', 
    #             ha='center', va='bottom', fontsize=12,
    #             path_effects=[patheffects.withStroke(linewidth=3, foreground='white')])
    
    # For Adult Spawning
    # axs[0].text(datetime(2022, 9, 15), 15.5 + 0.10, 'Adult Spawning\n(15.5°C)', 
    #             ha='center', va='bottom', fontsize=12,
    #             path_effects=[patheffects.withStroke(linewidth=3, foreground='white')])
    
    # For Adult Holding
    # axs[0].text(datetime(2022, 6, 15), 17.0 + 0.10, 'Adult Holding\n(17.0°C)', 
    #             ha='center', va='bottom', fontsize=12,
    #             path_effects=[patheffects.withStroke(linewidth=3, foreground='white')])

    axs[0].set_ylabel('Weighted River Release Temperature (°C)')
    axs[0].set_title('Average Outflow Temperature Comparison (Regular vs TCD)')
    
    # Set y-axis ticks to increment by 1°C
    y_min = int(axs[0].get_ylim()[0])  # Get current y-axis minimum and round down
    y_max = 20  # Set maximum to 20°C
    axs[0].set_yticks(np.arange(y_min, y_max + 1, 1))  # Create ticks at 1°C intervals
    axs[0].set_ylim(y_min, y_max)  # Set the y-axis limits
    
    axs[0].legend(title='Calculation Method')
    axs[0].grid(True)
    
    # Second plot: Average difference
    # Calculate differences for each year
    differences = []
    for year in years:
        col_a = f'Weighted_Temp_{year}'
        col_b = f'Weighted_TCD_Temp_{year}'
        
        # Create DataFrames with just the columns we need
        df_a_subset = df_a[['DateTime', col_a]].copy()
        df_b_subset = df_b[['DateTime', col_b]].copy()
        
        # Rename columns to avoid suffix issues
        df_a_subset = df_a_subset.rename(columns={col_a: 'temp_a'})
        df_b_subset = df_b_subset.rename(columns={col_b: 'temp_b'})
        
        # Merge on DateTime
        merged = pd.merge(df_a_subset, df_b_subset, on='DateTime', how='inner')
        
        # Calculate difference
        diff = merged['temp_b'] - merged['temp_a']
        differences.append(diff)
    
    # Convert list of differences to DataFrame
    diff_df = pd.concat(differences, axis=1)
    
    # Calculate average difference and range
    avg_diff = diff_df.mean(axis=1)
    max_diff = diff_df.max(axis=1)
    min_diff = diff_df.min(axis=1)
    
    # Plot the difference range and average
    axs[1].fill_between(merged['DateTime'], 
                       min_diff, max_diff,
                       color='purple', alpha=0.2, label='Difference Range')
    axs[1].plot(merged['DateTime'], avg_diff, 
               color='purple', label='Average Difference')

    axs[1].set_xlabel('Date')
    axs[1].set_ylabel('Temperature Difference (°C)')
    axs[1].set_title('Average Outflow Temperature Difference (TCD - Regular)')
    axs[1].legend()
    axs[1].grid(True)
    
    plt.tight_layout()
    
    # Create output directory within input directory
    output_dir = os.path.join(input_dir, 'comparison_plots')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save the plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f'{model_run_a}_vs_{model_run_b}_tcd_average_{timestamp}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"TCD Average plot saved to: {output_file}")
    
    # Save the comparison data to CSV
    comparison_df = pd.DataFrame({
        'DateTime': df_a['DateTime'],
        'Regular_Weighted_Temp': avg_temp_a,
        'TCD_Weighted_Temp': avg_temp_b,
        'Temperature_Difference': avg_diff
    })
    
    comparison_csv = os.path.join(output_dir, f'{model_run_a}_vs_{model_run_b}_tcd_average_{timestamp}.csv')
    comparison_df.to_csv(comparison_csv, index=False)
    print(f"TCD Average comparison data saved to: {comparison_csv}")
    
    return plt