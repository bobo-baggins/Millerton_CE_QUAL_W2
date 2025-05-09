import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def julian_to_date(julian_day, year):
    """Convert Julian day to datetime using 1921 as reference year"""
    start_date = datetime(1921, 1, 1)
    return start_date + timedelta(days=float(julian_day) - 1)

def plot_temperature_comparison(file_a, file_b, file_c, year, jdays):
    """
    Plot temperature comparisons for different scenarios using provided Julian days
    
    Parameters:
    -----------
    file_a, file_b, file_c : str
        Paths to the CSV files containing temperature data
    year : int
        Year for the simulation
    jdays : array-like
        Array of Julian days from the simulation
    """
    # Read CSV files, skipping first 3 rows
    df_a = pd.read_csv(file_a, skiprows=3, header=None)
    df_b = pd.read_csv(file_b, skiprows=3, header=None)
    df_c = pd.read_csv(file_c, skiprows=3, header=None)
    
    # Use the provided Julian days
    time_a = jdays
    temperature_a = df_a[5]  # 6th column (temperature)
    
    time_b = jdays
    temperature_b = df_b[5]
    
    time_c = jdays
    temperature_c = df_c[5]
    
    # Convert Julian days to dates
    dates_a = [julian_to_date(jday, year) for jday in time_a]
    dates_b = [julian_to_date(jday, year) for jday in time_b]
    dates_c = [julian_to_date(jday, year) for jday in time_c]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[2, 1])
    
    # Plot temperatures on first subplot
    ax1.plot(dates_a, temperature_a, label='April_Scenario_A', color='blue')
    ax1.plot(dates_b, temperature_b, label='April_Scenario_B', color='red')
    ax1.plot(dates_c, temperature_c, label='April_Scenario_C', color='green')
    
    # Format first subplot
    ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %d'))
    ax1.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title(f'Temperature Comparison for {year}')
    ax1.grid(True)
    ax1.legend()
    
    # Plot differences on second subplot using matching colors
    ax2.plot(dates_a, temperature_b - temperature_a, label='B - A', color='red')
    ax2.plot(dates_a, temperature_c - temperature_a, label='C - A', color='green')
    
    # Format second subplot
    ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %d'))
    ax2.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Temperature Difference (°C)')
    ax2.grid(True)
    ax2.legend()
    
    # Rotate and align the tick labels
    plt.gcf().autofmt_xdate()
    
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
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
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
    
    # Interpolate WSE data to match hourly resolution and convert from meters to feet
    wse_interp = np.interp(jdays, df_wse['JDAY'], df_wse['ELWS(m)']) * 3.28084
    
    # Debug prints for array shapes
    print(f"\nArray shapes:")
    print(f"Temperature array shape: {temperature.shape}")
    print(f"Flow1 array shape: {flow1.shape}")
    print(f"Flow2 array shape: {flow2.shape}")
    print(f"WSE array shape: {wse_interp.shape}")
    print(f"Jdays array shape: {jdays.shape}")
    
    # Convert Julian days to dates
    dates = [julian_to_date(jday, year) for jday in jdays]
    
    # Create figure with three y-axes
    fig, ax1 = plt.subplots(figsize=(15, 8))
    
    # Create secondary axes
    ax2 = ax1.twinx()    # Flow axis
    ax3 = ax1.twinx()    # WSE axis
    
    # Position the axes
    ax2.spines['right'].set_position(('axes', 1.0))      # Flow on right
    ax3.spines['right'].set_position(('axes', 1.1))      # WSE on far right
    
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
    
    # Plot WSE on tertiary y-axis
    wse_line = ax3.plot(dates, wse_interp, label='WSE at Dam', color='purple', alpha=0.7)
    ax3.set_ylabel('WSE (ft)', color='purple', labelpad=15)
    ax3.tick_params(axis='y', labelcolor='purple')
    
    # Format x-axis
    ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %d'))
    ax1.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
    ax1.set_xlabel('Date')
    
    # Add title and grid
    plt.title(f'Water Temperature, Flow, and WSE QAQC Plot - {year} {run_name}')
    ax1.grid(True)
    
    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper left')
    
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

def Analog_Post_Plot(model_run_a, model_run_b):
    """
    Create comparison plots for weighted temperatures between two model runs using weighted data files
    
    Parameters:
    -----------
    model_run_a : str
        Name of the first model run (e.g., '2018_Analog_Scenario_A')
    model_run_b : str
        Name of the second model run (e.g., '2018_Analog_Scenario_B')
    """
    # Define the path to the weighted data files
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                           'CEQUAL_outputs', '2018_Analogs_Scenario_ABC')
    file_a = os.path.join(base_dir, f'{model_run_a}_weighted.csv')
    file_b = os.path.join(base_dir, f'{model_run_b}_weighted.csv')
    
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
                   label=f'Scenario A {year}', color=color, linestyle='-')
        axs[0].plot(df_b['DateTime'], df_b[col_b], 
                   label=f'Scenario B {year}', color=color, linestyle='--')
    
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
    
    # Save the plot
    output_dir = os.path.join(base_dir, 'comparison_plots')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f'{model_run_a}_vs_{model_run_b}_{timestamp}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    return plt

def Analog_Post_Plot_Average(model_run_a, model_run_b):
    """
    Create comparison plots for average weighted temperatures between two model runs using weighted data files
    
    Parameters:
    -----------
    model_run_a : str
        Name of the first model run (e.g., '2018_Analog_Scenario_A')
    model_run_b : str
        Name of the second model run (e.g., '2018_Analog_Scenario_B')
    """
    # Define the path to the weighted data files
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                           'CEQUAL_outputs', '2018_Analogs_Scenario_ABC')
    file_a = os.path.join(base_dir, f'{model_run_a}_weighted.csv')
    file_b = os.path.join(base_dir, f'{model_run_b}_weighted.csv')
    
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
    
    axs[1].set_xlabel('DateTime')
    axs[1].set_ylabel('Temperature Difference (°C)')
    axs[1].set_title('Average Outflow Temperature Difference')
    axs[1].legend()
    axs[1].grid(True)
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = os.path.join(base_dir, 'comparison_plots')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f'{model_run_a}_vs_{model_run_b}_average_{timestamp}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Average plot saved to: {output_file}")
    
    return plt

def Analog_Post_Plot_Interactive(model_run_a, model_run_b):
    """
    Create interactive comparison plots for weighted temperatures between two model runs using weighted data files
    
    Parameters:
    -----------
    model_run_a : str
        Name of the first model run (e.g., '2018_Analog_Scenario_A')
    model_run_b : str
        Name of the second model run (e.g., '2018_Analog_Scenario_B')
    """
    # Define the path to the weighted data files
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                           'CEQUAL_outputs', '2018_Analogs_Scenario_ABC')
    file_a = os.path.join(base_dir, f'{model_run_a}_weighted.csv')
    file_b = os.path.join(base_dir, f'{model_run_b}_weighted.csv')
    
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
                      name=f'Scenario A {year}',
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
                      name=f'Scenario B {year}',
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
        title_text='2018 Met Analog Year Comparison for Scenario A and B',
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
        # Add single range slider that controls both plots with reduced height
        xaxis=dict(
            rangeslider=dict(
                visible=True,
                thickness=0.05
            ),
            type="date"
        ),
        # Add bottom margin to prevent overlap
        margin=dict(b=100)
    )
    
    # Update y-axes labels
    fig.update_yaxes(title_text='Weighted River Release Temperature (°C)', row=1, col=1)
    fig.update_yaxes(title_text='Temperature Difference (°C)', row=2, col=1)
    
    # Update x-axes labels
    fig.update_xaxes(title_text='Date', row=2, col=1)
    
    # Link x-axes between subplots
    fig.update_xaxes(matches='x')
    
    # Save the interactive plot as HTML
    output_dir = os.path.join(base_dir, 'comparison_plots')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f'{model_run_a}_vs_{model_run_b}_interactive_{timestamp}.html')
    fig.write_html(output_file)
    print(f"Interactive plot saved to: {output_file}")
    
    return fig

fig = Analog_Post_Plot_Interactive('2018_Analog_Scenario_A', '2018_Analog_Scenario_B')