# Flow visualization script for comparing different flow scenarios
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ============ CONFIGURATION SECTION ============
# Flow data settings
FLOW_PATHS = {
    '24_SJA_Flows': 'SJA_24_Flows.csv',
    '23_SJA_Flows': 'SJA_23_Flows.csv',
    '23_K2P_Flows': 'K2P_23_Flows.csv',
    '24_K2P_Flows': 'K2P_24_Flows.csv',
    #'SCENARIO_B': '2024_USBR_Flow.csv'
}

# Column to plot
PLOT_COLUMN = 'Flow(cfs)'  # Changed from SJR_OUT to M_IN

# Plot settings
PLOT_LABELS = {
    '24_SJA_Flows': 'Flows at SJA (2024)',
    '23_SJA_Flows': 'Flows at SJA (2023)',
    '23_K2P_Flows': 'Flows at K2P (2023)',
    '24_K2P_Flows': 'Flows at K2P (2024)',
}

PLOT_COLORS = {
    '24_SJA_Flows': 'green',
    '23_SJA_Flows': 'green',
    '23_K2P_Flows': 'red',
    '24_K2P_Flows': 'red',
}

PLOT_STYLES = {
    '24_SJA_Flows': '-',
    '23_SJA_Flows': '--',
    '23_K2P_Flows': '--',
    '24_K2P_Flows': '-',
}

# Date range settings
DATE_SETTINGS = {
    'start_date': '2024-01-01',
    'end_date': '2024-12-31',
    'date_format': '%b'  # Changed from '%Y-%m-%d' to '%b' for month abbreviation
}

# Figure settings
FIGURE_SETTINGS = {
    'figsize': (12, 6),
    'dpi': 300,
    'title': 'Inflow Comparison',  # Updated title to reflect M_IN
    'xlabel': 'Date',
    'ylabel': 'Flow (cfs)',
    'grid': True,
    'x_rotation': 45,
    'font_size': 14
}

# Output settings
OUTPUT_SETTINGS = {
    'filename': 'flow_comparison.png',
    'dpi': 300,
    'transparent': False,
    'pad_inches': 0.05
}

# ============ END CONFIGURATION SECTION ============

def read_flow_data(file_path, scenario):
    """Read and process flow data from CSV file."""
    df = pd.read_csv(file_path)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
    df.dropna(subset=[df.columns[0]], inplace=True)
    
    # Convert Flow(cfs) to numeric
    df['Flow(cfs)'] = pd.to_numeric(df['Flow(cfs)'], errors='coerce')
    
    # Normalize dates to 2024 for comparison
    if '23' in scenario:  # If it's 2023 data
        df.iloc[:, 0] = df.iloc[:, 0] + pd.DateOffset(years=1)
    
    return df

def setup_plot():
    """Set up the plot with basic formatting."""
    plt.figure(figsize=FIGURE_SETTINGS['figsize'])
    plt.xlabel(FIGURE_SETTINGS['xlabel'], fontsize=FIGURE_SETTINGS['font_size'])
    plt.ylabel(FIGURE_SETTINGS['ylabel'], fontsize=FIGURE_SETTINGS['font_size'])
    plt.rcParams.update({'font.size': FIGURE_SETTINGS['font_size']})  # Set global font size
    if FIGURE_SETTINGS['grid']:
        plt.grid(True)

def format_axes(ax):
    """Format the axes with date formatting and limits."""
    ax.xaxis.set_major_formatter(mdates.DateFormatter(DATE_SETTINGS['date_format']))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xlim(pd.to_datetime(DATE_SETTINGS['start_date']), 
            pd.to_datetime(DATE_SETTINGS['end_date']))
    plt.xticks(rotation=FIGURE_SETTINGS['x_rotation'], fontsize=FIGURE_SETTINGS['font_size'])
    plt.yticks(fontsize=FIGURE_SETTINGS['font_size'])
    ax.tick_params(labelsize=FIGURE_SETTINGS['font_size'])

def save_plot():
    """Save the plot with specified settings."""
    plt.tight_layout(pad=0.5)
    plt.savefig(
        OUTPUT_SETTINGS['filename'],
        dpi=OUTPUT_SETTINGS['dpi'],
        bbox_inches='tight',
        format='png',
        transparent=OUTPUT_SETTINGS['transparent'],
        pad_inches=OUTPUT_SETTINGS['pad_inches']
    )

def main():
    # Initialize dictionary to store all flow data
    flow_data = {}

    # Read all flow data
    for scenario, path in FLOW_PATHS.items():
        flow_data[scenario] = read_flow_data(path, scenario)

    # Create plot
    setup_plot()

    # Define the order we want for plotting
    plot_order = ['23_SJA_Flows', '24_SJA_Flows', '23_K2P_Flows', '24_K2P_Flows']

    # Plot each flow series in our specified order
    for scenario in plot_order:
        plt.plot(
            flow_data[scenario].iloc[:, 0].values,  # Date column
            flow_data[scenario][PLOT_COLUMN].values,  # Use configured column
            label=PLOT_LABELS[scenario],
            color=PLOT_COLORS[scenario],
            linestyle=PLOT_STYLES[scenario]
        )

    # Format axes
    format_axes(plt.gca())

    # Add legend with font size
    plt.legend(fontsize=FIGURE_SETTINGS['font_size'])

    # Save and show plot
    save_plot()
    plt.show()

if __name__ == "__main__":
    main()
