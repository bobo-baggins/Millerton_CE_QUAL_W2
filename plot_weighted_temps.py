import pandas as pd
import matplotlib.pyplot as plt

# Path to the CSV files
csv_path_a = 'CEQUAL_outputs/2018_Analogs_Scenario_ABC/2018_Analog_Scenario_A_combined_data.csv'
csv_path_b = 'CEQUAL_outputs/2018_Analogs_Scenario_ABC/2018_Analog_Scenario_B_combined_data.csv'

# Load the data, parsing the DateTime column as dates
df_a = pd.read_csv(csv_path_a, parse_dates=['DateTime'])
df_b = pd.read_csv(csv_path_b, parse_dates=['DateTime'])

# Find all columns that start with 'Weighted_Temp'
temp_cols_a = [col for col in df_a.columns if col.startswith('Weighted_Temp')]
temp_cols_b = [col for col in df_b.columns if col.startswith('Weighted_Temp')]

# Ensure both scenarios have the same years/columns, and remove 2023
years = [col.split('_')[-1] for col in temp_cols_a if col in temp_cols_b and col.split('_')[-1] != '2023']

fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Use a color map for consistent coloring
colors = plt.cm.tab10.colors  # Up to 10 distinct colors

# First plot: Scenario A and B
mask_a = (df_a['DateTime'].dt.month >= 8)
mask_b = (df_b['DateTime'].dt.month >= 8)
for idx, year in enumerate(years):
    col_a = f'Weighted_Temp_{year}'
    col_b = f'Weighted_Temp_{year}'
    color = colors[idx % len(colors)]
    axs[0].plot(df_a.loc[mask_a, 'DateTime'], df_a.loc[mask_a, col_a], label=f'Scenario A {year}', color=color, linestyle='-')
    axs[0].plot(df_b.loc[mask_b, 'DateTime'], df_b.loc[mask_b, col_b], label=f'Scenario B {year}', color=color, linestyle='--')

axs[0].set_ylabel('Weighted Temperature')
axs[0].set_title('Weighted Temperatures Over Time: Scenario A (solid) vs B (dashed)')
axs[0].legend()
axs[0].grid(True)

# Second plot: Difference (A - B) for each year
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
    # Filter to August-December
    mask = (merged['DateTime'].dt.month >= 8)
    diff = merged.loc[mask, f'{col_a}_a'] - merged.loc[mask, f'{col_b}_b']
    axs[1].plot(merged.loc[mask, 'DateTime'], diff, label=f'Diff {year}', color=color)

axs[1].set_xlabel('DateTime')
axs[1].set_ylabel('A - B Difference')
axs[1].set_title('Difference in Weighted Temperature (A - B) by Year')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show() 