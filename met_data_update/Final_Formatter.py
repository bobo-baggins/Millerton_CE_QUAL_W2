import pandas as pd
from pathlib import Path
import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def combine_met_files(input_dir: Path = Path('QAQC'), 
                     output_file: Path = Path('2024_CEQUAL_met_inputs.csv'),
                     start_julian_day: float = None):
    """
    Combine meteorological data files into a single CE-QUAL formatted file.
    
    Parameters:
        input_dir (Path): Directory containing the input CSV files
        output_file (Path): Path for the output combined CSV file
        start_julian_day (float): Starting Julian day for the dataset
    """
    try:
        # Dictionary to store our dataframes with their purpose
        dfs = {}
        
        # Read all CSV files from the input directory
        for csv_file in input_dir.glob('*.csv'):
            df = pd.read_csv(csv_file)
            logging.info(f"Loaded {csv_file.name}")
            logging.info(f"Date range in {csv_file.name}: {df['DATE TIME'].min()} to {df['DATE TIME'].max()}")
            
            # The column might be named differently - check what we actually have
            print("Available columns:", df.columns.tolist())
            
            # It's likely that the date column has a different name
            # Try to identify the date column (it's probably the index)
            if 'DATE TIME' not in df.columns:
                # If it was read as the index, reset it to a column
                df = df.reset_index()
                # Rename it to what we expect
                df = df.rename(columns={'index': 'DATE TIME'})
            
            df['DATE TIME'] = pd.to_datetime(df['DATE TIME'])
            df = df.set_index('DATE TIME')
            dfs[csv_file.stem] = df

        # Create a complete datetime index
        min_date = min(df.index.min() for df in dfs.values())
        max_date = max(df.index.max() for df in dfs.values())
        complete_index = pd.date_range(start=min_date, end=max_date, freq='H')

        # Initialize the final dataframe with the complete datetime index
        final_df = pd.DataFrame(index=complete_index)

        # Calculate the correct Julian day for the start date
        if start_julian_day is None:
            # Convert the first date to Julian day (counting from January 1, 2024)
            base_date = pd.Timestamp('2024-01-01')
            first_date = min(df.index.min() for df in dfs.values())
            days_since_base = (first_date - base_date).total_seconds() / (24 * 3600)
            start_julian_day = 37621.0 + days_since_base  # 37621 is Jan 1, 2024 Julian day
            
            # Debug prints
            print(f"Base date: {base_date}")
            print(f"First date from data: {first_date}")
            print(f"Days since base: {days_since_base}")
            print(f"Calculated start Julian day: {start_julian_day}")

        # Calculate Julian days
        hours_elapsed = (final_df.index - final_df.index[0]).total_seconds() / 3600
        final_df['JDAY'] = start_julian_day + (hours_elapsed / 24)

        # More debug prints
        print(f"First few Julian days in output:")
        print(final_df['JDAY'].head())

        # Map the input files to CE-QUAL variables (no unit conversion)
        mappings = {
            'air_temp_avg_hourly_QAQC': ('Temp_C', 'TAIR'),
            'dewpoint_temp_hourly_QAQC': ('DP_temp_C', 'TDEW'),
            'wind_speed_hourly_QAQC': ('speed_mph', 'WIND'),
            'atmospheric_pressure_hourly_QAQC': ('Pressure_In', 'PHI'),
            'cloud_cover_hourly_QAQC': ('cloud_cover_10', 'CLOUD'),
            'solar_rad_avg_hourly_QAQC': ('Radiation_Wm2', 'SRO'),
            'wind_direction_hourly_QAQC': ('Dir_degrees', 'WIND')
        }

        # Process each variable
        for file_base, (input_col, output_col) in mappings.items():
            if file_base in dfs:
                if file_base == 'wind_direction_hourly_QAQC':
                    # Convert from meteorological wind direction to mathematical angle in radians
                    wind_degrees = dfs[file_base][input_col]
                    wind_radians = np.radians((270 - wind_degrees) % 360)
                    final_df[output_col] = wind_radians
                    # Add validation
                    logging.info(f"Wind direction range (radians): {wind_radians.min():.2f} to {wind_radians.max():.2f}")
                else:
                    final_df[output_col] = dfs[file_base][input_col]
                logging.info(f"Processed {file_base} into {output_col}")

        # Ensure all required columns are present and in correct order
        required_columns = ['JDAY', 'TAIR', 'TDEW', 'WIND', 'PHI', 'CLOUD', 'SRO']
        for col in required_columns:
            if col not in final_df.columns:
                logging.warning(f"Missing required column: {col}")
                final_df[col] = 0  # or another appropriate default value

        # Reorder columns to match CE-QUAL format
        final_df = final_df[required_columns]

        # Save the combined file
        final_df.to_csv(output_file)
        logging.info(f"Successfully created combined file: {output_file}")
        
        return final_df

    except Exception as e:
        logging.error(f"Error combining meteorological files: {str(e)}")
        raise

if __name__ == "__main__":
    # Let the function automatically calculate the Julian day
    combine_met_files(
        input_dir=Path('QAQC'),
        output_file=Path('2024_CEQUAL_met_inputs.csv')
    )
