import pandas as pd
from pathlib import Path
import logging
import warnings
from typing import Dict, Tuple, List
from datetime import datetime, timedelta

# Suppress openpyxl warning about default styles
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl.styles.stylesheet')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CDECFormatter:
    DATE_TIME_FORMAT = '%m/%d/%Y %H:%M'
    
    # Updated mapping to handle FRT files
    COLUMN_MAPPING: Dict[str, Tuple[str, str]] = {
        'frt_4': ('Temp_F', 'air_temp_avg_hourly.csv'),
        'frt_9': ('speed_mph', 'wind_speed_hourly.csv'),
        'frt_10': ('Dir_degrees', 'wind_direction_hourly.csv'),
        'frt_12': ('pcnt_hum', 'relative_hum_hourly.csv'),
        'frt_17': ('Pressure_In', 'atmospheric_pressure_hourly.csv')
    }

    def __init__(self, input_dir: Path = Path('cdec'), output_dir: Path = Path('hourly')):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        self.dataframes = {}  # Store dataframes in memory

    def process_xlsx_files(self) -> None:
        """Process all XLSX files in the input directory."""
        xlsx_files = list(self.input_dir.glob('*.xlsx'))
        
        for xlsx_file in xlsx_files:
            try:
                base_name = xlsx_file.stem.lower()
                if base_name not in self.COLUMN_MAPPING:
                    logging.warning(f"Skipping unmapped file: {xlsx_file}")
                    continue
                
                self._process_single_file(xlsx_file, base_name)
                logging.info(f"Successfully processed {xlsx_file}")
                
            except Exception as e:
                logging.error(f"Error processing {xlsx_file}: {str(e)}")

    def _process_single_file(self, file_path: Path, base_name: str) -> None:
        """Process a single XLSX file and store in memory."""
        new_col_name, output_filename = self.COLUMN_MAPPING[base_name]
        
        df = pd.read_excel(file_path)
        df = (df[['DATE TIME', 'VALUE']]
              .rename(columns={'VALUE': new_col_name}))
        
        df['DATE TIME'] = pd.to_datetime(df['DATE TIME'])
        df = df.set_index('DATE TIME')
        
        self.dataframes[output_filename] = df

    def process_solar_data(self) -> None:
        """Process the CIMIS solar radiation data."""
        try:
            solar_file = self.input_dir / 'CIMIS_Solar.csv'
            if not solar_file.exists():
                logging.error("Solar radiation file not found")
                return

            solar_df = pd.read_csv(solar_file, dtype={'Hour (PST)': str})
            
            solar_df = (solar_df[['Date', 'Hour (PST)', 'Sol Rad (W/sq.m)']]
                       .rename(columns={'Sol Rad (W/sq.m)': 'Radiation_Wm2'}))
            
            solar_df['Hour (PST)'] = solar_df['Hour (PST)'].str.split('.').str[0]
            solar_df['DATE TIME'] = solar_df['Date'] + ' ' + solar_df['Hour (PST)']
            solar_df = solar_df[['DATE TIME', 'Radiation_Wm2']]
            
            solar_df['DATE TIME'] = pd.to_datetime(
                solar_df['DATE TIME'], 
                format='%m/%d/%Y %H%M', 
                errors='coerce'
            )
            
            solar_df = solar_df.dropna(subset=['DATE TIME'])
            solar_df = solar_df.set_index('DATE TIME')  # Sets datetime index
            
            # Stores in the same dataframes dictionary used by other files
            self.dataframes['solar_rad_avg_hourly.csv'] = solar_df
            
        except Exception as e:
            logging.error(f"Error processing solar radiation data: {str(e)}")

    def standardize_and_save(self) -> None:
        """Standardize all dataframes and save to CSV."""
        try:
            if not self.dataframes:
                logging.warning("No data to standardize")
                return

            # Find overall date range
            min_date = min(df.index.min() for df in self.dataframes.values())
            max_date = max(df.index.max() for df in self.dataframes.values())

            # Create complete datetime index
            complete_index = pd.date_range(start=min_date, end=max_date, freq='H')

            # Add debug prints
            logging.info(f"Date range: {min_date} to {max_date}")
            
            # Standardize and save each dataframe
            for filename, df in self.dataframes.items():
                # Check for duplicates before reindexing
                if df.index.duplicated().any():
                    logging.warning(f"Found duplicate timestamps in {filename}")
                    # Keep first occurrence of duplicates
                    df = df[~df.index.duplicated(keep='first')]
                
                # Reindex to include all hours
                df = df.reindex(complete_index)
                
                # Interpolate missing values
                df = df.interpolate(method='linear', limit=24)
                
                # Format datetime back to string
                df.index = df.index.strftime(self.DATE_TIME_FORMAT)
                
                # Save standardized file
                output_path = self.output_dir / filename
                df.to_csv(output_path, index=True, index_label='DATE TIME')
                logging.info(f"Saved standardized file: {filename}")

        except Exception as e:
            logging.error(f"Error standardizing and saving files: {str(e)}")

    def process_all(self) -> None:
        """Process all files and standardize the results."""
        self.process_xlsx_files()
        self.process_solar_data()
        self.standardize_and_save()

if __name__ == "__main__":
    formatter = CDECFormatter()
    formatter.process_all()
