import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from CDEC_Formatter import CDECFormatter

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MetDataProcessor:
    def __init__(self):
        """Initialize MetDataProcessor using date range from input files."""
        # Get date range from CDEC data files
        cdec_dir = Path('cdec')
        hourly_dir = Path('hourly')
        
        # Process CDEC data first to ensure files exist
        formatter = CDECFormatter()
        formatter.process_all()
        
        # Get all CSV files in hourly directory
        csv_files = list(hourly_dir.glob('*.csv'))
        if not csv_files:
            raise FileNotFoundError("No input files found in hourly directory")
            
        # Read first file to get initial date range
        df = pd.read_csv(csv_files[0], index_col=0, parse_dates=True)
        self.start_date = df.index.min()
        self.end_date = df.index.max()
        
        # Check other files to ensure consistent date range
        for file in csv_files[1:]:
            df = pd.read_csv(file, index_col=0, parse_dates=True)
            self.start_date = max(self.start_date, df.index.min())
            self.end_date = min(self.end_date, df.index.max())
            
        logging.info(f"Date range from input files: {self.start_date} to {self.end_date}")
        self.setup_plotting()
        
    def setup_plotting(self):
        """Initialize plotting parameters"""
        sns.set_style("whitegrid", {"axes.facecolor": "1",'axes.edgecolor': '0.6','grid.color': '0.6'})
        sns.set_context({'grid.linewidth':'1'})
        plt.rcParams['figure.figsize'] = (10, 5)
        plt.rcParams['font.size'] = 12
        plt.rcParams['lines.linewidth'] = 1.5
        plt.rcParams['lines.linestyle'] = '-'
        plt.rcParams['axes.labelsize'] = plt.rcParams['font.size']
        plt.rcParams['axes.titlesize'] = plt.rcParams['font.size']
        plt.rcParams['legend.fontsize'] = 0.9*plt.rcParams['font.size']
        plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
        plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']

    def process_cdec_data(self):
        """Process CDEC data first"""
        from CDEC_Formatter import CDECFormatter
        formatter = CDECFormatter()
        formatter.process_all()
        logging.info("Completed CDEC data processing")

    def process_basic_met_data(self):
        """Process and QA/QC basic meteorological data"""
        # Create output directory if it doesn't exist
        Path('QAQC').mkdir(exist_ok=True)
        
        # Process air temperature
        df_temp = pd.read_csv('hourly/air_temp_avg_hourly.csv', index_col=0, parse_dates=True)
        df_temp['Temp_C'] = (df_temp.Temp_F-32)*(5/9)
        df_temp.to_csv('QAQC/air_temp_avg_hourly_QAQC.csv', index_label='DATE TIME')
        
        # Process wind speed
        df_wind = pd.read_csv('hourly/wind_speed_hourly.csv', index_col=0, parse_dates=True)
        df_wind['speed_ms'] = df_wind.speed_mph * 0.44704  # Convert mph to m/s
        df_wind.to_csv('QAQC/wind_speed_hourly_QAQC.csv', index_label='DATE TIME')
        # Process other meteorological variables
        variables = {
            'atmospheric_pressure_hourly.csv': 'Pressure_In',
            'solar_rad_avg_hourly.csv': 'Radiation_Wm2',
            'wind_direction_hourly.csv': 'Dir_degrees',
            'wind_speed_hourly.csv': 'speed_ms',  # Changed to indicate meters/second
            'relative_hum_hourly.csv': 'pcnt_hum'
        }
        
        for filename, column in variables.items():
            df = pd.read_csv(f'hourly/{filename}', index_col=0, parse_dates=True)
            df = df.interpolate()
            
            # Special handling for wind direction
            if filename == 'wind_direction_hourly.csv':
                # Convert from meteorological wind direction to mathematical angle in radians
                # Meteorological: 0° = North, 90° = East
                # Mathematical: 0 radians = East, π/2 radians = North
                df['WIND'] = np.radians((270 - df.Dir_degrees) % 360)
                # Validate the conversion
                assert all((df['WIND'] >= 0) & (df['WIND'] <= 2*np.pi)), "Wind directions must be between 0 and 2π radians"
            
            df.to_csv(f'QAQC/{filename.replace(".csv", "_QAQC.csv")}', index_label='DATE TIME')
            
        logging.info("Completed basic meteorological data processing")

    def calculate_dewpoint(self):
        """Calculate dewpoint temperature"""
        df = pd.read_csv('QAQC/air_temp_avg_hourly_QAQC.csv', index_col=0, parse_dates=True)
        dfrh = pd.read_csv('QAQC/relative_hum_hourly_QAQC.csv', index_col=0, parse_dates=True)
        
        df['rel_hum'] = dfrh.pcnt_hum
        RH = df.rel_hum.values * 0.01
        T = df.Temp_C.values
        Td = (112 + 0.9*T)*(RH**(1/8)) - 112 + 0.1*T
        df['DP_temp_C'] = Td
        df.to_csv('QAQC/dewpoint_temp_hourly_QAQC.csv', index_label='DATE TIME')
        logging.info("Completed dewpoint temperature calculation")

    def calculate_cloud_cover(self):
        """Calculate cloud cover"""
        df = pd.read_csv('QAQC/solar_rad_avg_hourly_QAQC.csv', index_col=0, parse_dates=True)
        
        # Calculate solar parameters
        df['day_of_year'] = df.index.dayofyear + df.index.hour/24
        df['hour_of_day'] = df.index.hour
        
        # Solar calculations
        num_days = df.day_of_year.values
        declination_angle = -23.44 * np.cos(np.radians((360/365)*(num_days - 1 + 10)))
        df['declination_angle'] = declination_angle
        
        hour = df.hour_of_day.values
        hour_angle = 15*(hour-12)
        df['solar_hour_angle'] = hour_angle
        
        # Calculate solar elevation angle
        lat = 32.32  # latitude
        sin_alpha = (np.sin(np.radians(lat)) * np.sin(np.radians(declination_angle)) +
                    np.cos(np.radians(lat)) * np.cos(np.radians(declination_angle)) *
                    np.cos(np.radians(hour_angle)))
        solar_elevation_angle = np.degrees(np.arcsin(sin_alpha))
        df['solar_elevation_angle'] = solar_elevation_angle
        
        # Calculate cloud cover
        theta_p = solar_elevation_angle
        theta = np.zeros(len(theta_p))
        for t in range(1, len(theta_p)):
            theta[t] = (theta_p[t-1] + theta_p[t])/2
        
        clear_sky_insolation = 990*np.sin(np.radians(theta))-30
        df['clear_sky_insolation_Wm2'] = clear_sky_insolation
        
        R = df.Radiation_Wm2.values
        R0 = clear_sky_insolation
        cloud_cover = ((1/0.65)*(1-(R/R0)))**(1/2)
        df['cloud_cover_raw'] = cloud_cover
        
        # Process cloud cover data
        cloud_cover_new = np.nan_to_num(cloud_cover, nan=0.0)
        cloud_cover_new = np.clip(cloud_cover_new, 0, 1)
        
        df['cloud_cover'] = cloud_cover_new
        df = df[(df.index.hour >= 13) & (df.index.hour <= 15)]
        
        # Reindex and interpolate
        ix = pd.date_range(self.start_date, self.end_date, freq='h')
        df = df.reindex(ix)
        df.loc[self.start_date, 'cloud_cover'] = 1
        df.loc[self.end_date, 'cloud_cover'] = 1
        df['cloud_cover'] = df.cloud_cover.interpolate()
        df['cloud_cover_10'] = df.cloud_cover * 10
        
        df.to_csv('QAQC/cloud_cover_hourly_QAQC.csv', index_label='DATE TIME')
        logging.info("Completed cloud cover calculation")

    def create_final_output(self):
        """Create final CE-QUAL formatted output"""
        from Final_Formatter import combine_met_files
        combine_met_files()
        logging.info("Created final CE-QUAL formatted output")

    def process_all(self):
        """Run all processing steps in order"""
        self.process_cdec_data()
        self.process_basic_met_data()
        self.calculate_dewpoint()
        self.calculate_cloud_cover()
        self.create_final_output()
        logging.info("Completed all meteorological data processing")

    def validate_angles(self, angles, name):
        """Validate angle ranges"""
        if np.any(np.abs(angles) > 2*np.pi):
            logging.warning(f"Warning: {name} contains values larger than 2π radians, might be using degrees instead of radians")
        return angles

if __name__ == "__main__":
    processor = MetDataProcessor()
    processor.process_all()