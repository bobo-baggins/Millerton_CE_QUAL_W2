# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 09:35:24 2022

@author: jcohen
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import matplotlib
def init_plotting():
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
init_plotting()
fig, ((ax0,ax1),(ax2,ax3),(ax4,ax5)) = plt.subplots(3,2)
#### air avg hourly #######
df = pd.read_csv('hourly/air_temp_avg_hourly.csv', index_col = 0, parse_dates=True)
df = df.interpolate()
df['Temp_C']= (df.Temp_F-32)*(5/9)
df.Temp_C.plot(ax = ax0, label = 'temp C')
df.to_csv('QAQC/air_temp_avg_hourly_QAQC.csv')


### pressure hourly#######
df = pd.read_csv('hourly/atmospheric_pressure_hourly.csv', index_col = 0, parse_dates=True)
df.Pressure_In.plot(ax = ax1, label = 'pressure in')
df.to_csv('QAQC/atmospheric_pressure_hourly_QAQC.csv')

### avg solar radiation #######
df = pd.read_csv('hourly/solar_rad_avg_hourly.csv', index_col = 0, parse_dates=True)
df = df.interpolate()
df.Radiation_Wm2.plot(ax = ax2, label = 'Radiation Wm2')
#plt.show()
df.to_csv('QAQC/solar_rad_avg_hourly_QAQC.csv')

### avg wind direction #######
df = pd.read_csv('hourly/wind_direction_hourly.csv', index_col = 0, parse_dates=True)
df = df.interpolate()
df.Dir_degrees.plot(ax = ax3, label = 'wind direction degrees')
df['Dir_radians'] = np.radians(df.Dir_degrees.values)
df.to_csv('QAQC/wind_direction_hourly_QAQC.csv')


### avg wind speed #######
df = pd.read_csv('hourly/wind_speed_hourly.csv', index_col = 0, parse_dates=True)
df = df.interpolate()
df.speed_mph.plot(ax = ax4, label = 'wind speed mph')
df['speed_ms'] = df.speed_mph*(1/2.23693629)
df.to_csv('QAQC/wind_speed_hourly_QAQC.csv')

### relative humidity #######
df = pd.read_csv('hourly/relative_hum_hourly.csv', index_col = 0, parse_dates=True)
df = df.interpolate()
df.pcnt_hum.plot(ax = ax5, label = 'relative humidity %')
df.to_csv('QAQC/relative_hum_hourly_QAQC.csv')

ax0.legend()
ax0.set_ylabel('Temp C')

ax1.legend()
ax1.set_ylabel('Pressure inches')

ax2.legend()
ax2.set_ylabel('Radiation W/m2')

ax3.legend()
ax3.set_ylabel('Direction Degrees')

ax4.legend()
ax4.set_ylabel('Speed mph')

ax5.legend()
ax5.set_ylabel('% humidity')
plt.tight_layout()

plt.savefig('QAQC/QAQC_hourly.png', dpi = 300)

#plt.show()