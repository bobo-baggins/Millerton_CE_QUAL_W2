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
   plt.rcParams['figure.figsize'] = (8, 4)
   plt.rcParams['font.size'] = 12
   plt.rcParams['lines.linewidth'] = 1.5
   plt.rcParams['lines.linestyle'] = '-'
   plt.rcParams['axes.labelsize'] = plt.rcParams['font.size']
   plt.rcParams['axes.titlesize'] = plt.rcParams['font.size']
   plt.rcParams['legend.fontsize'] = 0.9*plt.rcParams['font.size']
   plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
   plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']
init_plotting()
##################################  USER DEFINED VARIABLES #################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################

start_date = '2024-01-01 00:00:00'
end_date = '2024-12-31 23:00:00'

############################################################################################
############################################################################################
############################################################################################
############################################################################################

###calculate solar elevation angle
df = pd.read_csv('QAQC/solar_rad_avg_hourly_QAQC.csv', index_col = 0, parse_dates=True)
df['day_of_year'] = df.index.dayofyear + df.index.hour/24
df['hour_of_day'] = df.index.hour
num_days = df.day_of_year.values
declination_angle = -23.44 * np.cos(np.radians((360/365)*(num_days - 1 + 10)))
df['declination_angle'] = declination_angle
hour = df.hour_of_day.values
hour_angle = 15*(hour-12)
df['solar_hour_angle'] = hour_angle
lat = 	32.32 #latitude
sin_alpha = np.sin(np.radians(lat))*np.sin(np.radians(declination_angle))+np.cos(np.radians(lat))*np.cos(np.radians(declination_angle))*np.cos(np.radians(hour_angle))
solar_elevation_angle = np.degrees(np.arcsin(sin_alpha))
df['solar_elevation_angle'] = solar_elevation_angle

##### calculate clear sky insolation
theta_p = df.solar_elevation_angle.values
theta = np.zeros(len(theta_p))
for t in range(1,len(theta_p)):
    theta[t] = (theta_p[t-1]+theta_p[t])/2
clear_sky_insolation = 990*np.sin(np.radians(theta))-30    
df['clear_sky_insolation_Wm2'] = clear_sky_insolation


#### calculate cloud cover (method 1)
# R = df.Radiation_Wm2.values
# R0 = clear_sky_insolation
# print(np.mean(R/R0))
# cloud_cover  = (4/3*(1-(R/R0)))**(1/3.4)
# df['cloud_cover'] = cloud_cover

#### calculate cloud cover (method 2)
R = df.Radiation_Wm2.values
R0 = clear_sky_insolation
cloud_cover  = ((1/0.65)*(1-(R/R0)))**(1/2)
df['cloud_cover_raw'] = cloud_cover
cloud_cover_new = np.nan_to_num(cloud_cover, nan = 0.0)
for i in range(len(cloud_cover_new)):
    if cloud_cover_new[i] > 1:
        cloud_cover_new[i] = 1
df['cloud_cover'] = cloud_cover_new
df = df[(df.index.hour >= 13) & (df.index.hour <= 15)]
ix = pd.date_range(start_date,end_date, freq = 'h')
df = df.reindex(ix)
df.loc[start_date,'cloud_cover'] = 1
df.loc[end_date,'cloud_cover'] = 1

df['cloud_cover'] = df.cloud_cover.interpolate()
df.cloud_cover.plot()
plt.ylabel('Fraction cover')

df['cloud_cover_10'] = df.cloud_cover*10
df.index.name = 'DATE TIME'
df.to_csv('QAQC/cloud_cover_hourly_QAQC.csv')