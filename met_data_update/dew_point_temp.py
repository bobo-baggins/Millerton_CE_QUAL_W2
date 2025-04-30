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
df = pd.read_csv('QAQC/air_temp_avg_hourly_QAQC.csv', index_col = 0, parse_dates = True)
dfrh = pd.read_csv('QAQC/relative_hum_hourly_QAQC.csv', index_col = 0, parse_dates = True)
df['rel_hum'] = dfrh.pcnt_hum
RH = df.rel_hum.values*0.01
T = df.Temp_C.values
Td = (112+0.9*T)*(RH**(1/8))-112 + 0.1*T
df['DP_temp_C'] = Td
df.DP_temp_C.plot()
plt.ylabel('Dewpoint temp C')
df.to_csv('QAQC/dewpoint_temp_hourly_QAQC.csv')