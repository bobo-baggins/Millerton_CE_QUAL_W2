# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 12:45:23 2022

@author: jcohen
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import os
from matplotlib import gridspec
import seaborn as sns
def init_plotting():
   sns.set_style("whitegrid", {"axes.facecolor": "0.8",'axes.edgecolor': '1','grid.color': '0.6'}) 
   sns.set_context({'grid.linewidth':'1'})
   plt.rcParams['figure.figsize'] = (7,5)
   plt.rcParams['font.size'] = 11
   plt.rcParams['lines.linewidth'] = 2
   plt.rcParams['lines.linestyle'] = '-'
   plt.rcParams['font.weight'] = 'bold'
   plt.rcParams['axes.labelsize'] = 1.15*plt.rcParams['font.size']
   plt.rcParams['axes.titlesize'] = 1.15*plt.rcParams['font.size']
   plt.rcParams['legend.fontsize'] = 11#0.95*plt.rcParams['font.size']
   plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
   plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']
init_plotting()

##################################  USER DEFINED VARIABLES #################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################

start_date = '2022-07-7 01:00:00' #start date of simulation results
end_date = '2022-12-31 23:00:00' #end date of simulation results

x_min = '2022-7-1' #minimum date shown on plot
next_day = '2022-7-2' #used for adding temperatures to plot. Should be day after x_min
x_max = '2023-1-1' #maximum date shown on plot

ymin = 45 #in degrees. Maxmimum temperature shown on plot
ymax = 75 #in degrees. Minimum temperature shown on plot
analogs =  [1988,1989,1990,1994,2002,2007,2008,2013,2020,2021]

time_step = '1D' #time frequency at which results are processed. Must be in pandas format. e.g. '1D', '1H'
statistic = 'maximum' #Statstic used when resampling data. can be 'average', 'maximum', or 'minumum'

rolling_window = 7 #used for rolling average. 
plot_title = 'max daily temperatures for ensemble average \n 7 day rolling average'

plot_file_name = 'release_temps_ensemble_averages.png'
### choose statistics to plot
mean = True
median = True
maximum = True
minimum = True
percent_90 = True
percent_75 = True
percent_25 = True
percent_10 = True

############################################################################################
############################################################################################
############################################################################################
############################################################################################

fig, ax = plt.subplots(1,1,sharey = True, sharex = True)
analogs = [1988,1989,1990,1994,2002,2007,2008,2013,2020,2021]#[1988,1989,1990,1994,2002,2007,2008,2013,2015,2020,2021]


ix = pd.date_range(start = '2022-07-7 01:00:00', end = '2022-12-31 23:00:00', freq = 'H')
dfa = pd.DataFrame(index = ix)
for analog_year in analogs:
    df = pd.read_csv('output_csvs/outflow_temps/%s_analog-release_outputs.csv'%( analog_year), index_col = 0, parse_dates = True)

    df['SJR_temp_F'] = df.SJR_temp_C.values*1.8+32
    dfa[analog_year] = df['SJR_temp_F']

if statistic == 'maximum':
    dfa = dfa.resample(time_step).max().rolling(rolling_window).mean() 

elif statistic == 'average':
    dfa = dfa.resample(time_step).mean().rolling(rolling_window).mean() 
    
elif statistic == 'minimum':
    dfa = dfa.resample(time_step).min().rolling(rolling_window).mean()

dfa['average'] = dfa.mean(axis = 1)
dfa['maximum'] = dfa.max(axis = 1)
dfa['minimum'] = dfa.min(axis = 1)
dfa['median'] = dfa.median(axis = 1)
dfa['90th_percentile'] = dfa.quantile(q = 0.9, axis = 1)
dfa['75th_percentile'] = dfa.quantile(q = 0.75, axis = 1)
dfa['25th_percentile'] = dfa.quantile(q = 0.25, axis = 1)
dfa['10th_percentile'] = dfa.quantile(q = 0.1, axis = 1)

if median == True:
    ax.plot(dfa['median'], label = 'median', c = 'b')
if maximum == True:
    ax.plot(dfa['maximum'], label = 'maximum', c = 'purple', ls = '--')
if minimum == True:
    ax.plot(dfa['minimum'], label = 'minimum', c = 'purple', ls = '--')
if percent_90 == True:
    ax.plot(dfa['90th_percentile'], label = '90th percentile', c = 'r', ls = ':')
if percent_75 == True:
    ax.plot(dfa['75th_percentile'], label = '75th percentile', c = 'g', ls = '-.')
if percent_25 == True:
    ax.plot(dfa['25th_percentile'], label = '25th percentile', c = 'g', ls = '-.')
if percent_10 == True:
    ax.plot(dfa['10th_percentile'], label = '10th percentile', c = 'r', ls = ':')
if mean == True: 
    ax.plot(dfa['average'], label = 'mean', c = 'k')

   # ax.set_title('%s%% exceedance'%exc)
ax.tick_params(axis = 'x', labelrotation = 45)
xmin, xmax = pd.to_datetime(x_min), pd.to_datetime(x_max)
ax.plot([xmin,xmax],[58,58], c = 'k')
ax.plot([xmin,xmax],[60,60], c = 'k')
ax.plot([xmin,xmax],[62.6,62.6], c = 'k')
ax.plot([xmin,xmax],[68,68], c = 'k')

ax.text(pd.to_datetime(next_day),58.4,r"58 $\degree$F", size = 'medium')
ax.text(pd.to_datetime(next_day),60.4,r"60 $\degree$F", size = 'medium')
ax.text(pd.to_datetime(next_day),63,r"62.6 $\degree$F", size = 'medium')
ax.text(pd.to_datetime(next_day),68.4,r"68 $\degree$F")

ax.set_ylim([ymin,ymax])
ax.set_xlim([xmin,xmax])
ax.yaxis.set_tick_params(labelleft = True)
ax.xaxis.set_tick_params(labelbottom = True)
ax.set_title(plot_title)
ax.legend(ncol = 2)
ax.set_ylabel(r'Release temp $\degree$F')
ax.set_ylabel(r'Release temp $\degree$F')

plt.tight_layout(w_pad = 0, h_pad = 0.3)
plt.savefig('figures/ensemble/%s'%plot_file_name)