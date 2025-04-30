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
   plt.rcParams['lines.linewidth'] = 1.5
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
########################################################################################################################################################################################


start_date = '2022-07-7 01:00:00' #start date of simulation results
end_date = '2022-10-31 23:00:00' #end date of simulation results

x_min = '2022-7-1' #minimum date shown on plot
next_day = '2022-7-2' #used for adding temperatures text to plot. should be day after x_min
x_max = '2022-10-31' #maximum date shown on plot

ymin = 45 #in degrees. Maxmimum temperature shown on plot
ymax = 75 #in degrees. Minimum temperature shown on plot
analogs =  [1988,1989,1990,1994,2002,2007,2008,2013,2020,2021] #analog year simulations which we are plotting

time_step = '1H' #time frequency at which results are processed. Must be in pandas format. e.g. '1D', '1H'
statistic = 'average' #Statstic used when resampling data. can be 'average', 'maximum', or 'minumum'. Does nothing if hourly data. 

rolling_window = 1 #used for rolling average. Same units as time step.

plot_title = 'max daily temperatures for each analog year \n 7 day rolling average' #\n - to skip line in title

plot_file_name = 'release_temps_ensemble.png' # goes in figures/ensemble file
############################################################################################
############################################################################################
############################################################################################
############################################################################################

fig, ax = plt.subplots(1,1,sharey = True, sharex = True)        
ix = pd.date_range(start = start_date, end = end_date, freq = 'H')
dfa = pd.DataFrame(index = ix)
for analog_year in analogs:
    df = pd.read_csv('output_csvs/outflow_temps/%s_analog-release_outputs.csv'%(analog_year), index_col = 0, parse_dates = True)

    df['SJR_temp_F'] = df.SJR_temp_C.values*1.8+32
    dfa[analog_year] = df['SJR_temp_F']

if statistic == 'maximum':
    dfa = dfa.resample(time_step).max().rolling(rolling_window).mean() #here we can change rolling 

elif statistic == 'average':
    dfa = dfa.resample(time_step).mean().rolling(rolling_window).mean() #here we can change rolling 

elif statistic == 'minimum':
    dfa = dfa.resample(time_step).min().rolling(rolling_window).mean() #here we can change rolling 

ax.plot(dfa)

dfa['mean_temp'] = dfa.mean(axis = 1)
dfa['max_temp'] = dfa.max(axis = 1)
dfa['min_temp'] = dfa.min(axis = 1)

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
ax.set_title(plot_title)       
ax.yaxis.set_tick_params(labelleft = True)
ax.xaxis.set_tick_params(labelbottom = True)

ax.legend(analogs,ncol = 3, columnspacing = 0.6, handlelength = 1, loc = 'upper right', framealpha = 0.95,handletextpad = 0.38)
#ax[0][0].legend(labelspacing = 0.2, borderpad = 0.2, borderaxespad = 0.3, handlelength = 0.6, columnspacing = 0.75, handletextpad = 0.4, ncol = 4, loc = 'lower right').set_title('GRF flow', prop = {'size':11})
ax.set_ylabel(r'Release temp $\degree$F')
ax.set_ylabel(r'Release temp $\degree$F')

plt.tight_layout(w_pad = 0, h_pad = 0.3)
plt.savefig('figures/ensemble/%s'%plot_file_name)
plt.show()