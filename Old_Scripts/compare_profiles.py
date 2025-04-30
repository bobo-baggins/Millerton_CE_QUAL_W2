# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 10:20:09 2022

@author: jcohen
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import math 

def init_plotting():
   sns.set_style("whitegrid", {"axes.facecolor": "0.8",'axes.edgecolor': '1','grid.color': '0.5'}) 
   sns.set_context({'grid.linewidth':'1'})
   plt.rcParams['figure.figsize'] = (6,6)
   plt.rcParams['font.size'] = 12
   plt.rcParams['lines.linewidth'] = 2
   plt.rcParams['lines.linestyle'] = '-'
   plt.rcParams['font.weight'] = 'bold'
   plt.rcParams['axes.labelsize'] = 1.1*plt.rcParams['font.size']
   plt.rcParams['axes.titlesize'] = 1.1*plt.rcParams['font.size']
   plt.rcParams['legend.fontsize'] = 0.86*plt.rcParams['font.size']
   plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
   plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']
init_plotting()

##################################  USER DEFINED VARIABLES #################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################

start_date = '2022-07-7' #start date of simulation results. Must be just day
end_date = '2022-12-31' #end date of simulation results. Must be just day

xmin = 47 #in degrees. Maxmimum temperature shown on plot
xmax = 82 #in degrees. Minimum temperature shown on plot
#put in ymin and ymax
analogs =  [1988,1989,1990,1994,2002,2007,2008,2013,2020,2021]
profile_date = '8/30/2022' # must fall between start_date and end_date

plot_title = 'Comparision of analog profiles on 8/30/2022'

plot_file_name = 'analog_profiles_8_30_2022.png'

############################################################################################
############################################################################################
############################################################################################
############################################################################################
fig, ax = plt.subplots(1,1,sharey=True, sharex = True)

ix = np.arange(90,184)
dates =  pd.date_range(start = start_date, end = end_date,freq = 'D')
for an in analogs:
    with open('CEQUAL_outputs/%s/spr_1.opt'%(an),"r") as f:
        lines = f.readlines()
    lines.pop(0)
    
    for i,l in enumerate(lines):
        lines[i] = np.array(l.split(',')[:-1], dtype = np.float32)
    dfW2 = pd.DataFrame(index = ix,columns = dates)
    dfW2.index.name = 'Elevation_m'
    for d,l in enumerate(np.arange(0,len(lines)-3,3)): ### add temperatures to array indexed by elevation
        elevs = np.round(lines[l+1][1:])
        date = dates[d]
        dfW2.loc[elevs,date] = lines[l+2][1:]*1.8+32
    dfW2[dfW2.columns] = dfW2[dfW2.columns].values
    dfW2.index = dfW2.index.values*3.2808399
    ax.plot(dfW2[profile_date].values,dfW2.index.values, label = an)
ax.set_xlim(xmin,xmax)
ax.plot([xmin,xmax],[380,380],c = 'k', ls = '--',label = 'River outlet')
ax.plot([xmin,xmax],[446,446],c = 'k',ls = ':',label = 'Madera Canal')
ax.plot([xmin,xmax],[464,464],c = 'k',ls = '-.', label = 'Friant Kern Canal')

ax.set_ylabel('Elevation (ft)')
ax.set_xlabel('Temperature F')
plt.legend()
ax.set_title(plot_title)
plt.tight_layout()
plt.savefig('figures/ensemble/%s'%plot_file_name)