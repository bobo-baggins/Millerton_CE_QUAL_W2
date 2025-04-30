# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 12:02:08 2022

@author: jcohen
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import math
from matplotlib import gridspec
import seaborn as sns
def init_plotting():
   sns.set_style("whitegrid", {"axes.facecolor": "1",'axes.edgecolor': '1','grid.color': '0.6'}) 
   sns.set_context({'grid.linewidth':'1'})
   plt.rcParams['figure.figsize'] = (12, 5)
   plt.rcParams['font.size'] = 13
   plt.rcParams['lines.linewidth'] = 1.5
   plt.rcParams['lines.linestyle'] = '-'
   plt.rcParams['font.weight'] = 'bold'
   plt.rcParams['axes.labelsize'] = 1.1*plt.rcParams['font.size']
   plt.rcParams['axes.titlesize'] = 1.1*plt.rcParams['font.size']
   plt.rcParams['legend.fontsize'] = 0.8*plt.rcParams['font.size']
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

x_min = start_date #minimum date shown on plot
next_day = '2022-7-8' #used for adding temperatures to plot
x_max = '2023-1-1' #maximum date shown on plot. Must be day after end date

ymin = 45 #in degrees. Maxmimum temperature shown on plot
ymax = 75 #in degrees. Minimum temperature shown on plot
analogs =  [1988,1989,1990,1994,2002,2007,2008,2013,2020,2021]


############################################################################################
############################################################################################
############################################################################################
############################################################################################

for an in analogs:
    print(an)
    analog_year = an
    
    fig = plt.figure()
    spec = gridspec.GridSpec(ncols=2, nrows=1,
                             width_ratios=[20, 1])
    
    ax0 = fig.add_subplot(spec[0])
    ax1 = fig.add_subplot(spec[1])
    

    with open('CEQUAL_outputs/%s/spr_1.opt'%(analog_year),"r") as f:
        lines = f.readlines()
    lines.pop(0)
    
    for i,l in enumerate(lines):
        lines[i] = np.array(l.split(',')[:-1], dtype = np.float32)
    ix = np.arange(90,184)
    dates =  pd.date_range(start = start_date, end =end_date, freq = 'D')
    df = pd.DataFrame(index = ix,columns = dates)
    df.index.name = 'Elevation'
    df_dec = df
    profiles = []
    for d,l in enumerate(np.arange(0,len(lines)-3,3)): ### add temperatures to array indexed by elevation
        elevs = np.round(lines[l+1][1:])
        date = dates[d]
        df.loc[elevs,date] = np.round(lines[l+2][1:]*1.8+32)
        df_dec.loc[elevs,date] = lines[l+2][1:]*1.8+32
    df.to_csv('output_csvs/profiles/%s_analog_profiles.csv'%an)
    df_dec = df_dec.fillna(value = 0)
    ### create colors
    norm = np.arange(0,51)/(np.sum(np.arange(0,51)**2)**0.5)
    norm = norm*(1/norm[50])
    colors = cm.jet(norm)
    dates_next = pd.date_range(start = next_day,end =x_max,freq = 'D')
    crit_temp = 58
    crit_elev_arr = np.zeros(len(df.columns))
    
    crit_temp_2 = 60
    crit_elev_arr_2 = np.zeros(len(df.columns))
    
    crit_temp_3 = 68
    crit_elev_arr_3 = np.zeros(len(df.columns))
    ax0.set_xlim(pd.Timestamp(x_min),pd.Timestamp(x_max))
    
    ### plotting
    for i in np.arange(0,len(dates)): 
        d = dates[i]
        dnext = dates_next[i]
        if np.max(df_dec[d]) >= crit_temp:
            difference_array = np.absolute(df_dec[d].values-crit_temp)
            index = difference_array.argmin()
            crit_elev = df_dec.index.values[index]
            crit_elev_arr[i] = crit_elev
        else: 
            crit_elev_arr[i] = np.nan
            
        if np.max(df_dec[d]) >= crit_temp_2:
            difference_array = np.absolute(df_dec[d].values-crit_temp_2)
            index = difference_array.argmin()
            crit_elev_2 = df_dec.index.values[index]
            crit_elev_arr_2[i] = crit_elev_2
        else: 
            crit_elev_arr_2[i] = np.nan
        
        if np.max(df_dec[d]) >= crit_temp_3:
            difference_array = np.absolute(df_dec[d].values-crit_temp_3)
            index = difference_array.argmin()
            crit_elev_3 = df_dec.index.values[index]
            crit_elev_arr_3[i] = crit_elev_3
        else: 
            crit_elev_arr_3[i] = np.nan
    
            
        for j,temp in enumerate(df[d]):
            if math.isnan(temp) == False:
                temp_index = math.trunc(temp-40)
                evalue = df.index[j]#elevation, in meters 
                y = np.array([evalue-0.5,evalue+0.5])*3.2808399
                x1 = d
                x2 = dnext
                ax0.fill_betweenx(y = y, x1 = x1, x2 = x2, color = colors[temp_index])
                
    ax0.plot(ax0.get_xlim(),[380,380],c = 'k', ls = '--',label = 'River outlet')
    ax0.plot(ax0.get_xlim(),[446,446],c = 'k',ls = ':',label = 'Madera Canal')
    ax0.plot(ax0.get_xlim(),[464,464],c = 'k',ls = '-.', label = 'Friant Kern Canal')
    ax0.plot(df.columns, crit_elev_arr_3*3.2808399, color = 'xkcd:silver', lw = 2, ls = '-', label = '%0.0f F elevation'%crit_temp_3)
    ax0.plot(df.columns, crit_elev_arr_2*3.2808399, color = 'xkcd:slate grey', lw = 2, ls = '-', label = '%0.0f F elevation'%crit_temp_2)
    ax0.plot(df.columns, crit_elev_arr*3.2808399, color = 'k', lw = 2, ls = '-', label = '%0.0f F elevation'%crit_temp)
    
    ################## colorbar###################
    cmap = plt.cm.jet  # define the colormap
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    
    # create the new map
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, cmap.N)
    
    # define the bins and normalize
    bounds = np.arange(40,91)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    ticks = np.arange(40,91,5)
    
    # create a second axes for the colorbar
    cb = mpl.colorbar.ColorbarBase(ax = ax1,cmap=cmap, norm=norm,
        spacing='proportional', ticks=ticks, boundaries=bounds, format='%2i')
    cb.set_label('Temperature, F')
    ax0.set_ylabel('Elevation (ft)')
    ax0.legend(loc = 'upper right', ncol = 2, bbox_to_anchor = (1.1,1.1))
  #  plt.tight_layout()
    plt.savefig('figures/profiles/%s_analog_profiles.png'%(analog_year))
    plt.close()