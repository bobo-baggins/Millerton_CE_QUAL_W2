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
   sns.set_style("whitegrid", {"axes.facecolor": "1",'axes.edgecolor': '1','grid.color': '0.6'}) 
   sns.set_context({'grid.linewidth':'1'})
   plt.rcParams['figure.figsize'] = (12,4)
   plt.rcParams['font.size'] = 12
   plt.rcParams['lines.linewidth'] = 2
   plt.rcParams['lines.linestyle'] = '-'
   plt.rcParams['font.weight'] = 'bold'
   plt.rcParams['axes.labelsize'] = 1.2*plt.rcParams['font.size']
   plt.rcParams['axes.titlesize'] = 1.2*plt.rcParams['font.size']
   plt.rcParams['legend.fontsize'] = 1.1*plt.rcParams['font.size']
   plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
   plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']
init_plotting()

##################################  USER DEFINED VARIABLES #################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################

profile_file = '2025-04-17_profile.csv' ###user input

############################################################################################
############################################################################################
############################################################################################
############################################################################################

fig, (ax0,ax1,ax2) = plt.subplots(1,3,sharey=True)
df = pd.read_csv(profile_file, index_col = 0)
df.index = np.round(df.index.values,decimals = 4)
elev = df.index
temp = df.Temp_C.values
TDS = df.TDS_mgl.values
DO = df.DO_mgl.values

upper_elev_bound = math.ceil(elev.values[0])*1.000
lower_elev_bound = 50*1.000
ix = np.arange(lower_elev_bound,upper_elev_bound+0.0001,0.0001)

tx = np.empty(len(ix))
TDSx = np.empty(len(ix))
DOx = np.empty(len(ix))

tx[:] = np.nan
TDSx[:] = np.nan
DOx[:] = np.nan

for i,e in enumerate(ix):
    er = round(e,4)
    if er in elev:
        tx[i] = df.loc[er,'Temp_C']
        TDSx[i] = df.loc[er,'TDS_mgl']
        DOx[i] = df.loc[er,'DO_mgl']
print('step 1 done')
df2 = pd.DataFrame(index = ix)

df2['Temp_C'] = tx
df2['TDS_mgl'] = TDSx
df2['DO_mgl'] = DOx

df2.loc[upper_elev_bound,'Temp_C'] = temp[0]
df2.loc[upper_elev_bound,'TDS_mgl'] = TDS[0]
df2.loc[upper_elev_bound,'DO_mgl'] = DO[0]

df2.loc[lower_elev_bound,'Temp_C'] = temp[-1]
df2.loc[lower_elev_bound,'TDS_mgl'] = TDS[-1]
df2.loc[lower_elev_bound,'DO_mgl'] = DO[-1]


df2 = df2.interpolate()
elev = df2.index.values
initheight = int(upper_elev_bound-lower_elev_bound)

tseries = np.zeros(initheight)
TDSseries = np.zeros(initheight)
DOseries = np.zeros(initheight)

temp = df2.Temp_C.values
TDS = df2.TDS_mgl.values
DO = df2.DO_mgl.values

for i in np.arange(0,initheight,1):
    
    ind = int(i*10000)
    arr_temp = []
    arr_TDS = []
    arr_DO = []
    
    for j in np.arange(-5000,5001,1):
        arr_temp.append(temp[ind+j])
        arr_TDS.append(TDS[ind+j])
        arr_DO.append(DO[ind+j])

    avg_temp = np.sum(arr_temp)*0.0001
    avg_TDS = np.sum(arr_TDS)*0.0001
    avg_DO = np.sum(arr_DO)*0.0001

    tseries[i] = avg_temp
    TDSseries[i] = avg_TDS
    DOseries[i] = avg_DO

tseries[0] = tseries[1]
TDSseries[0] = TDSseries[1]
DOseries[0] = DOseries[1]

tseries = np.flip(tseries)
TDSseries = np.flip(TDSseries)
DOseries = np.flip(DOseries)

tprofile = np.zeros(216)
TDSprofile = np.zeros(216)
DOprofile = np.zeros(216)

for i in range(0,216):
    if i < initheight:
        tprofile[i] = tseries[i]
        TDSprofile[i] = TDSseries[i]
        DOprofile[i] = DOseries[i]

    elif i >= initheight:
        tprofile[i] = tseries[-1]
        TDSprofile[i] = TDSseries[-1]
        DOprofile[i] = DOseries[-1]

tarr = np.zeros([24,9])
TDSarr = np.zeros([24,9])
DOarr = np.zeros([24,9])

for i in np.arange(0,24):
    tarr[i] = tprofile[i*9:(i+1)*9]
    TDSarr[i] = TDSprofile[i*9:(i+1)*9]
    DOarr[i] = DOprofile[i*9:(i+1)*9]

tarr = np.round(tarr, decimals = 2)
TDSarr = np.round(TDSarr, decimals = 2)
DOarr = np.round(DOarr, decimals = 2)

lines = []
lines.append(f'Profile file: {profile_file} \n')
lines.append('File created from SJRRP Milerton Temperature Profile Viewer real string measurements')
lines.append('\nTemperC       T1      T1      T1      T1      T1      T1      T1      T1      T1   ')

for i in range(0, len(tarr)):
    prof_line = tarr[i]
    #for j in range(0, len(prof_line):
    l = f"\n{prof_line[0] : >16}{prof_line[1] : >8}{prof_line[2] : >8}{prof_line[3] : >8}{prof_line[4] : >8}{prof_line[5] : >8}{prof_line[6] : >8}{prof_line[7] : >8}{prof_line[8] : >8}" #make line for CEQUAL timestep
    lines.append(l)

lines.append('\n            ')            
lines.append('\nTDS mgl       C1      C1      C1      C1      C1      C1      C1      C1      C1')
for i in range(0, len(TDSarr)):
    prof_line = TDSarr[i]
    #for j in range(0, len(prof_line):
    l = f"\n{prof_line[0] : >16}{prof_line[1] : >8}{prof_line[2] : >8}{prof_line[3] : >8}{prof_line[4] : >8}{prof_line[5] : >8}{prof_line[6] : >8}{prof_line[7] : >8}{prof_line[8] : >8}" #make line for CEQUAL timestep
    lines.append(l)

lines.append('\n\nDO mgl        C2      C2      C2      C2      C2      C2      C2      C2      C2')
for i in range(0, len(DOarr)):
    prof_line = DOarr[i]
    #for j in range(0, len(prof_line):
    l = f"\n{prof_line[0] : >16}{prof_line[1] : >8}{prof_line[2] : >8}{prof_line[3] : >8}{prof_line[4] : >8}{prof_line[5] : >8}{prof_line[6] : >8}{prof_line[7] : >8}{prof_line[8] : >8}" #make line for CEQUAL timestep
    lines.append(l)
lines.append('\n')    
#with open('../CEQUAL_model/mvpr1.npt',"w") as update:
#    update.writelines(lines)
with open('mvpr1.npt',"w") as update:
    update.writelines(lines)
    
ax0.plot(tseries,np.arange(upper_elev_bound,lower_elev_bound,-1))
ax1.plot(TDSseries,np.arange(upper_elev_bound,lower_elev_bound,-1)) 
ax2.plot(DOseries,np.arange(upper_elev_bound,lower_elev_bound,-1)) 

ax0.set_xlabel('Temp, C')
ax1.set_xlabel('TDS, mgl')
ax2.set_xlabel('DO, mgl')
ax0.set_ylabel('Elevation, m')
plt.show()


    

    
