# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 15:48:22 2022
Inherited Novemeber 2023
Last update: Feb 2025

@author: jcohen
@editor: jHawkins
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import subprocess
import glob, os, shutil
import time

##################################  USER DEFINED VARIABLES #################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################

#include file to update in excel: 
start_date = '01/13/2022 1:00' #start date of simulation (always start at 1:00)
end_date ='12/15/2022 23:00' #end date of simulation (always at at 23:00)
analog_years =  [2022]# available:  [1988,1989,1990,1994,2002,2007,2008,2013,2020,2021,2022,2023,2024]
wait_time = 333 #seconds between simulations. Program will break if simulations take longer than wait time

#Code will excute after X amount of seconds, error if Model hasn't finished running by the end.
#Minimum wait time is 90 seconds.

make_output_folders =False #makes output folders, only if none exist
#####################################################################################
############################################################################################
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)
############################################################################################
############################################################################################

for analog_year in analog_years:
    print(analog_year)
    
    # Backup w2_con.csv before any operations
    try:
        shutil.copy2('w2_con.csv', 'w2_con.csv.backup')
        print("Backed up w2_con.csv")
    except Exception as e:
        print(f"Error backing up w2_con.csv: {str(e)}")
        raise

    df_flow = pd.read_csv('flow_data/flow_data_base.csv', index_col = 0, parse_dates = True)
    df_temp = pd.read_csv('flow_data/flow_data_temp.csv', index_col = 0, parse_dates = True)
    
    # Convert index to datetime if it's not already
    df_flow.index = pd.to_datetime(df_flow.index)

    # Adjust start_date to include the first hour
    start_date_inclusive = pd.Timestamp(start_date).floor('D')  # Set to beginning of day
    end_date_inclusive = pd.Timestamp(end_date).ceil('D')  # Set to end of day
    
    # Filter with adjusted dates
    df_flow = df_flow[(df_flow.index >= start_date_inclusive) & (df_flow.index <= end_date_inclusive)]
    df_temp = df_temp[(df_temp.index >= start_date_inclusive) & (df_temp.index <= end_date_inclusive)]
    
    # Debug prints after filtering
    print(f"\nAfter filtering:")
    print(f"First date in filtered data: {df_flow.index[0]}")
    print(f"Last date in filtered data: {df_flow.index[-1]}")
    print(f"Number of days: {len(df_flow)}")

    JDAY_init = (pd.Timestamp(start_date)-pd.Timestamp('1-1-1921')).days + 1 #calulate initial Julinan Day for added data
    print(JDAY_init)
    JDAYS = np.arange(JDAY_init, JDAY_init + len(df_flow.index)) #create Julian Day array
    print(JDAYS[-1])
    df_flow['JDAY'] = JDAYS #add Julian day array to flow dataframe

    
    ###################### create flow input files ###############################
    SPL_OUT = df_flow.SPL_OUT.values*0.028316847  # cfs to m3/s
    FKC_OUT = df_flow.FKC_OUT.values*0.028316847  # cfs to m3/s
    MC_OUT = df_flow.MC_OUT.values*0.028316847  # cfs to m3/s
    SJR_OUT = df_flow.SJR_OUT.values*0.028316847  # cfs to m3/s
    M_IN = np.abs(df_flow.M_IN.values*0.028316847)  # cfs to m3/s, take absolute value
    MIL_EVAP = df_flow.MIL_EVAP.values*0.028316847  # cfs to m3/s
    JDAY = df_flow.JDAY.values *1.000
    Temp_predicted = df_temp['%s_Temp'%analog_year].values *1.000
    zero_filler = np.zeros(len(df_flow.index))*1.000 #fill-in for qin br2-4
    
    # Before the loop, let's check array lengths
    print(f"Length of JDAY array: {len(JDAY)}")
    print(f"Length of Temp_predicted array: {len(Temp_predicted)}")

    # Use the shorter length for the loop
    loop_length = min(len(JDAY), len(Temp_predicted))
    print(f"Using loop length of: {loop_length}")

    for i in range(loop_length):  # This ensures we don't go beyond the shorter array
        JDAY[i] = '%0.2f'%JDAY[i]
        SPL_OUT[i] = '%0.2f'%SPL_OUT[i]
        FKC_OUT[i] = '%0.2f'%FKC_OUT[i]
        MC_OUT[i] = '%0.2f'%MC_OUT[i]
        SJR_OUT[i] = '%0.2f'%SJR_OUT[i]
        MIL_EVAP[i] = '%0.2f'%MIL_EVAP[i]
        M_IN[i] = '%0.2f'%M_IN[i]
        Temp_predicted[i] = '%0.2f'%Temp_predicted[i]
        zero_filler[i] = '%0.2f'%zero_filler[i]
    # changed 0.99 to 1.99 as this was causing an error during model run with input files being written missing the last day.
    #JDAY_end = JDAY[-1]+1.99
    #print(JDAY_end)
    ## create qot_br1.npt file
    with open('initial_files/mqot_br1_init.npt',"r") as f:
        lines = f.readlines()
    for i in range(0, len(JDAY)):
        l = f"\n{JDAY[i] : >8}{SPL_OUT[i] : >8}{FKC_OUT[i] : >8}{MC_OUT[i] : >8}{SJR_OUT[i] : >8}" #make line for CEQUAL timestep
        lines.append(l)
    #lines.append(f"\n{JDAY_end : >8}{0.0 : >8}{0.0 : >8}{0.0 : >8}{0.0 : >8}\n")
    
    with open('mqot_br1.npt',"w") as update:
        update.writelines(lines)
    update.close()
    
    
    ##create mqdt_br1.npt
    with open('initial_files/mqdt_br1_init.npt',"r") as f:
        lines = f.readlines()
    
    for i in range(0, len(JDAY)):
        l = f"\n{JDAY[i] : >8}{MIL_EVAP[i] : >8}"
        lines.append(l)
    #lines.append(f"\n{JDAY_end : >8}{0.0 : >8}\n")
    
    with open('mqdt_br1.npt',"w") as update:
        update.writelines(lines)
    update.close()
    
    
    ## create mqin_br1.npt
    with open('initial_files/mqin_br1_init.npt',"r") as f:
        lines = f.readlines()
    
    for i in range(0, len(JDAY)):
        l = f"\n{JDAY[i] : >8}{M_IN[i] : >8}"
        lines.append(l)
   # lines.append(f"\n{JDAY_end : >8}{0.0 : >8}\n")
    
    with open('mqin_br1.npt',"w") as update:
        update.writelines(lines)
    update.close()
    
    
    ### create mqin_br2-4
    with open('initial_files/mqin_br2_init.npt',"r") as f:
        lines = f.readlines()
    for i in range(0, len(JDAY)):
        l = f"\n{JDAY[i] : >8}{zero_filler[i] : >8}"
        lines.append(l)
    #lines.append(f"\n{JDAY_end : >8}{0.0 : >8}\n")
    with open('mqin_br2.npt',"w") as update:
        update.writelines(lines)
    update.close()
    
    with open('initial_files/mqin_br3_init.npt',"r") as f:
        lines = f.readlines()
    for i in range(0, len(JDAY)):
        l = f"\n{JDAY[i] : >8}{zero_filler[i] : >8}"
        lines.append(l)
    #lines.append(f"\n{JDAY_end : >8}{0.0 : >8}\n")
    with open('mqin_br3.npt',"w") as update:
        update.writelines(lines)
    update.close()
    
    with open('initial_files/mqin_br4_init.npt',"r") as f:
        lines = f.readlines()
    for i in range(0, len(JDAY)):
        l = f"\n{JDAY[i] : >8}{zero_filler[i] : >8}"
        lines.append(l)
    #lines.append(f"\n{JDAY_end : >8}{0.0 : >8}\n")
    with open('mqin_br4.npt',"w") as update:
        update.writelines(lines)
    update.close()
    
    ### create mtin_br1-4
    with open('initial_files/mtin_br1_init.npt',"r") as f:
        lines = f.readlines()
    for i in range(0, len(JDAY)):
        l = f"\n{JDAY[i] : >8}{Temp_predicted[i] : >8}"
        lines.append(l)
    #lines.append(f"\n{JDAY_end : >8}{0.0 : >8}\n")
    with open('mtin_br1.npt',"w") as update:
        update.writelines(lines)
    update.close()
    
    with open('initial_files/mtin_br2_init.npt',"r") as f:
        lines = f.readlines()
    for i in range(0, len(JDAY)):
        l = f"\n{JDAY[i] : >8}{Temp_predicted[i] : >8}"
        lines.append(l)
    #lines.append(f"\n{JDAY_end : >8}{0.0 : >8}\n")
    with open('mtin_br2.npt',"w") as update:
        update.writelines(lines)
    update.close()
    
    with open('initial_files/mtin_br3_init.npt',"r") as f:
        lines = f.readlines()
    for i in range(0, len(JDAY)):
        l = f"\n{JDAY[i] : >8}{Temp_predicted[i] : >8}"
        lines.append(l)
    #lines.append(f"\n{JDAY_end : >8}{0.0 : >8}\n")
    with open('mtin_br3.npt',"w") as update:
        update.writelines(lines)
    update.close()
    
    with open('initial_files/mtin_br4_init.npt',"r") as f:
        lines = f.readlines()
    for i in range(0, len(JDAY)):
        l = f"\n{JDAY[i] : >8}{Temp_predicted[i] : >8}"
        lines.append(l)
    #lines.append(f"\n{JDAY_end : >8}{0.0 : >8}\n")
    with open('mtin_br4.npt',"w") as update:
        update.writelines(lines)
    update.close()
    
    ###############################convert met data##########################
    df_met = pd.read_csv('met_data/%s_CEQUAL_met_inputs.csv'%analog_year, index_col = 0, parse_dates= True)

    # Calculate the exact JDAY range we need
    start_jday = JDAY_init
    end_jday = JDAYS[-1]

    # Filter met data based on JDAY values to match flow data exactly
    df_met = df_met[(df_met.JDAY >= start_jday) & (df_met.JDAY <= end_jday)]

    # Add debug prints to verify we have all days
    print("\nMet data range:")
    print(f"First met date: {df_met.index[0]}")
    print(f"Last met date: {df_met.index[-1]}")
    print(f"Number of met records: {len(df_met)}")
    print(f"First JDAY: {df_met.JDAY.iloc[0]}")
    print(f"Last JDAY: {df_met.JDAY.iloc[-1]}")
    print(f"Target end JDAY: {end_jday}")

    # The met file already has JDAY column, so we can use that directly
    JDAY = df_met.JDAY.values*1.000
    TAIR = df_met.TAIR.values*1.000  # cfs to m3/s
    TDEW = df_met.TDEW.values*1.000  # cfs to m3/s
    WIND = df_met.WIND.values*1.000  # cfs to m3/s
    PHI = df_met.PHI.values*1.000  # cfs to m3/s
    CLOUD = df_met.CLOUD.values*1.000  # cfs to m3/s
    SRO = df_met.SRO.values *1.000
    
    for i in range(0,len(JDAY)): #rounding to hundredths place
        JDAY[i] = '%0.2f'%JDAY[i]
        TAIR[i] = '%0.2f'%TAIR[i]
        TDEW[i] = '%0.2f'%TDEW[i]
        WIND[i] = '%0.2f'%WIND[i]
        PHI[i] = '%0.2f'%PHI[i]
        CLOUD[i] = '%0.2f'%CLOUD[i]
        SRO[i] = '%0.2f'%SRO[i]
    
    with open('initial_files/mmet3_init.npt',"r") as f:
        lines = f.readlines()
    
    for i in range(0, len(JDAY)):
        l = f"\n{JDAY[i] : >8}{TAIR[i] : >8}{TDEW[i] : >8}{WIND[i] : >8}{PHI[i] : >8}{CLOUD[i] : >8}{SRO[i] : >8}" #make line for CEQUAL timestep
        lines.append(l)
    lines.append("\n")
    
    with open('mmet3.npt',"w") as update:
        update.writelines(lines)
    update.close()
 
    # Just before model execution, restore from backup
    try:
        shutil.copy2('w2_con.csv.backup', 'w2_con.csv')
        print("Restored w2_con.csv from backup")
    except Exception as e:
        print(f"Error restoring w2_con.csv: {str(e)}")
        raise

    try:
        # Platform-specific configuration
        if os.name == 'nt':  # Windows
            process = subprocess.Popen(['w2_v45_64.exe'], 
                                    creationflags=subprocess.CREATE_NEW_CONSOLE)
        else:  # Linux/Unix
            process = subprocess.Popen(['./w2_v45_64.exe'],
                                    stdout=subprocess.PIPE, 
                                    stderr=subprocess.PIPE)
        
        print("Starting model execution...")
        # Give the model time to complete
        process.wait(timeout=wait_time)
        print("Model execution completed")
            
    except subprocess.TimeoutExpired:
        print(f"Model execution timed out after {wait_time} seconds")
        if os.name == 'nt':
            subprocess.run(['taskkill', '/F', '/IM', 'w2_v45_64.exe'])
        else:
            process.kill()
        raise
    except Exception as e:
        print(f"Error running executable: {str(e)}")
        raise

    try:
        # Get run name from user (only ask once at the start of the script)
        if 'run_name' not in locals():
            run_name = input("Enter the name for this run: ")
        
        # Create nested directory structure
        base_dir = '../CEQUAL_outputs'
        year_dir = os.path.join(base_dir, str(analog_year))
        run_dir = os.path.join(year_dir, run_name)
        
        # Create directories if they don't exist
        if not os.path.exists(year_dir):
            os.makedirs(year_dir)
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
        
        # Save model output files with year and run name in the filename
        shutil.copy2('two_31.csv', os.path.join(run_dir, f'two_31_{analog_year}_{run_name}.csv'))
        shutil.copy2('qwo_31.csv', os.path.join(run_dir, f'qwo_31_{analog_year}_{run_name}.csv'))
        shutil.copy2('tsr_1_seg31.csv', os.path.join(run_dir, f'tsr_1_seg31_{analog_year}_{run_name}.csv'))
        
        print(f"Model outputs saved to: {run_dir}")
        print(f"Files saved with format: [filename]_{analog_year}_{run_name}.csv")
        
    except Exception as e:
        print(f"Error saving model output: {str(e)}")





