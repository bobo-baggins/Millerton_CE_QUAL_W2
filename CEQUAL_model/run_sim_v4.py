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
start_date = '02/07/2024 1:00' #start date of simulation (always start at 1:00)
end_date ='12/15/2024 23:00' #end date of simulation (always at at 23:00)
analog_years =  [2024]# available:  [1988,1989,1990,1994,2002,2007,2008,2013,2020,2021,2022,2023]
wait_time = 300 #seconds between simulations. Program will break if simulations take longer than wait time
#Code will excute after X amount of seconds, error if Model hasn't finished running by the end.
#Minimum wait time is 90 seconds.
make_output_folders = False #makes output folders, only if none exist
############################################################################################
############################################################################################
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)
############################################################################################
############################################################################################

for analog_year in analog_years:
    print(analog_year)
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
    M_IN = df_flow.M_IN.values*0.028316847  # cfs to m3/s
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
    df_met = pd.read_csv('met_data/%s_CEQUAL_met_inputs.csv'%analog_year,index_col = 0, parse_dates= True)
    df_met = df_met[(df_met.index >= start_date) & (df_met.index <= end_date)]
    #This creates the JDAY column for the met data. Commenting out for testing due to long run time
    hour_cycle = [0,0.04,0.08,0.13,0.17,0.21,0.25,0.29,0.34,0.38,0.42,0.46,0.50,0.55,0.59,0.63,0.67,0.71,0.76,0.80,0.84,0.88,0.92,0.97]
    num_days = (end_date - start_date).days + 1  # Get exact number of days needed
    JDAYS_hourly = []
    for d in range(JDAY_init, JDAY_init + num_days):
        for h in hour_cycle:
            JDAYS_hourly.append(d+h)
    JDAYS_hourly = JDAYS_hourly[1:]  # Remove first hour

    # Add a check to ensure lengths match
    if len(JDAYS_hourly) != len(df_met):
        print(f"Warning: JDAYS_hourly length ({len(JDAYS_hourly)}) doesn't match df_met length ({len(df_met)})")
        print(f"First JDAY value: {JDAYS_hourly[0]}")
        print(f"Last JDAY value: {JDAYS_hourly[-1]}")

    df_met['JDAY'] = JDAYS_hourly
    
    JDAY = df_met.JDAY.values*1.000  # cfs to m3/s
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
    
    # # Clean up any existing output files before running, except w2_con.csv
    # for csv_file in glob.glob("*.csv"):
    #     if csv_file != "w2_con.csv":  # Skip w2_con.csv
    #         try:
    #             os.remove(csv_file)
    #         except Exception as e:
    #             print(f"Warning: Could not remove existing file {csv_file}: {e}")

    # Run the executable and wait for completion
    try:
        # Platform-specific configuration
        if os.name == 'nt':  # Windows
            process = subprocess.Popen(['w2_v45_64.exe'], 
                                    stdout=subprocess.PIPE, 
                                    stderr=subprocess.PIPE,
                                    creationflags=subprocess.CREATE_NEW_CONSOLE)
        else:  # Linux/Unix
            process = subprocess.Popen(['./w2_v45_64.exe'],
                                    stdout=subprocess.PIPE, 
                                    stderr=subprocess.PIPE)
        
        # Track file modifications
        last_modification_time = time.time()
        stable_period = 10  # Time in seconds with no file changes to consider complete
        
        start_time = time.time()
        while time.time() - start_time < wait_time:
            # Check all CSV files
            current_time = time.time()
            newest_modification = last_modification_time
            
            for csv_file in glob.glob("*.csv"):
                if csv_file != "w2_con.csv":
                    try:
                        mtime = os.path.getmtime(csv_file)
                        newest_modification = max(newest_modification, mtime)
                    except Exception:
                        continue
            
            # If no new modifications for stable_period seconds and we have some output files
            if (current_time - newest_modification > stable_period and 
                len([f for f in glob.glob("*.csv") if f != "w2_con.csv"]) > 0):
                print("No file changes detected for 5 seconds, assuming model has completed")
                process.kill()
                break
                
            last_modification_time = newest_modification
            time.sleep(1)  # Wait 1 second before checking again
            
        else:  # This runs if the while loop completes without finding stable files
            process.kill()
            print(f"Error: Process timed out after {wait_time} seconds")
            raise TimeoutError("Model did not complete within the specified time")
            
    except Exception as e:
        print(f"Error running executable: {str(e)}")
        raise

    source_dir = r'../CEQUAL_model'
    dest_dir =  r'../CEQUAL_outputs/%s'%analog_year  

    if make_output_folders == True:         
        os.mkdir(dest_dir)

    # opt_files = glob.iglob(os.path.join("*.opt"))
    # for file in opt_files:
    #     if os.path.isfile(file):
    #         shutil.copy2(file, dest_dir)      
            
    npt_files = glob.iglob(os.path.join("*.npt"))
    for file in npt_files:
        if os.path.isfile(file):
            shutil.copy2(file, dest_dir)      
        # os.mkdir(dest_dir)
    csv_files = glob.iglob(os.path.join("*.csv"))
    for file in csv_files:
        if os.path.isfile(file):
            shutil.copy2(file, dest_dir)      

    # with open(f'../CEQUAL_outputs/{analog_year}/str_br1.opt',"r") as f:
    #     lines = f.readlines()
    # lines.pop(0)
    # lines.pop(0)
    
    # # Convert lines to numpy arrays and print lengths for debugging
    # print(f"Number of lines from opt file: {len(lines)}")
    
    # for i,l in enumerate(lines):
    #     lines[i] = np.array(list(filter(None, np.array(l.split(' ')))), dtype = np.float32)
    
    # # Create date range and print lengths
    # ix = pd.date_range(start = start_date, end = end_date, freq = 'h')
    # print(f"Length of date range: {len(ix)}")
    # print(f"First date: {ix[0]}")
    # print(f"Last date: {ix[-1]}")
    
    # # Clip the longer array to match the shorter one
    # if len(ix) > len(lines):
    #     ix = ix[:len(lines)]
    # elif len(lines) > len(ix):
    #     lines = lines[:len(ix)]
        
    # print(f"Final lengths - ix: {len(ix)}, lines: {len(lines)}")
    
    # # Create DataFrame with matched lengths
    # df = pd.DataFrame(index = ix, columns = ['JDAY', 'SPL_temp_C', 'FKC_temp_C', 'MC_temp_C', 'SJR_temp_C',
    #                               'SPL_Q_m3s', 'FKC_Q_m3s', 'MC_Q_m3s', 'SJR_Q_m3s',
    #                               'SPL_ELEVCL_m', 'FKC_ELEVCL_m', 'MC_ELEVCL_m', 'SJR_ELEVCL_m'])
    # df[df.columns] = lines
    # dest_dir = r'../output_csvs/%s-analog-%s-exceedance'%(analog_year,exceedance)
    # os.mkdir(dest_dir)
    #df.to_csv('../output_csvs/outflow_temps/%s_temp_analog-release_outputs.csv'%(analog_year))

    # can't locate parent directory for whatever reason... my virtual environment?
    #This is a quick fix..  .   .
   # df.to_csv('%s_analog-release_outputs.csv'%(analog_year))

    # After creating and processing the DataFrame...
    
    # Create output directory if it doesn't exist
    dest_dir = f'CEQUAL_outputs/{analog_year}'
    os.makedirs(dest_dir, exist_ok=True)
    
    # Save only the temperature output DataFrame
    #df.to_csv(f'CEQUAL_outputs/{analog_year}/{analog_year}_temp_analog-release_outputs.csv')
    
    # Remove the old output_csvs save
    # df.to_csv('output_csvs/outflow_temps/%s_temp_analog-release_outputs.csv'%(analog_year))
