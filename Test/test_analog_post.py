import os
import sys
from pathlib import Path
from datetime import datetime

# Set matplotlib backend before importing matplotlib
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend instead of default

import matplotlib.pyplot as plt
import pandas as pd

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Modules.DataVis import Analog_Post_Plot, Analog_Post_Plot_Interactive, Analog_Post_Plot_Average, Analog_Post_3Plot_Avg, Analog_Post_Plot_TCD_Average

def run_all_plots():
    """
    Run all plots for the three analog scenarios
    """
    # Define model run names
    model_run_a = '2022_Analog_Obs'
    model_run_b = '2022_Madera_Analog'
    #model_run_c = '2018_Analog_C_May'
    
    # Define scenario display names
    scenario_a_name = "Analog A May"
    scenario_b_name = "Analog B May"
    #scenario_c_name = "Analog C May"
    
    #print(f"\nCreating plots for {model_run_a}, {model_run_b}, and {model_run_c}")
    print(f"\nCreating plots for {model_run_a} and {model_run_b}")
    print(f"Current working directory: {os.getcwd()}\n")
    
    # Define base directories
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'CEQUAL_outputs')
    input_dir = os.path.join(base_dir, '2022_Madera_TCD_Analog')
    
    # Print directory information
    print(f"Searching for files in:")
    print(f"Base directory: {base_dir}")
    print(f"Input directory: {input_dir}")
    
    # Define file paths
    file_a = os.path.join(input_dir, f'{model_run_a}_weighted.csv')
    file_b = os.path.join(input_dir, f'{model_run_b}_weighted.csv')
    #file_c = os.path.join(input_dir, f'{model_run_c}_weighted.csv')
    
    print(f"File A path: {file_a}")
    print(f"File B path: {file_b}")
    #print(f"File C path: {file_c}")
    
    # Check if files exist
    print("\nChecking for required files:")
    print(f"File A exists: {os.path.exists(file_a)}")
    print(f"File B exists: {os.path.exists(file_b)}")
    #print(f"File C exists: {os.path.exists(file_c)}")
    
    # Create basic comparison plots
    print("\nCreating basic comparison plots for scenarios A and B...")
    print(f"Creating plot for {scenario_a_name} and {scenario_b_name}")
    Analog_Post_Plot(model_run_a, model_run_b, input_dir, scenario_a_name, scenario_b_name)
    
    #print(f"\nCreating plot for {scenario_a_name} and {scenario_c_name}")
    #Analog_Post_Plot(model_run_a, model_run_c, input_dir, scenario_a_name, scenario_c_name)
    
    # Create average comparison plots
    print("\nCreating average comparison plots...")
    print(f"Creating average plot for {scenario_a_name} and {scenario_b_name}")
    Analog_Post_Plot_Average(model_run_a, model_run_b, input_dir)
    
    print(f"\nCreating average plot for {scenario_a_name} and {scenario_c_name}")
    Analog_Post_Plot_Average(model_run_a, model_run_c, input_dir)
    
    # Create interactive comparison plots
    print("\nCreating interactive comparison plots...")
    print(f"Creating interactive plot for {scenario_a_name} and {scenario_b_name}")
    Analog_Post_Plot_Interactive(model_run_a, model_run_b, input_dir, scenario_a_name, scenario_b_name)
    
    print(f"\nCreating interactive plot for {scenario_a_name} and {scenario_c_name}")
    Analog_Post_Plot_Interactive(model_run_a, model_run_c, input_dir, scenario_a_name, scenario_c_name)
    
    # Create three-scenario average plot
    print("\nCreating three-scenario average plot...")
    Analog_Post_3Plot_Avg(model_run_a, model_run_b, model_run_c, input_dir)
    print("\nAll plots have been created successfully!")

def test_analog_post_tcd_average():
    """Test the Analog_Post_Plot_TCD_Average function with sample data"""
    # Configuration
    YEAR = 2022
    MODEL_RUN_A = f"{YEAR}_Analog_Obs"
    MODEL_RUN_B = f"{YEAR}_Madera_Analog"
    #MODEL_RUN_C = f"{YEAR}_Analog_C_May"
    
    # Set up input directory path
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_dir = os.path.join(base_dir, 'CEQUAL_outputs', f"{YEAR}_Madera_TCD_Analog")
    
    print("\nTest configuration:")
    print(f"Year: {YEAR}")
    print(f"Model Run A: {MODEL_RUN_A}")
    print(f"Model Run B: {MODEL_RUN_B}")
    #print(f"Model Run C: {MODEL_RUN_C}")
    print(f"Base directory: {base_dir}")
    print(f"Input directory: {input_dir}")
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"\nError: Input directory not found: {input_dir}")
        print("Please run Analog_Year_Post.py first to generate the required files.")
        return
    
    # Check for required weighted files
    file_a = os.path.join(input_dir, f"{MODEL_RUN_A}_weighted.csv")
    file_b = os.path.join(input_dir, f"{MODEL_RUN_B}_weighted.csv")
    
    print("\nChecking for required files:")
    print(f"File A exists: {os.path.exists(file_a)}")
    print(f"File B exists: {os.path.exists(file_b)}")
    
    if not all(os.path.exists(f) for f in [file_a, file_b]):
        print("Error: One or more required files are missing")
        return
    
    # Read and check the contents of the files
    print("\nChecking file contents:")
    
    try:
        df_a = pd.read_csv(file_a)
        print(f"\nFile A columns: {df_a.columns.tolist()}")
        print(f"File A shape: {df_a.shape}")
        print("\nFirst few rows of File A:")
        print(df_a.head())
    except Exception as e:
        print(f"Error reading File A: {str(e)}")
        return
        
    try:
        df_b = pd.read_csv(file_b)
        print(f"\nFile B columns: {df_b.columns.tolist()}")
        print(f"File B shape: {df_b.shape}")
        print("\nFirst few rows of File B:")
        print(df_b.head())
    except Exception as e:
        print(f"Error reading File B: {str(e)}")
        return
    
    print(f"\nCreating TCD average comparison plot for year {YEAR}")
    print(f"Using files:")
    print(f"A: {file_a} (Regular weighted temperature)")
    print(f"B: {file_b} (TCD weighted temperature)")
    
    # Create the plot
    try:
        plt = Analog_Post_Plot_TCD_Average(
            model_run_a=MODEL_RUN_A,
            model_run_b=MODEL_RUN_B,
            input_dir=input_dir
        )
        plt.show()
    except Exception as e:
        print(f"\nError creating plot: {str(e)}")
        print("\nDebug information:")
        print(f"Model Run A: {MODEL_RUN_A}")
        print(f"Model Run B: {MODEL_RUN_B}")
        print(f"Input directory: {input_dir}")
        print(f"File A columns: {df_a.columns.tolist()}")
        print(f"File B columns: {df_b.columns.tolist()}")
        return

if __name__ == '__main__':
    #run_all_plots()
    test_analog_post_tcd_average() 