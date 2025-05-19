import os
import sys
import matplotlib.pyplot as plt

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Modules.DataVis import QAQC_plot

def main():
    # Get user input
    year = input("Enter the year to plot (e.g., 2024): ")
    run_name = input("Enter the run name to plot (e.g., test_run): ")
    
    # Construct path to the TWO_31 file
    base_dir = '../CEQUAL_outputs'
    year_dir = os.path.join(base_dir, str(year))
    run_dir = os.path.join(year_dir, run_name)
    csv_file = os.path.join(run_dir, f'two_31_{year}_{run_name}.csv')
    
    # Check if file exists
    if not os.path.exists(csv_file):
        print(f"Error: Could not find file: {csv_file}")
        return
    
    try:
        # Create the QAQC plot
        print(f"\nCreating QAQC plot for {year} {run_name}...")
        plt = QAQC_plot(csv_file, int(year), run_name)
        print("Plot created successfully!")
        
        # Show the plot
        plt.show()
        
    except Exception as e:
        print(f"Error creating plot: {str(e)}")

if __name__ == "__main__":
    main() 