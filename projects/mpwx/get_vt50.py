from mpwx_interpreter import *
import argparse
from scipy.optimize import OptimizeWarning
import warnings
import matplotlib.pyplot as plt
import re
from glob import glob
import sys
import os
import pandas as pd

pd.set_option('mode.chained_assignment', None)
files = glob(os.path.join(sys.argv[1], '*.txt'))
files.sort()
injVoltage = float(sys.argv[2])
baseline = float(sys.argv[3])
first = True
grouped_dict = {}
pixelDataTemplate = {
    'Voltage': [],
    'Hits': [],
    'Index': [],
    'Injection': []
}
pixel_data = {'Pixel': [], 'Voltage': [], 'Hits': [], 'Index': []}
for file in files:
    df = readScurveData(file)
    thr = float(re.search(r'thr_(\d+\.*\d*)', file).group(1)) * 1e3 # get threshold voltage and convert to mV
    effectiveThr = thr - baseline
    # Explode the 'Voltage' and 'Hits' lists
    exploded_df = df.explode('Voltage').explode('Hits')

    # Select rows where 'Voltage' column contains the specific voltage
    selected_rows = exploded_df[exploded_df['Voltage'] == injVoltage]
    selected_rows['Injection'] = selected_rows['Voltage'][:]
    selected_rows['Voltage'][:] = effectiveThr

    # Group by 'Pixel' column and create a dictionary for each group
    for pixel, group in selected_rows.groupby('Pixel'):
        pixel_data['Pixel'].append(pixel)
        pixel_data['Voltage'].append(*group['Voltage'].tolist())
        pixel_data['Hits'].append(*group['Hits'].tolist())
        pixel_data['Index'].append(*group['Index'].tolist())
        pixel_dict = {
            'Voltage': group['Voltage'].tolist(),
            'Hits': group['Hits'].tolist(),
            'Index': group['Index'].tolist(),
            'Injection': injVoltage  # You can add more information if needed
        }
        grouped_dict[pixel] = pixel_dict

df = pd.DataFrame(pixel_data)
interpretScurve(df, xAxisLabel='Threshold voltage [mV]', title='Response at V$_{inj} = $' + str(injVoltage) + 'mV')
plt.show()
print(pd.DataFrame(pixel_data))
    # Print or use the new dataframe as needed
