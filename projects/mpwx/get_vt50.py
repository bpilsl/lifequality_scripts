import numpy as np

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
injVoltage = None
plotSingle = False
baseline = float(sys.argv[2])
if len(sys.argv) == 4:
    injVoltages = [float(sys.argv[3])]
    plotSingle = True

first = True
grouped_dict = {}
pixelDataTemplate = {
    'Voltage': [],
    'Hits': [],
    'Index': [],
    'Injection': []
}
pixel_data = {'Pixel': [], 'Voltage': [], 'Hits': [], 'Index': []}
stats = {'VInj': [], 'VT50Err': [], 'VT50Mean': []}
dummy = readScurveData(files[0])
df = pd.DataFrame(dummy)
if not plotSingle:
    injVoltages = df['Voltage'].unique()#[10:15]
    # injVoltages = injVoltages[injVoltages > 161]
    # injVoltages = injVoltages[injVoltages < 310]

for injVoltage in injVoltages:
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
    for key in pixel_data.keys():
        pixel_data[key] = []
    s_data = interpretScurve(df, doPlot=plotSingle, xAxisLabel='Threshold voltage [mV]', title='Response at V$_{inj} = $' + str(injVoltage) + 'mV')
    if not s_data['halfWayGaussFit']:
        continue

    vt50Mean = s_data['halfWayGaussFit'][0]
    vt50Err = s_data['halfWayGaussFit'][2]
    stats['VT50Mean'].append(vt50Mean)
    stats['VT50Err'].append(vt50Err)
    stats['VInj'].append(injVoltage)

if not plotSingle:
    fitInMean = np.array(stats['VT50Mean'])
    fitInInj = np.array(stats['VInj'])
    # mask = np.bitwise_and(fitInInj > 160, fitInInj < 310)
    mask = np.bitwise_and(fitInInj > 330, fitInInj < 490)
    # import pdb;pdb.set_trace()
    fitInInj = fitInInj[mask]
    fitInMean = fitInMean[mask]
    fit = np.polyfit(fitInInj, fitInMean, deg=1)
    fitQ = np.polyfit(v_to_q(fitInInj), fitInMean, deg=1)

    print('Fit U:', fit, '\nFit Q', fitQ)
    fitX = np.linspace(min(fitInInj), max(fitInInj), 100)
    fitY = np.poly1d(fit)(fitX)
    ax = plt.subplot(111)
    df = pd.DataFrame(stats)
    # sns.lineplot(data=df, x='VInj', y='VT50Mean', marker='o')
    ax.errorbar(df['VInj'], df['VT50Mean'], yerr=df['VT50Err'], fmt='o', markersize=2, capsize=5, label='Data')
    ax.plot(fitX, fitY, linestyle='dashed', label=f'Fit: {fitQ[0]:.4f} * Q + {fitQ[1]:.2f}')
    plt.title(f'Gain = {fitQ[0] * 1e3:.2f} $\mu V / e^-$')
    ax.set_xlabel('$V_{Inj}$ [mV]')
    ax.set_ylabel('VT50 [mV]')
    plt.legend()
    plt.grid()
    secax_y = ax.secondary_yaxis('right', functions=(v_to_q, q_to_v))
    secax_y.set_ylabel('VT50 [$e^-$]')
    secax_x = ax.secondary_xaxis('top', functions=(v_to_q, q_to_v))
    secax_x.set_xlabel(r'$Q_{inj}$ [$e^-$]')

    # plt.ylim(0, 500)

plt.show()
