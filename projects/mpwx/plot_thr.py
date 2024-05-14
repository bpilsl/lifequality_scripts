import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import re
import sys
import warnings
from glob import glob
from scipy.optimize import curve_fit
from scipy.optimize import OptimizeWarning
from mpwx_interpreter import *

font_small = 14
font_large = 24

def v_to_q(x):
    # Q = C * U
    # with C ~ 2.8fF and electron charge
    # a charge of 16.56 ke-/V is evaluated
    return x * 16.56

def q_to_v(x):
    return x / 16.56

if __name__ == '__main__':
    # Process multiple data files
    data_files = glob(sys.argv[2] + '*.txt')
    data_files.sort()
    warnings.simplefilter("error", OptimizeWarning)
    warnings.filterwarnings('ignore')
    baseline = float(sys.argv[1])

    statistic = []
    for i, file in enumerate(data_files):
        thr = (float(re.search(r'thr_(\d+\.*\d*)', file).group(1))) * 1e3 - baseline  # convert to mV
        # mean, err = s_curve_stats(file, thr, len(data_files), i)
        df = readScurveData(file)
        # print(df)
        try:
            gaussFit = interpretScurve(df, doPlot=False)['halfWayGaussFit']
            mean = gaussFit[0]
            err = gaussFit[2]
            statistic.append([thr, mean, err])
        except Exception as e:
            print(f'file {file} not processable: {e}')

    # Plot average threshold values for different TDAC values
    statistic = np.array(statistic)
    statistic = statistic[statistic[:, 0].argsort()]  # sort by thr values

    x = statistic[:, 0]
    q = v_to_q(x)
    y = statistic[:, 1]
    err = statistic[:, 2]

    kV, dV = np.polyfit(x, y, 1)
    kQ, dQ = np.polyfit(q, y, 1)
    print(f'Fit V vs. V {kV:.4f} * x(mV) + {dV:.4f}\nFit Q vs. V {kQ:.4f} * x(Q) + {dQ:.4f}')
    if len(sys.argv) > 3:
        with open(sys.argv[3], 'w') as f:
            for i in range(len(x)):
                f.write(f'{x[i]} {y[i]} {err[i]}\n')

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.errorbar(x, y, yerr=err, fmt='o', markersize=4, capsize=8, label='Data')
    fit_data = x * kV + dV
    ax.plot(x, fit_data, linestyle='dashed', label=f'Fit: $\\frac{"{"}{kV:.3f}{"}"}{"{"}mV{"}"}$ * $V_{"{"}Thr{"}"}$ + {dV:.2f}mV')
    ax.legend(loc='upper left')
    ax.set_xlabel('$V_{Thr}$ [mV]', fontsize=font_small)
    ax.set_ylabel(r'$\mu(V_{inj, 50})$ [mV]', fontsize=font_small)
    ax.set_title('Avg. $V_{inj, 50}$ vs. threshold voltage', fontsize=font_large)
    # secax_x = ax.secondary_xaxis('top', functions=(v_to_q, q_to_v))
    # secax_x.set_xlabel('Threshold ($ke^-$)')
    secax_y = ax.secondary_yaxis('right', functions=(v_to_q, q_to_v))
    secax_y.set_ylabel(r'$\mu(V_{inj, 50})$ [$e^-$]', fontsize=font_small)
    ax.tick_params(axis='x', labelsize=font_small)
    ax.tick_params(axis='y', labelsize=font_small)
    secax_y.tick_params(axis='y', labelsize=font_small)
    ax.grid()
    plt.legend(prop={'size': font_small})

    plt.show()
