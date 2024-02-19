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
import seaborn as sns

def v_to_q(x):
    # Q = C * U
    # with C ~ 2.8fF and electron charge
    # a charge of 16.56 ke-/V is evaluated
    return x * 16.56

def q_to_v(x):
    return x / 16.56

if __name__ == '__main__':
    # Process multiple data files
    data_files = glob(sys.argv[1] + '*.txt')
    # nominalDACval = int(sys.argv[2], 0)
    register_defaults = {
        "VPCOMP": 0x13,
        "VPTRIM": 0x24,
        "VNSENSBIAS": 0x32,
        "VBLR": 0x26,
        "VNSF": 0x2D,
        "VNFB": 0x12,
        "VBFB": 0x26,
        "VPBIAS": 0x25,
        "VN": 0x15
    }
    data_files.sort()
    warnings.simplefilter("error", OptimizeWarning)
    warnings.filterwarnings('ignore')

    vt50Stat = []
    powerStat = {'dac': [], 'name': [], 'u': [], 'i': [], 'p': []}
    for file in data_files:
        dacMatch = re.search(r'DAC_(.+?)(\d+)', file)
        if dacMatch:
            dacName = dacMatch.group(1).upper()
            dacVal = int(dacMatch.group(2))
            nominalDACval = register_defaults[dacName]
        else:
            continue
        report = getPowerReport(file)
        for i in report:
            powerStat['dac'].append(dacVal)
            powerStat['name'].append(i['name'])
            powerStat['u'].append(i['U'])
            powerStat['i'].append(i['I'])
            powerStat['p'].append(i['P'])

        try:
            df = readScurveData(file)
            gaussFit = interpretScurve(df, doPlot=False)['halfWayGaussFit']
            mean = gaussFit[0]
            err = gaussFit[2]
            vt50Stat.append([dacVal, mean, err])
        except Exception as e:
            print(f'file {file} not processable: {e}')

    df = pd.DataFrame(powerStat)
    ax = plt.subplot(121)
    sns.lineplot(x='dac', y='i', hue='name', marker='o', data=df)
    plt.axvline(x=nominalDACval, color='r', linestyle='--', label=f'Nominal DAC = {nominalDACval}')
    plt.grid()
    plt.legend()
    ax.set_xlabel(f'{dacName}')
    ax.set_ylabel('I[mA]')
    ax.set_title(f'Power consumption scan of {dacName}')

    vt50Stat = np.array(vt50Stat)
    vt50Stat = vt50Stat[vt50Stat[:, 0].argsort()]  # sort by thr values

    x = vt50Stat[:, 0]
    q = v_to_q(x)
    y = vt50Stat[:, 1]
    err = vt50Stat[:, 2]

    ax = plt.subplot(122)
    ax.errorbar(x, y, yerr=err, fmt='o', markersize=2, capsize=5, label='Data')
    plt.axvline(x=nominalDACval, color='r', linestyle='--', label=f'Nominal DAC = {nominalDACval}')
    ax.legend(loc='upper left')
    ax.set_xlabel(f'{dacName}')
    ax.set_ylabel(r'$\mu(VT50)$ [mV]')
    ax.set_title(f'Avg. VT50 for different {dacName} values')
    secax_y = ax.secondary_yaxis('right', functions=(v_to_q, q_to_v))
    secax_y.set_ylabel(r'$\mu(VT50)$ [$e^-$]')
    ax.grid()

    plt.show()
