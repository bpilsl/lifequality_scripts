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
    # a charge of 17.5 ke-/V is evaluated
    return x * 17.5

def q_to_v(x):
    return x / 17.5

if __name__ == '__main__':
    # Process multiple data files
    data_files = glob(sys.argv[1] + '*.txt')
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

    power_items = ['p1v8_vdda', 'p1v8_vddc', 'p1v8_vdd!', 'p1v3_vssa', 'p1v8_nw_ring', 'p1v8_vsensbus', 'p2v5d']
    # power_items = ['p1v8_vdda', 'p1v8_vddc', 'p1v8_vdd!', 'p1v3_vssa', 'p1v8_nw_ring', 'p1v8_vsensbus']

    plt.rcParams['figure.figsize'] = [32, 18]

    data_files.sort()
    warnings.simplefilter("error", OptimizeWarning)
    warnings.filterwarnings('ignore')

    sStats = []
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
        totalPower = {'i': .0, 'p': .0}
        for i in report:
            if not i['name'] in power_items:
                continue
            powerStat['dac'].append(dacVal)
            powerStat['name'].append(i['name'])
            powerStat['u'].append(i['U'])
            powerStat['i'].append(i['I'])
            powerStat['p'].append(i['P'])
            totalPower['i'] += i['I']
            totalPower['p'] += i['P']


        # import pdb;pdb.set_trace()
        powerStat['dac'].append(dacVal)
        powerStat['name'].append('Total')
        powerStat['p'].append(totalPower['p'])
        powerStat['i'].append(totalPower['i'])
        powerStat['u'].append(0)  # nonsense but necessary for processing to have same length arrays
        try:
            df = readScurveData(file)
            result = interpretScurve(df, doPlot=False)
            gaussFit = result['halfWayGaussFit']
            meanResponse = gaussFit[0]
            errResponse = gaussFit[2]
            noise = result['noiseGaussFit']
            meanNoise = noise[0]
            errNoise = noise[2]

            sStats.append([dacVal, meanResponse, errResponse, meanNoise, errNoise])
        except Exception as e:
            print(f'file {file} not processable: {e}')

    fig = plt.figure()
    gs = GridSpec(2, 2, figure=fig)

    df = pd.DataFrame(powerStat)
    ax = fig.add_subplot(gs[:, 0])
    sns.lineplot(x='dac', y='p', hue='name', marker='o', data=df)
    plt.ylim(330, 370)
    plt.axvline(x=nominalDACval, color='r', linestyle='dashdot', label=f'Nominal DAC = {nominalDACval}', linewidth=.95)
    plt.grid()
    plt.legend(loc='upper left', fontsize=12)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    ax.set_xlabel(f'{dacName}', fontsize=15)
    ax.set_ylabel('P [mW]', fontsize=15)
    ax.set_title(f'Power consumption scan of {dacName}', fontsize=30)

    sStats = np.array(sStats)
    sStats = sStats[sStats[:, 0].argsort()]  # sort by thr values

    x = sStats[:, 0]
    y = sStats[:, 1]
    errResponse = sStats[:, 2]
    ax = fig.add_subplot(gs[0, 1])
    ax.errorbar(x, y, yerr=errResponse, fmt='o', markersize=2, capsize=5, label='Data')
    plt.axvline(x=nominalDACval, color='r', linestyle='dashdot', label=f'Nominal DAC = {nominalDACval}', linewidth=.95)
    ax.legend(loc='upper left', fontsize=15)
    ax.set_xlabel(f'{dacName}', fontsize=15)
    ax.set_ylabel(r'$\mu(V_{inj, 50})$ [mV]', fontsize=15)
    ax.set_title(f'Avg. 50% response {"$V_{inj, 50}$"} for different {dacName} values', fontsize=30)
    secax_y = ax.secondary_yaxis('right', functions=(v_to_q, q_to_v))
    secax_y.set_ylabel(r'$\mu(V_{inj, 50})$ [$e^-$]', fontsize=15)
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=13)
    secax_y.tick_params(axis='y', labelsize=13)
    ax.grid()

    x = sStats[:, 0]
    y = sStats[:, 3]
    errResponse = abs(sStats[:, 4])
    ax = fig.add_subplot(gs[1, 1])
    ax.errorbar(x, y, yerr=errResponse, fmt='o', markersize=2, capsize=5, label='Data')
    plt.axvline(x=nominalDACval, color='r', linestyle='dashdot', label=f'Nominal DAC = {nominalDACval}', linewidth=.95)
    ax.legend(loc='upper left', fontsize=15)
    ax.set_xlabel(f'{dacName}', fontsize=15)
    ax.set_ylabel(r'$\mu(Noise)$ [mV]', fontsize=15)
    ax.set_title(f'Avg. Noise for different {dacName} values', fontsize=30)
    secax_y = ax.secondary_yaxis('right', functions=(v_to_q, q_to_v))
    secax_y.set_ylabel(r'$\mu(Noise)$ [$e^-$]')
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=13)
    secax_y.tick_params(axis='y', labelsize=13)
    ax.grid()

    # plt.show()
    plt.savefig(sys.argv[2])

