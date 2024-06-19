import numpy as np
# from /home/bernhard/hephy/sw_dev/lifequality_scripts/projects/mpwx/mpwx_interpreter import *
import argparse
from scipy.optimize import OptimizeWarning
import warnings
import matplotlib.pyplot as plt
import re
from glob import glob
import os
import pandas as pd
import sys
# from home.bernhard.hephy.sw_dev.lifequality_script.projects.mpws.mpwx_interpreter import *
import matplotlib as mpl

sys.path.append('/home/bernhard/hephy/sw_dev/lifequality_scripts/projects/mpwx')
from mpwx_interpreter import *


pd.set_option('mode.chained_assignment', None)

def parse_args():
    parser = argparse.ArgumentParser(description='Analysis script for multiple directories.')
    parser.add_argument('dirs', nargs='+', help='List of directories to analyze')
    return parser.parse_args()

def main():
    args = parse_args()

    first = True
    for directory in args.dirs:
        files = glob(os.path.join(directory, '*'))
        files.sort()

        data = {'dac':[], 'mean':[], 'err':[]}
        for file in files:
            dac_match = re.search(r'dacScan_.+_(\w+?)(\d+)', file)
            if dac_match:  # if we plot a dac folder, get a color in the range from 0 to 63
                dac_val = int(dac_match.group(2))
                dac_name = str.upper(dac_match.group(1))
                c_norm = mpl.colors.Normalize(vmin=0, vmax=63)

                c_map = mpl.cm.viridis
                s_map = mpl.cm.ScalarMappable(cmap=c_map, norm=c_norm)
                s_map.set_array([])
                color = s_map.to_rgba(dac_val)
                data['dac'].append(dac_val)
            else:
                color = None

            hist, map = readSpectrum(file)
            non_zero_indices = hist > 0
            bins = np.arange(256)
            non_zero_bins = bins[non_zero_indices]
            non_zero_counts = hist[non_zero_indices]

            mean = np.average(bins, weights=hist)
            std_dev = np.sqrt(np.average((bins - mean) ** 2, weights=hist))
            err = std_dev / np.sqrt(sum(bins))
            data['mean'].append(mean)
            data['err'].append(err)
        df = pd.DataFrame(data)
        df = df.sort_values('dac')

        ax = plt.subplot(111)
        plt.title('Mean ToT ' + dac_name)
        ax.errorbar(df['dac'], df['mean'], df['err'], fmt='o', markersize=2, capsize=5, label=dac_name, c='b')
        plt.plot(df['dac'], df['mean'], c='b')

        ax.set_xlabel(dac_name, fontsize=30)
        ax.set_ylabel(r'$\mu$(ToT)', fontsize=30)
        plt.legend(fontsize=12)
        # plt.ylim(0, max(fit_in_mean) * 1.2)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
