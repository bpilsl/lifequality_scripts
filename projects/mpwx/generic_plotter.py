from mpwx_interpreter import *
import argparse
from scipy.optimize import OptimizeWarning
import warnings
import matplotlib.pyplot as plt
import re

def deduce_data_type(file):
    with open(file, 'r') as f:
        first_line = f.readline()
        match = re.search(r'type = (\w+)', first_line)
        if match:
            return match.group(1)
        else:
            print('unable to deduce data type from line ', first_line)
            return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plotting suite for data of the RD50-MPWx series")
    parser.add_argument('-s', '--save_path', help='Path to save plot to')
    parser.add_argument('files', nargs='+', help='Input file(s)')
    args = parser.parse_args()
    save_path = args.save_path

    warnings.simplefilter("error", OptimizeWarning)
    for i, file in enumerate(args.files):
        data_type = deduce_data_type(file)
        if data_type == 'hitmap':
            data = readHitmap(file)
            plotHitmap(data)
        elif data_type == 'scurve':
            data = readScurveData(file)
            interpretScurve(data)

        elif data_type == 'tdac_map':
            data = readHitmap(file)
            plotHitmap(data, plotHist=True)

        elif data_type == 'spectrum':
            pass
            # plot_spectrum(file)
        else:
            print('unsupported data type', data_type)
            exit(1)

        if save_path:
            plt.savefig(f'{save_path}_{i}.png')
        else:
            plt.show()