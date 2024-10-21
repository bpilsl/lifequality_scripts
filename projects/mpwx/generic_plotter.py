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
    parser.add_argument('-c', '--calib_file', nargs='?', default=None, help='Path to capacitance calibration file')
    parser.add_argument('-t', '--type', default=None, choices=['hm', 's', 'n', 'c'], help='Type of the data (options: hm, s, n, c)')
    parser.add_argument('files', nargs='+', help='Input file(s)')
    args = parser.parse_args()
    save_path = args.save_path
    t = args.type
    calib_file = args.calib_file

    cap_map = None
    if calib_file:
        cap_map = parse_capmap(calib_file)

    data_type = None
    if t:
        if t == 'hm':
            data_type = 'hitmap'
        elif t == 's':
            data_type = 'scurve'
        elif t == 'c':
            data_type = 'tdac_map'
        elif t == 'n':
            data_type = 'noise_map'

    warnings.simplefilter("error", OptimizeWarning)
    for i, file in enumerate(args.files):
        if not data_type:
            data_type = deduce_data_type(file)
        if data_type == 'hitmap':
            data, _ = readHitmap(file)
            plotHitmap(data)
        elif data_type == 'scurve':
            data = readScurveData(file)
            interpretScurve(data, cap_map=cap_map)
            # powerReport = getPowerReport(file)
            # for i in powerReport:
            #     pass
                #print(f'{i["name"]}: U = {i["U"]}V, I = {i["I"]}mA, P = {i["P"]}mW')

        elif data_type == 'tdac_map':
            data, _ = readHitmap(file)
            plotHitmap(data, plotHist=True)

        elif data_type == 'spectrum':
            data = readSpectrum(file)
            plotSpectrum(*data)
        elif data_type == 'noise_map':
            data, time = readHitmap(file)
            plotNoise(data, time)

        else:
            print('unsupported data type', data_type)
            exit(1)

        if save_path:
            plt.savefig(f'{save_path}_{i}.png')
        else:
            plt.show()
