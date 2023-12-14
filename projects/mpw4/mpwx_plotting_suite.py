import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import re
import sys
import warnings
from scipy.optimize import curve_fit
from scipy.optimize import OptimizeWarning

sensor_dim = (64, 64)


def plot_hitmap(file):
    hitmap = np.zeros(sensor_dim)

    with open(file, 'r') as f:
        row = 0
        for line in f:
            if line.startswith('#') or len(line) == 0 or len(line.strip()) == 0:
                # comment or empty or just whitespace line
                continue

            splitted = line.split(' ')
            if len(splitted) != sensor_dim[0] + 1:
                print('invalid line with ', len(splitted), ' entries in hitmap')
                return
            row = int(splitted[0]) # first number in line indicates row number of pixels
            if row >= sensor_dim[0]:
                print('invalid line', line)
                continue
            for col in range(1, sensor_dim[1] + 1):
                # each line contains number of hits for column index of pixel in current row,
                # column index corresponds to position in line
                # eg : 39 1 0 2 4 7 0 1 0 0 13 1 186 3 1 0 2 0 2 4 0 0 7 14 4 3 3 1 ...
                # corresponds to hits in row 39, pixel 39:00 got 1 hit, 39:01 got 0 hits, 39:02 got 2 hits, ...

                hits = int(splitted[col])
                hitmap[row, col - 1] = hits

        hot_pixel = np.argwhere(hitmap > 200)

        if hot_pixel.any():
            print('hot_pixels: ', hot_pixel)
        ax = sns.heatmap(hitmap)
        ax.invert_yaxis()


def plot_scurve(file):
    # Define the sigmoidal function (logistic function)
    def sigmoid(x, L, x0, k, b):
        y = L / (1 + np.exp(-k * (x - x0))) + b
        return (y)

    # Initialize lists to store data
    pixel_data = []
    current_pixel = {}

    # Read the file line by line
    with open(file, 'r') as f:
        for line in f:
            if line.startswith('#! Pixel'):
                if current_pixel:
                    pixel_data.append(current_pixel.copy())
                    current_pixel = {}
                _, pixel_info = line.split('#! Pixel')
                current_pixel['Pixel'] = pixel_info.strip()
                current_pixel['Voltage'] = []
                current_pixel['Hits'] = []
            elif line.startswith('0.') or line.startswith('1.'):
                voltage, hits, _ = line.split()
                current_pixel['Voltage'].append(float(voltage))
                current_pixel['Hits'].append(int(hits))

    # Append the last pixel's data
    if current_pixel:
        pixel_data.append(current_pixel.copy())

    for pixel in pixel_data:
        x_data = np.array(pixel['Voltage'])
        y_data = np.array(pixel['Hits'])
        plt.scatter(x_data, y_data, marker='.', label=f"Pixel {pixel['Pixel']}")
        try:
            p0 = [max(y_data), np.median(x_data), 1, min(y_data)]  # this is a mandatory initial guess
            popt, _ = curve_fit(sigmoid, x_data, y_data, p0, maxfev=10000)
            print(popt)
            plt.plot(x_data, sigmoid(x_data, *popt), label='fit')
        except RuntimeWarning:
            pass
            # plt.scatter(x_data, y_data, marker='.', label=f"Pixel {pixel['Pixel']} (Fit Skipped - Overflow)")
        except RuntimeError as rte:
            print(rte)
        except OptimizeWarning as ow:
            print('optimization failed for', pixel["Pixel"], ow)

    plt.xlabel('Injection Voltage [V]')
    plt.ylabel('Number of Hits')
    plt.title('Number of Hits vs. Injection Voltage for Pixels')
    # plt.legend()
    plt.grid(True)


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
    warnings.simplefilter("error", OptimizeWarning)
    file = sys.argv[1]
    data_type = deduce_data_type(file)
    if data_type == 'hitmap':
        plot_hitmap(file)
    elif data_type == 'scurve':

        plot_scurve(file)
    else:
        print('unsupported data type', data_type)
        exit(1)

    plt.show()
