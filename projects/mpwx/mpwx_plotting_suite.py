import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import argparse
import re
import sys
import warnings
from scipy.optimize import curve_fit
from scipy.optimize import OptimizeWarning


sensor_dim = (64, 64)


def plot_hitmap(file, plot_hist):
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
            row = int(splitted[0])  # first number in line indicates row number of pixels
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

        hot_pixel = np.argwhere(hitmap > 10)

        if hot_pixel.any():
            print('hot_pixels: ', hot_pixel)
        if plot_hist:
            ax1 = plt.subplot(221)
            ax1.set(title='Histogram', xlabel='TDAC values', ylabel='Counts')
            plt.xlim(0, 15)
            tdac_map = hitmap.flatten()
            tdac_map = tdac_map[tdac_map >= 0]  # get rid of -1
            plt.hist(tdac_map, bins=15)
            ax2 = plt.subplot(222)
            ax2.set_title('TDAC map')
        else:
            ax2 = plt.subplot(111)
            ax2.set_xlabel('column')
            ax2.set_ylabel('row')
        sns.heatmap(hitmap, annot=False)
        ax2.invert_yaxis()


def plot_scurve(file):
    # Define the sigmoidal function (logistic function)
    def sigmoid(x, L, x0, k, b):
        y = L / (1 + np.exp(-k * (x - x0))) + b
        return y

    def gaussian(x, amplitude, mean, stddev):
        return amplitude * np.exp(-((x - mean) / stddev) ** 2 / 2)

    def v_from_s(y, L, x0, k, b):
        return - 1 / k * np.log(L / (y - b) - 1) + x0

    pixel_data = []
    current_pixel = {}

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
                row, col = pixel_info.strip().split(':')
                row = int(row)
                col = int(col)
                current_pixel['Index'] = (row, col)
            elif line.startswith('0.') or line.startswith('1.'):
                # Extract voltage and hits information for each pixel
                try:
                    voltage, hits, _ = line.split()
                    current_pixel['Voltage'].append(float(voltage))
                    current_pixel['Hits'].append(int(hits))
                except:
                    print('broken line ', line)
                    continue

    # Append the last pixel's data
    if current_pixel:
        pixel_data.append(current_pixel.copy())

    vt50_map = np.zeros(sensor_dim)
    noise_map = np.zeros(sensor_dim)
    fig = plt.figure()
    gs = GridSpec(2, 3, figure=fig)
    ax1 = fig.add_subplot(gs[:, 0])
    ax1.set(title='Number of Hits vs. Injection Voltage for Pixels', xlabel='Injection Voltage [mV]',
            ylabel='Number of Hits')
    # ax1.legend()
    ax1.grid(True)
    for pixel in pixel_data:
        x_data = np.array(pixel['Voltage']) * 1e3  ##convert to mV
        y_data = np.array(pixel['Hits'])
        ax1.scatter(x_data, y_data, marker='.', label=f"Pixel {pixel['Pixel']}")
        plt.xlim(min(x_data), max(x_data))
        try:
            p0 = [max(y_data), np.median(x_data), 1, min(y_data)]  # this is a mandatory initial guess
            popt, _ = curve_fit(sigmoid, x_data, y_data, p0, maxfev=100000)
            x_fit = np.arange(min(x_data), max(x_data), (max(x_data) - min(x_data)) / 200)  # generate points for plot
            # of fit
            ax1.plot(x_fit, sigmoid(x_fit, *popt), label='fit')
            vt50_map[pixel["Index"][0], pixel['Index'][1]] = v_from_s(50, *popt)
            noise_map[pixel["Index"][0], pixel['Index'][1]] = v_from_s(84, *popt) - v_from_s(16, *popt)
            # print(pixel["Index"][0], pixel['Index'][1], noise_map[pixel["Index"][0], pixel['Index'][1]])
        except RuntimeWarning:
            pass
        except RuntimeError as rte:
            print(rte)
        except OptimizeWarning as ow:
            print('optimization failed for', pixel["Pixel"], ow)

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set(title='VT50 map', xlabel='Col', ylabel='Row')
    sns.heatmap(vt50_map)
    ax2.invert_yaxis()
    ax3 = fig.add_subplot(gs[0, 1])
    ax3.set(title='VT50 histogram', xlabel='VT50 [mV]', ylabel='Counts')
    no_nan = vt50_map.flatten()[~np.isnan(vt50_map.flatten())]
    counts, bins = np.histogram(no_nan, bins=100)

    bins = bins[1:]
    counts = counts[1:]  # 0th bin contains fit fails, or not scanned pixels

    # Calculate bin centers
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Use curve_fit to fit the Gaussian function to the histogram data
    initial_guess = [1.0, np.mean(no_nan), np.std(no_nan)]
    params, covariance = curve_fit(gaussian, bin_centers, counts, p0=initial_guess)
    amplitude, mean, stddev = params
    stddev = abs(stddev)
    x_fit = np.linspace(min(x_data), max(x_data), 1000)
    ax3.plot(x_fit, gaussian(x_fit, amplitude, mean, stddev), '--', label='Fit', color='black')
    ax3.hist(bins[:-1], bins, weights=counts)

    # box with statistics
    props = dict(boxstyle='round', facecolor='wheat', alpha=.5)
    stats = f'$\\mu$ = {mean:.1f}mV\n$\\sigma$ = {stddev:.1f}mV'
    ax3.text(0.75, 0.95, stats, transform=ax3.transAxes, fontsize=15,
             verticalalignment='top', bbox=props)

    ax3.grid()
    plt.xlim(min(x_data), max(x_data))

    ax4 = fig.add_subplot(gs[1, 2])
    ax4.set(title='Noise map', xlabel='Col', ylabel='Row')
    sns.heatmap(noise_map)
    ax4.invert_yaxis()
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set(title='Noise histogram', xlabel='Noise [mV]', ylabel='Counts')

    no_nan = noise_map.flatten()[~np.isnan(noise_map.flatten())]

    counts, bins = np.histogram(no_nan, bins=100)

    bins = bins[1:]
    counts = counts[1:]  # 0th bin contains fit fails, or not scanned pixels
    # Calculate bin centers
    bin_centers = (bins[:-1] + bins[1:]) / 2

    initial_guess = [1.0, np.mean(no_nan), np.std(no_nan)]
    params, covariance = curve_fit(gaussian, bin_centers, counts, p0=initial_guess)
    amplitude, mean, stddev = params
    stddev = abs(stddev)
    x_fit = np.linspace(0, 300, 1000)
    ax5.plot(x_fit, gaussian(x_fit, amplitude, mean, stddev), '--', label='Fit', color='black')
    ax5.hist(bins[:-1], bins, weights=counts)

    # box with statistics
    props = dict(boxstyle='round', facecolor='wheat', alpha=.5)
    stats = f'$\\mu$ = {mean:.1f}mV\n$\\sigma$ = {stddev:.1f}mV'
    ax5.text(0.75, 0.95, stats, transform=ax5.transAxes, fontsize=15,
             verticalalignment='top', bbox=props)

    ax5.grid()





def plot_spectrum(file):
    with open(file, 'r') as f:
        # counts = []
        for line in f:
            if line.startswith('#') or len(line) < 256:
                continue
            counts = (np.array(line.split())).astype(int)
        bins = np.arange(0, 257)

        mids = 0.5 * (bins[1:] + bins[:-1])
        mean = np.average(mids, weights=counts)
        var = np.average((mids - mean) ** 2, weights=counts)
        std_dev = np.sqrt(var)
        ax = plt.subplot(111)
        ax.set(xlabel='ToT [50ns]', ylabel='Counts', title='Spectrum')

        ax.text(200, 14, f'Avg: {mean:.2f}\nStdDev: {std_dev:.2f}')

        ax.stairs(counts, bins)


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
            plot_hitmap(file, False)
        elif data_type == 'scurve':
            plot_scurve(file)

        elif data_type == 'tdac_map':
            plot_hitmap(file, True)
        elif data_type == 'spectrum':
            plot_spectrum(file)
        else:
            print('unsupported data type', data_type)
            exit(1)

        if save_path:
            plt.savefig(f'{save_path}_{i}.png')
        else:
            plt.show()
