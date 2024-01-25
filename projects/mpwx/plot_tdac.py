import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import re
import sys
import warnings
from glob import glob
from scipy.optimize import curve_fit
from scipy.optimize import OptimizeWarning
import argparse

sensor_dim = (64, 64)


def process_scurve(file, tdac, n, do_plot):
    # Define the sigmoidal function (logistic function)
    def sigmoid(x, L, x0, k, b):
        y = L / (1 + np.exp(-k * (x - x0))) + b
        return y

    def gaussian(x, amplitude, mean, stddev):
        return amplitude * np.exp(-((x - mean) / stddev) ** 2 / 2)

    def vt_from_s(y, L, x0, k, b):
        return -1 / k * np.log(L / (y - b) - 1) + x0

    pixel_data = []
    current_pixel = {}

    # Read pixel data from file
    with open(file, 'r') as f:
        for line in f:
            if line.startswith('#! Pixel'):
                # Start of a new pixel's data
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
    if do_plot:
        # Plot individual pixel data if enabled
        ax1 = plt.subplot(2, n, tdac + 1)
        ax1.set(title='TDAC ' + str(tdac))
        ax1.grid(True)

    # Process each pixel's data
    for pixel in pixel_data:
        x_data = np.array(pixel['Voltage'])
        y_data = np.array(pixel['Hits'])
        if do_plot:
            ax1.scatter(x_data, y_data, marker='.', label=f"Pixel {pixel['Pixel']}")
        try:
            p0 = [max(y_data), np.median(x_data), 1, min(y_data)]  # Initial guess for curve fitting
            popt, _ = curve_fit(sigmoid, x_data, y_data, p0, maxfev=100000)
            x_fit = np.arange(min(x_data), max(x_data), (max(x_data) - min(x_data)) / 200)  # Generate points for fit plot
            if do_plot:
                ax1.plot(x_fit, sigmoid(x_fit, *popt), label='fit')
            vt50_map[pixel["Index"][0], pixel['Index'][1]] = vt_from_s(50, *popt)
        except (RuntimeWarning, RuntimeError) as e:
            print(e)
            continue
        except OptimizeWarning as ow:
            print('optimization failed for', pixel["Pixel"], ow)

    # Compute statistics for threshold values
    no_zeros = (vt50_map[vt50_map != 0]).flatten()
    no_nan = no_zeros[~np.isnan(no_zeros)]
    if len(no_zeros) == 0:
        print('no valid points in file ', file)
        return
    counts, bins = np.histogram(no_nan)

    bins = bins[1:]
    counts = counts[1:]  # 0th bin contains fit fails, or not scanned pixels

    # Calculate bin centers
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Use curve_fit to fit the Gaussian function to the histogram data
    initial_guess = [1.0, np.mean(no_nan), np.std(no_nan)]
    params, covariance = curve_fit(gaussian, bin_centers, counts, p0=initial_guess)
    amplitude, mean, stddev = params
    stddev = abs(stddev)
    std_err = stddev / np.sqrt(sum(counts))

    if do_plot:
        x_fit = np.linspace(min(x_data), max(x_data), 1000)
        ax3 = plt.subplot(2, n, tdac + n + 1)
        ax3.stairs(counts, bins)
        ax3.plot(x_fit, gaussian(x_fit, amplitude, mean, stddev), '--', label='Fit', color='black')
        ax3.hist(bins[:-1], bins, weights=counts)

        # box with statistics
        props = dict(boxstyle='round', facecolor='wheat', alpha=.5)
        stats = f'$\\mu$= {mean * 1000.0:.1f}mV\n$\\sigma$={stddev * 1000.0:.1f}mV'
        ax3.text(0.55, 0.95, stats, transform=ax3.transAxes, fontsize=8,
                 verticalalignment='top', bbox=props)
        ax3.grid()
        plt.xlim(min(x_data), max(x_data))
    return mean, std_err


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script to process and plot S-curves from evalTdac method.')
    parser.add_argument('directory', help='Directory containing data files')
    parser.add_argument('-s', '--scurves', action='store_true', help='Plot full scurves for current folder')
    args = parser.parse_args()

    # Process multiple data files
    data_files = glob(args.directory + '*.txt')
    
    warnings.simplefilter("error", OptimizeWarning)

    statistic = []
    for file in data_files:
        tdac = int(re.search(r'tdac_(\d+)', file).group(1))
        mean, err = process_scurve(file, tdac, len(data_files), args.scurves)
        statistic.append([tdac, mean, err])

    if not args.scurves:
        # Plot average threshold values for different TDAC values
        statistic = np.sort(statistic, 0)
        x = [row[0] for row in statistic]
        y = [row[1] for row in statistic]
        err = [row[2] for row in statistic]
        k, d = np.polyfit(x, y, 1)
        if len(sys.argv) > 2:
            with open(sys.argv[2], 'w') as f:
                for i in range(len(x)):
                    f.write(f'{x[i]} {y[i]} {err[i]}\n')

        ax = plt.subplot(111)
        ax.errorbar(x, y, yerr=err, fmt='o', markersize=8, capsize=20, label='Data')
        fit_data = np.array(x) * k + d
        ax.plot(x, fit_data, linestyle='dashed', label=f'Fit: {k:.4f} * x + {d:.2f}')
        ax.legend(loc='upper left')
        ax.set_xlabel('TDAC')
        ax.set_ylabel('$\mu$ [V]')
        ax.set_title('Avg. threshold for different TDAC values')

    plt.show()
