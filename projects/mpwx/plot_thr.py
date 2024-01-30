import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import re
import sys
import warnings
from glob import glob
from scipy.optimize import curve_fit
from scipy.optimize import OptimizeWarning

sensor_dim = (64, 64)
plot_s = False  # Flag for plotting individual pixel data


def s_curve_stats(file, thr, nFiles, iFile):
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
    if plot_s:
        # Plot individual pixel data if enabled
        ax1 = plt.subplot(2, nFiles, iFile + 1)
        ax1.set(title=f'Thr =  {thr * 1e3:.1f}mV')
        ax1.grid(True)

    # Process each pixel's data
    for pixel in pixel_data:
        x_data = np.array(pixel['Voltage'])
        y_data = np.array(pixel['Hits'])
        if plot_s:
            ax1.scatter(x_data, y_data, marker='.', label=f"Pixel {pixel['Pixel']}")
        try:
            p0 = [max(y_data), np.median(x_data), 1, min(y_data)]  # Initial guess for curve fitting
            popt, _ = curve_fit(sigmoid, x_data, y_data, p0, maxfev=100000)
            x_fit = np.arange(min(x_data), max(x_data), (max(x_data) - min(x_data)) / 200)  # Generate points for fit plot
            if plot_s:
                ax1.plot(x_fit, sigmoid(x_fit, *popt), label='fit')
                plt.xlim(min(x_data), max(x_data))
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
    if plot_s:
        ax3 = plt.subplot(2, nFiles, iFile + nFiles + 1)
        ax3.stairs(counts, bins)
        x_fit = np.linspace(min(x_data), max(x_data), 1000)
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


def deduce_data_type(file):
    # Deduce data type from the file
    with open(file, 'r') as f:
        first_line = f.readline()
        match = re.search(r'type = (\w+)', first_line)
        if match:
            return match.group(1)
        else:
            print('unable to deduce data type from line ', first_line)
            return None

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
    data_files.sort()
    warnings.simplefilter("error", OptimizeWarning)

    statistic = []
    for i, file in enumerate(data_files):
        thr = float(re.search(r'thr_(\d+\.\d+)', file).group(1)) - .9
        mean, err = s_curve_stats(file, thr, len(data_files), i)
        statistic.append([thr, mean, err])

    if not plot_s:
        # Plot average threshold values for different TDAC values
        statistic = np.sort(statistic, 0)
        x = np.array([row[0] for row in statistic])
        x = x * 1e3  # convert to mV
        q = v_to_q(x)
        y = np.array([row[1] for row in statistic])
        y = y * 1e3  # convert to mV
        err = [row[2] for row in statistic]
        kV, dV = np.polyfit(x, y, 1)
        kQ, dQ = np.polyfit(q, y, 1)
        print(f'Fit V vs. V {kV:.4f} * x(mV) + {dV:.4f}\nFit Q vs. V {kQ:.4f} * x(Q) + {dQ:.4f}')
        if len(sys.argv) > 2:
            with open(sys.argv[2], 'w') as f:
                for i in range(len(x)):
                    f.write(f'{x[i]} {y[i]} {err[i]}\n')

        ax = plt.subplot(111)
        ax.errorbar(x, y, yerr=err, fmt='o', markersize=8, capsize=20, label='Data')
        fit_data = np.array(x) * kV + dV
        ax.plot(x, fit_data, linestyle='dashed', label=f'Fit: {kV:.4f} * x + {dV:.2f}')
        ax.legend(loc='upper left')
        ax.set_xlabel('Threshold [mV]')
        ax.set_ylabel('$VT50_\mu$ [mV]')
        ax.set_title('Avg. VT50 for different threshold voltages')
        # secax_x = ax.secondary_xaxis('top', functions=(v_to_q, q_to_v))
        # secax_x.set_xlabel('Threshold ($ke^-$)')
        secax_y = ax.secondary_yaxis('right', functions=(v_to_q, q_to_v))
        secax_y.set_ylabel(r'$VT50_\mu$ ($ke^-$)')
        ax.grid()

    plt.show()
