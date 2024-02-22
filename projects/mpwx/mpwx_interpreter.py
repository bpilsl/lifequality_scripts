import pandas as pd
import seaborn as sns
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit
from scipy.optimize import OptimizeWarning

sensor_dim = (64, 64)


def sigmoid(x, L, x0, k, b):
    y = L / (1 + np.exp(-k * (x - x0))) + b
    return y


def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / stddev) ** 2 / 2)


def inverseSigmoid(y, L, x0, k, b):
    return - 1 / k * np.log(L / (y - b) - 1) + x0

def v_to_q(v):
    return v * 2.8e-15 / 1.6e-19

def q_to_v(q):
    return q * 1.6e-19 / 2.8e-15

def interpretScurve(data, **kwargs):
    defaultKwargs = {'doPlot': True, 'figure': None, 'xAxisLabel': 'Injection Voltage [mV]', 'yAxisLabel': 'Hits',
                     'title': 'S-curve scan'}
    kwargs = {**defaultKwargs, **kwargs}
    doPlot = kwargs['doPlot']
    retval = {'sigmoidFit': [], 'halfWayGaussFit': None, 'noiseGaussFit': None, 'vt50Map': None}

    vt50_map = np.zeros(sensor_dim)
    noise_map = np.zeros(sensor_dim)
    if doPlot:
        if not kwargs['figure']:
            fig = plt.figure()
        else:
            fig = kwargs['figure']
        gs = GridSpec(2, 3, figure=fig)
        ax1 = fig.add_subplot(gs[:, 0])
        ax1.set(title=kwargs['title'], xlabel=kwargs['xAxisLabel'],
                ylabel='Number of Hits')
        # ax1.legend()
        ax1.grid(True)
    for pixel, group in data.groupby('Pixel'):
        x_data = np.array(group['Voltage'])
        y_data = np.array(group['Hits'])
        if doPlot:
            ax1.scatter(x_data, y_data, marker='.', label=f"Pixel {pixel}")
            plt.xlim(min(x_data), max(x_data))
        try:
            p0 = [max(y_data), np.median(x_data), 1, min(y_data)]  # this is a mandatory initial guess
            popt, _ = curve_fit(sigmoid, x_data, y_data, p0, maxfev=100000)
            retval['sigmoidFit'].append(popt)
            # of fit
            index = pixel.strip().split(':')
            row = int(index[0])
            col = int(index[1])
            vt50_map[row, col] = inverseSigmoid(50, *popt)
            noise_map[row, col] = abs(inverseSigmoid(84, *popt) - inverseSigmoid(16, *popt))

            if doPlot:
                x_fit = np.linspace(min(x_data), max(x_data), 200)  # generate points for plot
                ax1.plot(x_fit, sigmoid(x_fit, *popt), label='fit')
        except RuntimeWarning as rtw:
            print(rtw)
        except RuntimeError as rte:
            print(rte)
        except OptimizeWarning as ow:
            print('optimization failed for', pixel, ow)

    no_nan = vt50_map.flatten()[~np.isnan(vt50_map.flatten())]
    retval['vt50Map']  = vt50_map
    counts, bins = np.histogram(no_nan, bins=100)

    bins = bins[1:]
    counts = counts[1:]  # 0th bin contains fit fails, or not scanned pixels

    # Calculate bin centers
    bin_centers = (bins[:-1] + bins[1:]) / 2

    if doPlot:
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.set(title='$V_{inj, 50}$ map', xlabel='Col', ylabel='Row')
        sns.heatmap(vt50_map)
        ax2.invert_yaxis()
        ax3 = fig.add_subplot(gs[0, 1])
        ax3.set(title='$V_{inj, 50}$ histogram', xlabel='$V_{inj, 50}$ [mV]', ylabel='Counts')

        ax3.hist(bins[:-1], bins, weights=counts)

    try:
        # Use curve_fit to fit the Gaussian function to the histogram data
        initial_guess = [1.0, max(no_nan), np.std(no_nan) + .2]
        params, covariance = curve_fit(gaussian, bin_centers, counts, p0=initial_guess)
        amplitude, mean, stddev = params
        stddev = abs(stddev)
        mean_error = stddev / np.sqrt(sum(counts))
        retval['halfWayGaussFit'] = (mean, stddev, mean_error)
        if doPlot:
            x_fit = np.linspace(min(x_data), max(x_data), 1000)
            ax3.plot(x_fit, gaussian(x_fit, amplitude, mean, stddev), '--', label='Fit', color='black')

            # box with statistics
            props = dict(boxstyle='round', facecolor='wheat', alpha=.5)
            stats = f'$\\mu = {mean:.1f}mV \\approx {v_to_q(mean * 1e-3):.0f} e^-$\n$\\sigma = {stddev:.1f}mV \\approx {v_to_q(stddev * 1e-3):.0f} e^-$'
            ax3.text(0.65, 0.95, stats, transform=ax3.transAxes, fontsize=15,
                     verticalalignment='top', bbox=props)
            ax3.grid()
            plt.xlim(min(x_data), max(x_data))
    except Exception as ex:
        print('error fitting gaussian to VT50: ', ex)

    # NOISE distribution
    no_nan = noise_map.flatten()[~np.isnan(noise_map.flatten())]

    counts, bins = np.histogram(no_nan, bins=100)

    bins = bins[1:]
    counts = counts[1:]  # 0th bin contains fit fails, or not scanned pixels
    # Calculate bin centers
    bin_centers = (bins[:-1] + bins[1:]) / 2
    if doPlot:
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.set(title='Noise map', xlabel='Col', ylabel='Row')
        sns.heatmap(noise_map)
        ax4.invert_yaxis()
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.set(title='Noise histogram', xlabel='Noise [mV]', ylabel='Counts')
        ax5.hist(bins[:-1], bins, weights=counts)
    try:
        initial_guess = [1.0, bin_centers[counts.argmax()], np.std(no_nan)]
        params, covariance = curve_fit(gaussian, bin_centers, counts, p0=initial_guess)
        amplitude, mean, stddev = params
        stddev = abs(stddev)
        mean_error = stddev / np.sqrt(sum(counts))
        retval['noiseGaussFit'] = (mean, stddev, mean_error)
        amplitude, mean, stddev = params
        stddev = abs(stddev)
        if doPlot:
            x_fit = np.linspace(0, 300, 1000)
            ax5.plot(x_fit, gaussian(x_fit, amplitude, mean, stddev), '--', label='Fit', color='black')

            # box with statistics
            props = dict(boxstyle='round', facecolor='wheat', alpha=.5)
            stats = f'$\\mu$ = {mean:.1f}mV\n$\\sigma$ = {stddev:.1f}mV'
            ax5.text(0.75, 0.95, stats, transform=ax5.transAxes, fontsize=15,
                     verticalalignment='top', bbox=props)
            ax5.grid()
    except Exception as ex:
        print('error fitting gaussian to noise: ', ex)
    return retval


def readScurveData(file):
    pixel_data = {'Pixel':  [], 'Voltage': [], 'Hits': [], 'Index': []}
    current_pixel = {}

    with open(file, 'r') as f:
        for line in f:
            if line.startswith('#! Pixel'):
                if current_pixel:
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
                    pixel_data['Pixel'].append(current_pixel['Pixel'])
                    pixel_data['Index'].append(current_pixel['Index'])
                    pixel_data['Voltage'].append(float(voltage) * 1e3)  #convert to mV
                    pixel_data['Hits'].append(int(hits))
                except:
                    print('broken line ', line)
                    continue

    return pd.DataFrame(pixel_data)


def readHitmap(file):
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
    return hitmap


def plotHitmap(data, **kwargs):
    defaultKwargs = {'plotHist': False, 'figure': None, 'xAxisLabelMap': 'column', 'yAxisLabelMap': 'row',
                     'titleMap': 'Chip-Map', 'xAxisLabelHist': 'TDAC values', 'yAxisLabelHist': 'Counts',
                     'titleHist': 'TDAC usage', 'annotMap': False}
    kwargs = {**defaultKwargs, **kwargs}
    fig = kwargs['figure']
    if not fig:
        fig = plt.figure()
    if kwargs['plotHist']:
        ax1 = plt.subplot(221)
        ax1.set(title=kwargs['titleHist'], xlabel=kwargs['xAxisLabelHist'], ylabel=kwargs['yAxisLabelHist'])
        plt.xlim(0, 15)
        tdac_map = data.flatten()
        tdac_map = tdac_map[tdac_map >= 0]  # get rid of -1
        plt.hist(tdac_map, bins=15)
        ax2 = plt.subplot(222)
    else:
        ax2 = plt.subplot(111)
    ax2.set_xlabel(kwargs['titleMap'])
    ax2.set_ylabel(kwargs['yAxisLabelMap'])
    ax2.set_title(kwargs['xAxisLabelMap'])
    sns.heatmap(data, annot=kwargs['annotMap'])
    ax2.invert_yaxis()


def readSpectrum(file):
    meanMap = np.zeros(sensor_dim)
    accumulated_hist = np.zeros(256)
    with open(file) as f:
        for line in f:
            if line.startswith('#'):
                continue
            match = re.match(r'(\d+):(\d+)(.+)', line)
            if not match:
                continue
            row = int(match.group(1))
            col = int(match.group(2))
            data = match.group(3)

            counts = np.array(data.split(), dtype=int)
            accumulated_hist += counts
            bins = np.arange(256)

            if counts[100:256].any():
                print(f'big ToT for pixel {row}:{col}')

            mean = np.average(bins, weights=counts)
            var = np.average((bins - mean) ** 2, weights=counts)
            std_dev = np.sqrt(var)
            meanMap[row][col] = mean


def getPowerReport(file):
    inBiasSection = False
    powerInfo = []
    currSection = {}
    with open(file) as f:
        for line in f:
            if line.startswith('#Power consumption'):
                inBiasSection = True
                continue
            if line.startswith('#! Pixel'):
                break
            if not inBiasSection:
                continue
            dacMatch = re.search(r'#(.+):', line)
            if dacMatch:
                if len(currSection.keys()) > 0:
                    powerInfo.append(currSection)
                    currSection = {}
                currSection['name'] = dacMatch.group(1)
                continue
            measurementMatch = re.search(r'#*(.) = (\d+\.*\d*)', line)
            if measurementMatch:
                currSection[measurementMatch.group(1)] = float(measurementMatch.group(2))

    return powerInfo


def parse_tdac(file):
    map = np.zeros(sensor_dim)
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

                tdac = int(splitted[col])
                map['tdac'][row][col - 1] = tdac
    return map





if __name__ == '__main__':
    pass
