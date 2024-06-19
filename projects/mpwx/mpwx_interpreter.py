import pandas as pd
import seaborn as sns
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit
from scipy.optimize import OptimizeWarning

sensor_dim = (64, 64)
font_small = 15
font_large = 20


def sigmoid(x, L, x0, k, b):
    y = L / (1 + np.exp(-k * (x - x0))) + b
    return y


def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / stddev) ** 2 / 2)


def inverseSigmoid(y, L, x0, k, b):
    return - 1 / k * np.log(L / (y - b) - 1) + x0


def v_to_q(v):
    #  v should be given in mV
    return v * 1e-3 * 2.8e-15 / 1.6e-19


def q_to_v(q):
    # q passed in units of e-; returns voltage in mV
    return q * 1.6e-19 / 2.8e-15 * 1e3


def interpretScurve(data, **kwargs):
    defaultKwargs = {'doPlot': True, 'figure': None, 'xAxisLabel': 'Injection Voltage [mV]', 'yAxisLabel': 'Hits',
                     'title': 'S-curve scan'}
    kwargs = {**defaultKwargs, **kwargs}
    doPlot = kwargs['doPlot']
    retval = {'sigmoidFit': [], 'halfWayGaussFit': None, 'noiseGaussFit': None, 'vt50Map': None}

    def find_transition_index(arr):
        saturated_values = (min(arr), max(arr))

        # Find indices where array is not equal to the saturated values
        non_saturated_indices = np.where(~np.isin(arr, saturated_values))[0]

        if non_saturated_indices.size < 2:
            # Array is entirely saturated, no transition found
            return None

        # Calculate the gradient of the array within the non-saturated subset
        gradient = np.gradient(arr[non_saturated_indices])

        # Find the index with the maximum absolute gradient
        max_gradient_index = non_saturated_indices[np.argmax(np.abs(gradient))]

        return max_gradient_index

    vt50_map = np.zeros(sensor_dim)
    noise_map = np.zeros(sensor_dim)
    if doPlot:
        if not kwargs['figure']:
            fig = plt.figure()
        else:
            fig = kwargs['figure']
        gs = GridSpec(2, 3, figure=fig)
        ax1 = fig.add_subplot(gs[:, 0])
        # ax1.set(title=kwargs['title'], xlabel=kwargs['xAxisLabel'],
        #         ylabel='Number of Hits', fontsize=font_large)
        ax1.set_title(kwargs['title'], fontsize=font_small)
        ax1.set_xlabel(kwargs['xAxisLabel'], fontsize=font_small)
        ax1.set_ylabel(r'Number of Hits', fontsize=font_small)
        secax_x = ax1.secondary_xaxis('top', functions=(v_to_q, q_to_v))
        secax_x.set_xlabel('Injection Charge [$e^-$]', fontsize=font_small)
        ax1.grid(True)
        ax1.tick_params(axis='x', labelsize=font_small)
        ax1.tick_params(axis='y', labelsize=font_small)
        secax_x.tick_params(axis='x', labelsize=font_small)
    for pixel, group in data.groupby('Pixel'):
        x_data = np.array(group['Voltage'])
        y_data = np.array(group['Hits'])
        if doPlot:
            ax1.scatter(x_data, y_data, marker='.', label=f"Pixel {pixel}")
            plt.xlim(min(x_data), max(x_data))
        try:
            # print('interpreter: ', x_data, y_data)
            x0index = find_transition_index(y_data)
            if not x0index:
                print('Problem finding initial fit params')
                continue
            x00 = x_data[x0index]
            k0 = np.sign(y_data[-1] - y_data[0])
            p0 = [max(y_data), x00, k0, min(y_data)]  # this is a mandatory initial guess
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
            print(rtw, 'at', x_data, '\n', p0, '\n', y_data)
        except RuntimeError as rte:
            print(rte, 'at', x_data, '\n', p0, '\n', y_data)
        except OptimizeWarning as ow:
            print('optimization failed for', pixel, ow)
        except Exception as ex:
            print('random ex: ', ex)

    no_nan = vt50_map.flatten()[~np.isnan(vt50_map.flatten())]
    no_nan = no_nan[~np.isinf(no_nan)]
    retval['vt50Map'] = vt50_map
    # import pdb; pdb.set_trace()
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
        # mu0 = bins[:-1][counts == max(counts)][0]  # bin value at which bin with maximum counts sits
        # initial_guess = [1.0, mu0, np.std(no_nan)]
        stddev0 = np.std(no_nan[no_nan > 0])
        if np.isnan(stddev0):
            stddev0 = 0.0
        initial_guess = [1.0, bin_centers[counts.argmax()], stddev0 + .1]
        # print('vt50 initi', initial_guess)
        params, covariance = curve_fit(gaussian, bin_centers, counts, p0=initial_guess)
        # print('vt50 fitted', params)
        amplitude, mean, stddev = params
        stddev = abs(stddev)
        mean_error = stddev / np.sqrt(sum(counts))
        retval['halfWayGaussFit'] = (mean, stddev, mean_error)
        if doPlot:
            x_fit = np.linspace(min(x_data), max(x_data), 1000)
            ax3.plot(x_fit, gaussian(x_fit, amplitude, mean, stddev), '--', label='Fit', color='black')

            # box with statistics
            props = dict(boxstyle='round', facecolor='wheat', alpha=.5)
            stats = f'$\\mu = {mean:.1f}mV \\approx {v_to_q(mean):.0f} e^-$\n$\\sigma = {stddev:.1f}mV \\approx {v_to_q(stddev):.0f} e^-$'
            ax3.text(0.65, 0.95, stats, transform=ax3.transAxes, fontsize=15,
                     verticalalignment='top', bbox=props)
            ax3.grid()
            plt.xlim(min(x_data), max(x_data))
    except RuntimeWarning as rtw:
        print(rtw, 'at', bin_centers, '\n', initial_guess, '\n', counts)
    except RuntimeError as rte:
        mean = np.average(bin_centers, weights=counts)
        stddev = np.sqrt(np.average((bin_centers - mean) ** 2, weights=counts))
        mean_error = stddev / np.sqrt(sum(counts))
        retval['halfWayGaussFit'] = (mean, stddev, mean_error)
        print(rte)
    except Exception as ex:
        print('error fitting gaussian to VT50: ', ex)

    # NOISE distribution
    no_nan = noise_map.flatten()[~np.isnan(noise_map.flatten())]
    no_nan = no_nan[~np.isinf(no_nan)]

    counts, bins = np.histogram(no_nan, bins=100)

    bins = bins[1:]
    counts = counts[1:]  # 0th bin contains fit fails, or not scanned pixels
    # Calculate bin centers
    bin_centers = (bins[:-1] + bins[1:]) / 2
    if doPlot:
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.set(title='CEN map', xlabel='Col', ylabel='Row')
        sns.heatmap(noise_map)
        ax4.invert_yaxis()
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.set(title='CEN histogram', xlabel='Noise [mV]', ylabel='Counts')
        ax5.hist(bins[:-1], bins, weights=counts)
    try:
        stddev0 = np.std(no_nan[no_nan > 0])
        if np.isnan(stddev0):
            stddev0 = 0.0
        initial_guess = [max(counts), bin_centers[counts.argmax()], stddev0 + .1]
        # initial_guess = [1.0, bin_centers[counts.argmax()], np.std(no_nan) + 0.02]
        params, covariance = curve_fit(gaussian, bin_centers, counts, p0=initial_guess)
        amplitude, mean, stddev = params
        stddev = abs(stddev)
        mean_error = stddev / np.sqrt(sum(counts))
        retval['noiseGaussFit'] = (mean, stddev, mean_error)
        amplitude, mean, stddev = params
        if doPlot:
            x_fit = np.linspace(0, 300, 1000)
            ax5.plot(x_fit, gaussian(x_fit, amplitude, mean, stddev), '--', label='Fit', color='black')

            # box with statistics
            props = dict(boxstyle='round', facecolor='wheat', alpha=.5)
            stats = f'$\\mu = {mean:.1f}mV \\approx {v_to_q(mean):.0f} e^-$\n$\\sigma = {stddev:.1f}mV \\approx {v_to_q(stddev):.0f} e^-$'
            ax5.text(0.75, 0.95, stats, transform=ax5.transAxes, fontsize=15,
                     verticalalignment='top', bbox=props)
            ax5.grid()
    except RuntimeError as rte:
        mean = np.average(bin_centers, weights=counts)
        stddev = np.sqrt(np.average((bin_centers - mean) ** 2, weights=counts))
        mean_error = stddev / np.sqrt(sum(counts))
        retval['noiseGaussFit'] = (mean, stddev, mean_error)
        print(rte)
    except Exception as ex:
        print('error fitting gaussian to noise: ', ex)
    return retval


def readScurveData(file):
    pixel_data = {'Pixel': [], 'Voltage': [], 'Hits': [], 'Index': []}
    current_pixel = {}

    with open(file, 'r') as f:
        for line in f:
            if line.startswith('#! Pixel'):
                if current_pixel:
                    current_pixel = {}
                _, pixel_info = line.split('#! Pixel')
                current_pixel['Pixel'] = pixel_info.strip()
                # if current_pixel['Pixel'] != '0:32' and current_pixel['Pixel'] != '0:33' and current_pixel['Pixel'] != '1:32':
                #     current_pixel = {}
                #     continue
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
                    if len(current_pixel.keys()) == 4:
                        pixel_data['Pixel'].append(current_pixel['Pixel'])
                        pixel_data['Index'].append(current_pixel['Index'])
                        pixel_data['Voltage'].append(float(voltage) * 1e3)  # convert to mV
                        pixel_data['Hits'].append(int(hits))
                except:
                    print('broken line ', line)
                    continue

    return pd.DataFrame(pixel_data)


def readHitmap(file):
    hitmap = np.zeros(sensor_dim)
    t = 0.0

    with open(file, 'r') as f:
        row = 0
        for line in f:
            match = re.search(r't = (\d+)', line)
            if match:
                t = float(match.group(1))
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
    return hitmap, t


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

def plotNoise(data, time):
    freq_data = data / (time * 1e-3)
    ax1 = plt.subplot(121)
    ax1.set(title='Noise-Map', xlabel='col', ylabel='row')
    sns.heatmap(freq_data, annot=False)
    ax1.invert_yaxis()

    ax2 = plt.subplot(122)
    ax2.set(title='Noise histogram', xlabel='f / Hz', ylabel='Counts')
    freq_data = freq_data.flatten()

    counts, bins = np.histogram(freq_data[freq_data > 0], bins=50)
    ax2.hist(bins[:-1], bins, weights=counts)
    total_freq = sum(freq_data[freq_data > 0])

    ax2.text(max(bins) * .5, max(counts), f'max. f $\\approx {total_freq:.2f}$ Hz')


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

            # if counts[100:256].any():
            #     print(f'big ToT for pixel {row}:{col}')

            mean = np.average(bins, weights=counts)
            var = np.average((bins - mean) ** 2, weights=counts)
            std_dev = np.sqrt(var)
            meanMap[row][col] = mean
        return accumulated_hist, meanMap


def plotSpectrum(accumulated_hist, meanMap):
    non_zero_indices = accumulated_hist > 0
    bins = np.arange(256)
    non_zero_bins = bins[non_zero_indices]
    non_zero_counts = accumulated_hist[non_zero_indices]

	# Find the index of the bin with the highest frequency
    max_freq_index = np.argmax(non_zero_counts)
    bin_centers = (bins[:-1] + bins[1:]) / 2

# Calculate the MPV by taking the center of the bin with the highest frequency
    mpv = bin_centers[max_freq_index]
    mean = np.average(bins, weights=accumulated_hist)
    var = np.average((bins - mean) ** 2, weights=accumulated_hist)
    std_dev = np.sqrt(var)
    multiplePixel = len(meanMap[meanMap > 0]) > 1
    if multiplePixel:
        ax1 = plt.subplot(121)
    else:
        ax1 = plt.subplot()
    ax1.bar(non_zero_bins, non_zero_counts, width=1.0, align='edge', edgecolor='black')

    props = dict(boxstyle='round', facecolor='wheat', alpha=.5)
    stats = f'$\\mu$ = {mean:.1f}LSB\n$\\sigma$ = {std_dev:.1f}LSB\nMPV = {mpv}LSB'
    ax1.text(0.75, 0.95, stats, transform=ax1.transAxes, fontsize=15,
             verticalalignment='top', bbox=props)

    ax1.set_title('Accumulated ToT Histogram', fontsize=30)

    ax1.set_xlabel('ToT (LSB)', fontsize=40)
    ax1.set_ylabel('Counts', fontsize=40)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.xlim(0, 250)
    # plt.ylim(0, 5)

    if multiplePixel:
        ax2 = plt.subplot(122)
        sns.heatmap(meanMap, ax=ax2)
        ax2.invert_yaxis()
        ax2.set_title('$\mu_{TOT}$ Map')


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

    df = pd.DataFrame(powerInfo)
    powerInfo.append({'name': 'Total', 'I': sum(df['I']), 'P': sum(df['P']), 'U': np.nan})
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
