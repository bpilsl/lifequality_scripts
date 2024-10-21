import uproot
import numpy as np
import copy
import matplotlib.pyplot as plt
import sys
import re
import seaborn as sns


def get_histogram_from_root(root_file, row, col, key_format):
    """
    Accesses the histogram for the specified pixel (row, col) in the ROOT file
    and returns it as a NumPy array.

    Args:
    root_file (str): Path to the ROOT file.
    row (int): Row number of the pixel.
    col (int): Column number of the pixel.
    key_format (str): The key format in the ROOT file for each histogram.
                      It should be a string that contains placeholders for the row and column.
                      Example: "histograms/pixel_{row}_{col}"

    Returns:
    tuple: A tuple of (bin_edges, histogram_values)
    """
    # Open the ROOT file using uproot
    with uproot.open(root_file) as file:
        # Generate the histogram key using the provided format
        hist_key = key_format.format(row=row, col=col)

        # Access the histogram
        if hist_key not in file:
            raise KeyError(f"Histogram with key '{hist_key}' not found in the ROOT file.")

        hist = file[hist_key]

        # Convert the histogram to a NumPy array
        hist_values = hist.values()
        bin_edges = hist.axes[0].edges()

    return bin_edges, hist_values


def compute_stats(bin_edges, hist_values):
    """
    Compute the mean and standard deviation from the histogram values.

    Args:
    bin_edges (array): Bin edges of the histogram.
    hist_values (array): Values of the histogram.

    Returns:
    tuple: (mean, std_dev)
    """
    # Compute the center of each bin
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Compute the mean and standard deviation using the histogram values
    mean = np.average(bin_centers, weights=hist_values)
    variance = np.average((bin_centers - mean) ** 2, weights=hist_values)
    std_dev = np.sqrt(variance)

    return mean, std_dev


def generate_pixel_stats_from_corry(root_file, key_format):
    """
    Generates a list of mean and standard deviation values for all 64x64 pixels.

    Args:
    root_file (str): Path to the ROOT file.
    key_format (str): The key format in the ROOT file for each histogram.

    Returns:
    list: A list of tuples, where each tuple contains (mean, std_dev, row, col).
    """
    pixel_stats = []

    # Loop over all rows and columns (64x64 grid)
    for row in range(64):
        for col in range(64):
            # Get histogram for the pixel
            bin_edges, hist_values = get_histogram_from_root(root_file, row, col, key_format)

            # Compute the statistics (mean, std_dev)
            mean, std_dev = compute_stats(bin_edges, hist_values)
            # print(f'pix {col}:{row} got {mean} {std_dev}')

            # Append the results to the list
            pixel_stats.append((row, col, mean, std_dev, (hist_values, bin_edges)))

    return pixel_stats

def generate_pixel_stats_from_peary(file):
    pixel_stats = []
    cnt = 0
    stats_curr_pix = {'pix': None, 'vinj': [], 'tot': []}
    with open(file) as f:
        for line in f:
            line = line.strip()
            pixel_match = re.search(r'Pixel: (\d+):(\d+)', line)
            if pixel_match:
                if len(stats_curr_pix['vinj']) > 0:
                    stats_curr_pix['vinj'] = np.array(stats_curr_pix['vinj']) # we are in a new line with a new pixel,
                    #store the data of the pixel we just processed
                    pixel_stats.append(stats_curr_pix)
                    cnt += 1
                currPix = (int(pixel_match.group(1)), int(pixel_match.group(2)))
                stats_curr_pix = {'pix': currPix, 'vinj': [], 'totMean': [], 'totStdDev': [], 'hist': []}
                continue
            if line.startswith('#') or len(line) == 0:
                continue
            if line.startswith('VNFB'):
                vnfb = int(line.split('=')[1])
                continue

            values = line.split(' ')
            inj_volt = float(values[0].replace(':', ''))
            ToTs = np.array(values[1:]).astype(int)
            stats_curr_pix['vinj'].append(inj_volt)
            if len(ToTs) > 0:
                mean = np.mean(ToTs)
                std = np.std(ToTs)
                # print(f'pix {currPix} got {mean} {std}')
                stats_curr_pix['totMean'].append(mean)
                stats_curr_pix['totStdDev'].append(std)
                hist = np.histogram(ToTs, range=(0,256), bins=256)
                print(hist)
                stats_curr_pix['hist'].append(hist)
            else:
                stats_curr_pix['totMean'].append(0)
                stats_curr_pix['totStdDev'].append(0)
                stats_curr_pix['hist'].append([])

    if len(stats_curr_pix['vinj']) > 0:
        # print('last ', stats_curr_pix['pix'])
        stats_curr_pix['vinj'] = np.array(stats_curr_pix['vinj'])
        pixel_stats.append(stats_curr_pix)
        cnt += 1

    return pixel_stats


def compare_pixel_stats(corry, peary):
    bestmatch = {'pix': [], 'pearyTot': [], 'corryTot': [], 'injVolt':[], 'histPeary': [], 'histCorry': []}
    for i, (row, col, mean, std, corryHist) in enumerate(corry):
        key = (row, col)
        if peary.get(key, None) is not None:
            stats = peary[key]
            best_injv = -1
            hist = ([], [])
            min_diff = float('inf')
            best_match_info = None
            for stat in stats:
                tot = stat[1]
                injv = stat[0]
                diff = abs(tot - mean)
                if diff < min_diff:
                    min_diff = diff
                    best_injv = injv
                    bestPearyTot = tot
                    pearyHist = stat[-1]
            bestmatch['pix'].append(key)
            bestmatch['pearyTot'].append(bestPearyTot)
            bestmatch['corryTot'].append(mean)
            bestmatch['injVolt'].append(best_injv)
            bestmatch['histPeary'].append(pearyHist)
            bestmatch['histCorry'].append(corryHist)

    return bestmatch

def plot_best_histograms(best, index, xlim=(0,256)):
    i = index
    histLab = best['histPeary'][i]
    histTb = best['histCorry'][i]
    pix = best['pix'][i]
    injVolt = best['injVolt'][i]
    counts = histLab[0]
    bins = histLab[1]
    # plt.hist(histLab[0], bins=histLab[1], label='ToT Lab injections')
    plt.hist(bins[:-1], bins, weights=counts, label='Lab')
    counts = histTb[0]
    bins = histTb[1]
    plt.hist(bins[:-1], bins, weights=counts, label='TB')
    plt.xlim(xlim)
    plt.title(f'Pixel {pix} at $V_{"{inj}"}$ = {injVolt}V')
    plt.legend()
    plt.show()


def transform_peary(peary):
    transformed_dict = {}

    for stats in peary:
        pix_key = (stats['pix'][0], stats['pix'][1])  # Use the pixel coordinates as the key
        if pix_key not in transformed_dict:
            transformed_dict[pix_key] = []  # Initialize a list for this pixel key

        for i, mean in enumerate(stats['totMean']):
            v_inj = stats['vinj'][i]
            std = stats['totStdDev'][i]
            hist = stats['hist'][i]

            # Append the (mean, std, v_inj, hist_copy) tuple to the list for this pixel
            transformed_dict[pix_key].append((v_inj, mean, std, hist))

    return transformed_dict


def bestToCap(bestmatch, mipQ=24e3):
    capMap = np.zeros((64, 64))
    qInCoulomb = mipQ * 1.602e-19
    for i, pix in enumerate(bestmatch['pix']):
        c = qInCoulomb / bestmatch['injVolt'][i]
        capMap[pix[0], pix[1]] = c

    return capMap


def exportCapMap(map, file):
    with open(file, 'w') as f:
        f.write('''#This file contains the calibrated capacitance values for each pixel of the RD50-MPW4.
#Each line contains the value in the unit of F for one pixel.
#The pixels are specified at the start of the line with <row>:<column>\n\n''')

        for row in range(len(map)):
            for col in range(len(map[row])):
                f.write(f'{row}:{col} {map[row][col]}' + '\n')

        print('successfully stored calibrated cap values to ', file)

root_file = '/home/bernhard/diss/mpw4/tb/desy_oct24/analysis/0E0/hv/run000490_hv-280_vnfbnan.root'
peary_calib_file = '/home/bernhard/diss/mpw4/lab/topside_w8/bias200V/tot_calib/tot_calib-vnfb18-nominal.txt'
destination_file = '/home/bernhard/diss/mpw4/tb/desy_oct24/analysis/cap_map.txt'
corry = generate_pixel_stats_from_corry(root_file, 'EventLoaderEUDAQ2/RD50_MPWx_0/pixelCharges/col-{col}_row-{row}')
peary = generate_pixel_stats_from_peary(peary_calib_file)
peary = transform_peary(peary)

best = compare_pixel_stats(corry, peary)
capMap = bestToCap(best)
sns.heatmap(capMap)
exportCapMap(capMap, destination_file)
plot_best_histograms(best, 3500, (0,30))