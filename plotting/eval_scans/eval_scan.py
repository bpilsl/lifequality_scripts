import os.path
from csv import excel

import ROOT
import uproot
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import argparse
import yaml
import matplotlib.ticker as ticker

config = None


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process ROOT files and extract data.')

    # Required positional argument that takes an arbitrary number of ROOT file paths (results in a list)
    parser.add_argument('root_path_pattern', type=str, nargs='*', help='Path(s) to ROOT files.', default=None)

    # Required config file argument specified with -c
    parser.add_argument('-c', '--config_file', type=str, required=True,
                        help='Path to the configuration file specifying keys to extract.')

    # Optional output image (PNG) argument
    parser.add_argument('--png', dest='output_img', type=str, default=None,
                        help='Path to save the output image (PNG format).')

    # Optional output CSV argument
    parser.add_argument('--csv', dest='output_csv', type=str, default=None,
                        help='Path to save the output data (CSV format).')

    return parser.parse_args()

def annotate_points(df, col_name, color=None):
    for index, row in df.iterrows():
        if col_name != row['Key']:
            continue
        x = row['xVal']
        y = row['Mean']
        val = f'{row["Mean"]:.3f}'

        plt.annotate(val, (x, y), textcoords="offset points", xytext=(10, 10), ha='center', fontsize=10, color=color)

def extractRMSForResiduals(hist, quantile=0.5, plot=False):
    # Extract bin contents (residuals) and bin edges
    bin_contents = hist.values()
    bin_edges = hist.axis().edges()

    # Reconstruct the distribution
    # To avoid type issues, use list comprehension and avoid np.repeat
    values = np.concatenate([
        np.full(int(count), (bin_edges[i] + bin_edges[i + 1]) / 2)
        for i, count in enumerate(bin_contents)
    ])

    # Determine the quantiles    with open(config_file, 'r') as file:
    lower_quantile = quantile
    upper_quantile = 100 - quantile
    lower_q_value = np.percentile(values, lower_quantile)
    upper_q_value = np.percentile(values, upper_quantile)

    # Filter residuals within the quantiles
    truncated_values = values[(values >= lower_q_value) & (values <= upper_q_value)]
    # Calculate RMS
    truncated_rms = np.std(truncated_values)

    N = np.sum(bin_contents)

    std_err = truncated_rms / np.sqrt(2* (N - 1))

    if plot:

        print(f'lower q ={lower_q_value} for {lower_quantile}th Percentile')
        print(f'upper q ={upper_q_value} for {upper_quantile}th Percetnile')
        print(f'Truncated RMS: {truncated_rms}')
        # Plot the histogram and quantiles
        plt.figure(figsize=(10, 10))
        plt.hist(values, bins=bin_edges, alpha=0.7, label='Residuals', edgecolor='black', color='black')

        # Add lines for quantiles
        plt.axvline(x=lower_q_value, color='r', linestyle='--', linewidth=2, label=f'{lower_quantile}th Percentile')
        plt.axvline(x=upper_q_value, color='g', linestyle='--', linewidth=2, label=f'{upper_quantile}th Percentile')

        plt.xlim((lower_q_value * 2, upper_q_value * 2))
        plt.title(f'Spatial residuals')
        plt.xlabel(r'$x_{Track} - x_{DUT}$ [$\mu m$]')
        plt.ylabel('# Events')
        plt.legend()
        # plt.xlim(xlims)
        plt.grid(True)
        # plt.savefig(output_plot)
        plt.show()
    return  truncated_rms, std_err

def extractEfficiency(root_file, key):
    file = ROOT.TFile(root_file, 'READ')
    efficiency = file.Get(key)

    if not efficiency:
        print("TEfficiency object not found in the file!")
        return None

    # Access the properties of the TEfficiency object
    if isinstance(efficiency, ROOT.TEfficiency):
            i = 1 # Corry stores effi in first bin
            eff_value = efficiency.GetEfficiency(i)
            eff_error_low = efficiency.GetEfficiencyErrorLow(i)
            eff_error_up = efficiency.GetEfficiencyErrorUp(i)

            # print(f"Bin {i}: Efficiency = {eff_value}, Error Low = {eff_error_low}, Error Up = {eff_error_up}")
    else:
        print("The object is not a TEfficiency object.")
        return  None

    # Close the file
    file.Close()
    return eff_value, eff_error_low, eff_error_up


import uproot
import glob
import os
import re
import numpy as np
import pandas as pd

def roots2Df(path, config):
    root_file_list = sorted(glob.glob(f'{path}/*.root'))
    plots = config['plots']
    x_regex = config['x_regex']

    results = {"File": [], "Key": [], "Name": [], "xVal": [], "Mean": [], "StdDev": [], "StdErr": [], "N": [], "Origin": [], 'errorLow': [], 'errorUp': []}

    for root_file in root_file_list:
        print('processing ', root_file)
        with uproot.open(root_file) as file:
            for plot_info in plots:
                keys = plot_info['keys'] if isinstance(plot_info['keys'], list) else [plot_info['keys']]
                
                for i, key in enumerate(keys):
                    tkey = None
                    mean_val = 0
                    std_dev_val = None
                    error_up = None
                    error_low = None
                    name = plot_info['names'][i]
                    
                    try:
                        if 'eTotalEfficiency' in key:
                            eff_value, error_low, error_up = extractEfficiency(root_file, key)
                            mean_val = eff_value
                        else:
                            tkey = file[key]
                    except Exception as e:
                        print(f'{root_file}: {key} -> {e}')
                        continue
                    
                    std_dev_val = None
                    std_error = None
                    N = 0
                    if 'residuals' in key:
                        mean_val, std_error = extractRMSForResiduals(tkey, plot=False)
                        N = 1
                    elif isinstance(tkey, uproot.behaviors.TH1.TH1):
                        hist_np = tkey.to_numpy()
                        unjagged_bins = (hist_np[1][:-1] + hist_np[1][1:]) / 2
                        
                        N = np.sum(hist_np[0])
                        mean_val = np.sum(hist_np[0] * unjagged_bins) / N
                        print(key, root_file, mean_val)
                        std_dev_val = np.sqrt(np.sum(hist_np[0] * (unjagged_bins - mean_val) ** 2) / N)
                                        
                    if std_dev_val:
                        std_error = std_dev_val / np.sqrt(N)
                    
                    if not error_up:
                        error_up = std_error
                    if not error_low:
                        error_low = std_error
                    
                    results["xVal"].append(float(re.search(x_regex, os.path.basename(root_file)).group(1)))
                    results["Mean"].append(mean_val)
                    results['StdDev'].append(std_dev_val)
                    results['File'].append(root_file)
                    results['Key'].append(key)
                    results['Name'].append(name)
                    results["N"].append(N)
                    results["StdErr"].append(std_error)
                    results['errorLow'].append(error_low)
                    results['errorUp'].append(error_up)
                    results["Origin"].append(path)

    df = pd.DataFrame(results)
    return df


# Define the transformation function
def primary_to_secondary(x):
    coeffs = config["polynomial_coefficients"]
    return np.polyval(coeffs, np.asarray(x))  # Ensure input is an array

# Define an approximate inverse function
def secondary_to_primary(x2):
    coeffs = config["polynomial_coefficients"]

    if len(coeffs) == 2:  # If it's a linear function, invert directly
        a, b = coeffs
        return (np.asarray(x2) - b) / a

    x_values = np.linspace(0, 10, 1000)
    y_values = np.polyval(coeffs, x_values)
    return np.interp(x2, y_values, x_values)  # Interpolate inverse

def main():
    global config  # Add this line
    args = parse_arguments()

    config_file = args.config_file

    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)  # Load YAML file

    plots = config['plots']
    x_name = config['x_name']
    fontsize = config.get('fontsize', None)


    if fontsize is not None:
        plt.rcParams.update({
            'font.size': fontsize,  # Base font size
            'axes.titlesize': fontsize,  # Title size
            'axes.labelsize': fontsize,  # X/Y label size
            'xtick.labelsize': fontsize,  # X tick label size
            'ytick.labelsize': fontsize,  # Y tick label size
            'legend.fontsize': fontsize,  # Legend text size
            'legend.title_fontsize': fontsize  # Legend title size
        })


    do_annotate = config.get('do_annotate', False)
    output_img = config.get('output_img', None)
    if args.output_img:
        output_img = args.output_img
        print('Overwriting output image to ', output_img)

    output_csv = config.get('output_csv', None)
    if args.output_csv:
        output_csv = args.output_csv
        print('Overwriting output CSV to ', output_csv)

    dfs = []

    data = []
    if 'inputs' in config:
        for input in config['inputs']:
            tmp = input
            tmp['data'] = roots2Df(input['path'], config)
            data.append(tmp)
    else:
        for i, rpp in enumerate(args.root_path_pattern):
            tmp = {}
            tmp['linestyle'] = None
            tmp['color'] = None
            tmp['name'] = rpp
            tmp['data'] = roots2Df(rpp, config)
            data.append(tmp)

    for d in data:
        csv = d.get('output_csv', None)
        if csv:
            d['data'].to_csv(csv)

    size = config.get('figsize', None)
    if size:
        size = (size[0], size[1])
    fig = plt.figure(figsize=size)
    show_legend = config.get('plot_legend', True)

    # Plot for each key
    for i, plot_info in enumerate(plots):
        ax1 = fig.add_subplot(len(plots), 1, i + 1)
        for d in data:
            df = d['data']
            keys = plot_info['keys'] if isinstance(plot_info['keys'], list) else [plot_info['keys']]
                        

            if "x2label" in config and 'polynomial_coefficients' in config:
                ax2 = ax1.secondary_xaxis("top", functions=(primary_to_secondary, secondary_to_primary))
                ax2.set_xlabel(config["x2label"])
            for j, key in enumerate(keys):                   
                ax = ax1
                if len(keys) > 1:
                    colors = sns.color_palette('colorblind')
                    grid_styles = ['-', '--', '-.', ':', '.']
                    grid_style = grid_styles[j % len(grid_styles)]
                    color = colors[j]
                    if j > 0:
                        ax = ax1.twinx()
                else:
                    color = None
                    grid_style = '-'

                keyrows = df[df['Key'] == key]
                print(keyrows)

                x = keyrows['xVal']
                y = keyrows['Mean']

                yscale = plot_info.get('yscale', None)
                if yscale:
                    y *= yscale
                

                if 'color' in plot_info:
                    color = plot_info['color']
                elif 'color' in d:
                    color = d['color']

                linestyle = plot_info.get('linestyle', None)
                linewidth = plot_info.get('linewidth', None)

                markersize = plot_info.get('markersize', None)
                if not markersize :
                    markersize = d.get('markersize', None)

                if not linestyle:
                    linestyle = d.get('linestyle', None)

                yerr = (keyrows['errorLow'].values, keyrows['errorUp'].values)            
                legend = 'auto' if show_legend else None

                marker = d.get('marker', 'o')

                plot = sns.lineplot(x=x, y=y, label=d['name'], markersize=markersize, marker=marker, ax=ax, color=color, legend=legend, linewidth=linewidth, linestyle=linestyle)                
                color = plot.get_lines()[-1].get_color() # get used color to also use it for error bars
                if config.get('color_yaxis', False):
                    ax.yaxis.label.set_color(color)
                    ax.tick_params(axis='y', colors=color)
                    ax.spines['left'].set_color(color)
                if config.get('plot_errorbars', True):
                    plt.errorbar(x, y, yerr=yerr, fmt='.', capsize=5, color=color)

                # breakpoint()
                ax.set_ylabel(keyrows['Name'].iloc[0])
                ax.set_title(config.get('title'), None)

                if do_annotate:
                    annotate_points(df, key, color=color)

                plt.grid(True, which='both', linestyle=grid_style)
                x_range = plot_info.get('x_range', None)
                y_range = plot_info.get('y_range', None)


                if x_range is not None:
                    plt.xlim(x_range)

                if y_range is not None:
                    plt.ylim(y_range)

                ax = plt.gca()
                if config.get('log_x', False):
                    ax.set_xscale('log')
                    ax.xaxis.set_major_formatter(ticker.LogFormatterSciNotation())
                    # ax.tick_params(axis='x', which='major', labelsize=10)
                if config.get('log_y', False):
                    ax.set_yscale('log')
                    ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
                    # ax.tick_params(axis='y', which='major', labelsize=10)

                # ax.set_xticks(ax.get_xticks())
                # ax.set_xticklabels(ax.get_xticks())
                ax.set_xlabel(x_name)       


    plt.tight_layout()
    if output_img:
        plt.savefig(output_img)
    plt.show()


if __name__ == "__main__":
    main()

