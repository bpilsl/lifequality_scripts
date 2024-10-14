import uproot
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import argparse
import yaml


def parse_arguments():
        parser = argparse.ArgumentParser(description='Process ROOT files and extract data.')

        # Required positional argument
        parser.add_argument('root_path_pattern', type=str, help='Path to ROOT files (e.g., ".")')

        # Optional config file argument
        parser.add_argument('config_file', type=str, nargs='?', default=None,
                            help='Path to the configuration file specifying keys to extract (optional). If not '
                                 'specified yml in "root_path_pattern" will be globed')

        return parser.parse_args()

def annotate_points(df, col_name):
    for index, row in df.iterrows():
        if col_name != row['Key']:
            continue
        x = row['xVal']
        y = row['Mean']
        val = f'{row["Mean"]:.3f}'

        plt.annotate(val, (x, y), textcoords="offset points", xytext=(15, 15), ha='center', fontsize=8)

def extractRMSForRresiduals(hist, quantile=0.5, plot=False):
    # Extract bin contents (residuals) and bin edges
    bin_contents = hist.values()
    bin_edges = hist.axis().edges()

    # Reconstruct the distribution
    # To avoid type issues, use list comprehension and avoid np.repeat
    values = np.concatenate([
        np.full(int(count), (bin_edges[i] + bin_edges[i + 1]) / 2)
        for i, count in enumerate(bin_contents)
    ])

    # Determine the quantiles
    lower_quantile = quantile
    upper_quantile = 100 - quantile
    lower_q_value = np.percentile(values, lower_quantile)
    upper_q_value = np.percentile(values, upper_quantile)

    # Filter residuals within the quantiles
    truncated_values = values[(values >= lower_q_value) & (values <= upper_q_value)]
    # Calculate RMS
    truncated_rms = np.std(truncated_values)

    if plot:

        print(f'lower q ={lower_q_value} for {lower_quantile}th Percentile')
        print(f'upper q ={upper_q_value} for {upper_quantile}th Percetnile')
        print(f'Truncated RMS: {truncated_rms}')
        # Plot the histogram and quantiles
        plt.figure(figsize=(10, 6))
        plt.hist(values, bins=bin_edges, alpha=0.7, label='Residuals', edgecolor='black')

        # Add lines for quantiles
        plt.axvline(x=lower_q_value, color='r', linestyle='--', linewidth=2, label=f'{lower_quantile}th Percentile')
        plt.axvline(x=upper_q_value, color='g', linestyle='--', linewidth=2, label=f'{upper_quantile}th Percentile')

        plt.title(f'Histogram of "bla" with Quantiles')
        plt.xlabel('Residual Value')
        plt.ylabel('Frequency')
        plt.legend()
        # plt.xlim(xlims)
        plt.grid(True)
        # plt.savefig(output_plot)
        plt.show()
    return  truncated_rms

def main():
    args = parse_arguments()
    
    root_file_list = sorted(glob.glob(f'{args.root_path_pattern}/*.root'))
    config_file = args.config_file
    if not config_file:
        config_file = glob.glob(f'{args.root_path_pattern}/*.yml')[0]

    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)  # Load YAML file

    keys_to_extract = config['keys_to_extract']
    x_name = config['x_name']
    x_regex = config['x_regex']
    do_annotate = config.get('do_annotate', False)
    output_img = config.get('output_img', None)
    output_csv = config.get('output_csv', None)

    
    results = {"File": [], "Key": [], "Name": [], "xVal": [], "Mean": [], "StdDev": [], "StdErr": [], "N": []}

    for root_file in root_file_list:
        with uproot.open(root_file) as file:
            for key_info in keys_to_extract:
                try:
                    tkey = file[key_info['key']]
                    std_dev_val = 0
                    N = 0
                    if 'residuals' in key_info['key']:
                        mean_val = extractRMSForRresiduals(tkey)
                        N = 1
                    elif isinstance(tkey, uproot.behaviors.TH1.TH1):
                        hist_np = tkey.to_numpy()
                        unjagged_bins = (hist_np[1][:-1] + hist_np[1][1:]) / 2
                        # breakpoint()
                        if 'efficiency' in key_info['key'].lower():
                            #unjagging to bin center leads in the TH1D efficiency histograms of Corry to the last bin
                            #being at 1.0025 and therefore >100%. You get the problem.
                            #Fix the last bin to 1.0
                            unjagged_bins[-1] = 1.0


                        N = np.sum(hist_np[0])
                        mean_val = np.sum(hist_np[0] * unjagged_bins) / N
                        std_dev_val = np.sqrt(np.sum(hist_np[0] * (unjagged_bins - mean_val) ** 2) / N)

                    elif 'efficiency' in key_info['key'].lower() and isinstance(tkey,
                                                                                uproot.behaviors.TProfile2D.TProfile2D):
                        vals = tkey.values()
                        # efficiency profile is embedded in 'ring' of 0 (edges not taken into account)
                        vals = vals[1:-1]  # remove upper and lower 0 band
                        vals = vals[:, 1:-1]  # remove left and right 0 band
                        mean_val = np.average(vals) * 100
                        std_dev_val = np.std(vals) * 100
                        N = 1
                    else:
                        continue

                    # Extract values for results
                    results["xVal"].append(float(re.search(x_regex, root_file).group(1)))
                    results["Mean"].append(mean_val)
                    results['StdDev'].append(std_dev_val)
                    results['File'].append(root_file)
                    results['Key'].append(key_info['key'])
                    results['Name'].append(key_info['name'])
                    results["N"].append(N)
                    results["StdErr"].append(std_dev_val / np.sqrt(N))

                except KeyError:
                    print(f"Key '{key_info['key']}' not found in file '{root_file}'")

    df = pd.DataFrame(results)

    if output_csv:
        df.to_csv(output_csv)

    plt.figure(figsize=(15, 10))

    # Plot for each key
    for i, key_info in enumerate(keys_to_extract):
        keyrows = df[df['Key'] == key_info['key']]
        plt.subplot(len(keys_to_extract), 1, i + 1)

        x = keyrows['xVal']
        y = keyrows['Mean']

        yerr = keyrows['StdErr'].values

        sns.lineplot(x=x, y=y, label='Data', marker='o')
        # breakpoint()
        if not 'efficiency' in key_info['key'].lower():
                pass
            # plt.errorbar(x, y, yerr=yerr, fmt='.', color='blue', capsize=5)

        plt.xlabel('')
        plt.ylabel(key_info['name'])
        plt.title(key_info['name'])

        if do_annotate:
            annotate_points(df, key_info['key'])

        plt.grid(True, which='both')

        # Setting x_range and y_range for individual plots, if specified in the key_info
        x_range = key_info.get('x_range', None)  # Check if 'x_range' exists in key_info
        y_range = key_info.get('y_range', None)  # Check if 'y_range' exists in key_info

        if x_range is not None:
            plt.xlim(x_range)

        if y_range is not None:
            plt.ylim(y_range)

        ax = plt.gca()
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(ax.get_xticks())

    plt.xlabel(x_name)
    plt.tight_layout()
    if output_img:
        plt.savefig(output_img)
    plt.show()


if __name__ == "__main__":
    main()

