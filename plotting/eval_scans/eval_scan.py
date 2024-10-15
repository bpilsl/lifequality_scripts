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


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process ROOT files and extract data.')

    # Required positional argument that takes an arbitrary number of ROOT file paths (results in a list)
    parser.add_argument('root_path_pattern', type=str, nargs='+', help='Path(s) to ROOT files.')

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


def roots2Df(path, config):
    root_file_list = sorted(glob.glob(f'{path}/*.root'))
    keys_to_extract = config['keys_to_extract']
    x_regex = config['x_regex']

    results = {"File": [], "Key": [], "Name": [], "xVal": [], "Mean": [], "StdDev": [], "StdErr": [], "N": [], "Origin": []}

    for root_file in root_file_list:
        with uproot.open(root_file) as file:
            for key_info in keys_to_extract:
                tkey = None
                mean_val = 0

                if 'eTotalEfficiency' in key_info['key']:
                    eff_value, eff_error_low, eff_error_up = extractEfficiency(root_file, key_info['key'])
                    mean_val = eff_value
                else:
                    try:
                            tkey = file[key_info['key']]
                    except Exception as e:
                            print(e)
                            continue

                std_dev_val = 0
                N = 0
                if 'residuals' in key_info['key']:
                    mean_val = extractRMSForRresiduals(tkey)
                    N = 1
                elif isinstance(tkey, uproot.behaviors.TH1.TH1):
                    hist_np = tkey.to_numpy()
                    unjagged_bins = (hist_np[1][:-1] + hist_np[1][1:]) / 2

                    N = np.sum(hist_np[0])
                    mean_val = np.sum(hist_np[0] * unjagged_bins) / N
                    std_dev_val = np.sqrt(np.sum(hist_np[0] * (unjagged_bins - mean_val) ** 2) / N)

                # Extract values for results
                results["xVal"].append(float(re.search(x_regex, root_file).group(1)))
                results["Mean"].append(mean_val)
                results['StdDev'].append(std_dev_val)
                results['File'].append(root_file)
                results['Key'].append(key_info['key'])
                results['Name'].append(key_info['name'])
                results["N"].append(N)
                results["StdErr"].append(std_dev_val / np.sqrt(N))
                results["Origin"].append(path)

    df = pd.DataFrame(results)
    return df


def main():
    args = parse_arguments()

    config_file = args.config_file

    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)  # Load YAML file

    keys_to_extract = config['keys_to_extract']
    x_name = config['x_name']
    # x_regex = config['x_regex']
    do_annotate = config.get('do_annotate', False)
    output_img = config.get('output_img', None)
    if args.output_img:
        output_img = args.output_img
        print('Overwriting output image to ', output_img)

    output_csv = config.get('output_csv', None)
    if args.output_csv:
        output_img = args.output_csv
        print('Overwriting output CSV to ', output_csv)


    dfs = []
    for rpp in args.root_path_pattern:
        dfs.append(roots2Df(rpp, config))

    # if output_csv:
    #     df.to_csv(output_csv)

    plt.figure(figsize=(15, 10))

    # Plot for each key
    for i, key_info in enumerate(keys_to_extract):
        for df in dfs:
            keyrows = df[df['Key'] == key_info['key']]
            plt.subplot(len(keys_to_extract), 1, i + 1)

            x = keyrows['xVal']
            y = keyrows['Mean']

            yerr = keyrows['StdErr'].values

            sns.lineplot(x=x, y=y, label=df['Origin'][0], marker='o')
            # breakpoint()
            if not 'efficiency' in key_info['key'].lower():
                # plt.errorbar(x, y, yerr=yerr, fmt='.', color='blue', capsize=5)
                pass


            plt.xlabel('')
            plt.ylabel(key_info['name'])
            plt.title(key_info['name'])

            if do_annotate:
                annotate_points(df, key_info['key'])

            plt.grid(True, which='both')

            x_range = key_info.get('x_range', None)
            y_range = key_info.get('y_range', None)

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

