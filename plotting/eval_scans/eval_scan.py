import uproot
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import argparse
import json

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process ROOT files and extract data.')
    parser.add_argument('root_path_pattern', type=str, help='Path pattern to match ROOT files (e.g., "*.root")')
    parser.add_argument('config_file', type=str, help='Path to the configuration file specifying keys to extract')
    return parser.parse_args()

def read_config(config_file):
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config['keys_to_extract'], config['x_name'], config['x_regex'], config['output_file']

def annotate_points(df, col_name):
    for index, row in df.iterrows():
        if col_name != row['Key']:
            continue
        x = row['xVal']
        y = row['Mean']
        val = f'{row["Mean"]:.3f}'

        if 'residuals' in col_name:
            y = row['StdDev']
            val = f'{row["StdDev"]:.3f}'
        plt.annotate(val, (x, y), textcoords="offset points", xytext=(15, 15), ha='center', fontsize=8)

def main():
    args = parse_arguments()
    
    root_file_list = glob.glob(f'{args.root_path_pattern}/*.root')
    keys_to_extract, x_name, x_regex, output_file = read_config(args.config_file)
    
    results = {"File": [], "Key": [], "Name": [], "xVal": [], "Mean": [], "StdDev": [], "StdErr": [], "N": []}
    
    for root_file in root_file_list:
        with uproot.open(root_file) as file:
            for key_name in keys_to_extract:
                try:
                    tkey = file[key_name[0]]
                    mean_val = 0
                    std_dev_val = 0
                    N = 0
                    if isinstance(tkey, uproot.behaviors.TH1.TH1):
                        hist_np = tkey.to_numpy()
                        unjagged_bins = (hist_np[1][:-1] + hist_np[1][1:]) / 2
                        N = np.sum(hist_np[0])
                        mean_val = np.sum(hist_np[0] * unjagged_bins) / N
                        std_dev_val = np.sqrt(np.sum(hist_np[0] * (unjagged_bins - mean_val)**2) / N)

                    elif isinstance(tkey, uproot.behaviors.TProfile2D.TProfile2D):
                        vals = tkey.values()
                        mean_val = np.average(vals) * 100
                        std_dev_val = np.std(vals) * 100
                        N = 1
                    else:
                        continue

                    results["xVal"].append(float(re.search(x_regex, root_file).group(1)))
                    results["Mean"].append(mean_val)
                    results['StdDev'].append(std_dev_val)
                    results['File'].append(root_file)
                    results['Key'].append(key_name[0])
                    results['Name'].append(key_name[1])
                    results["N"].append(N)
                    results["StdErr"].append(std_dev_val / np.sqrt(N))

                except KeyError:
                    print(f"Key '{key_name}' not found in file '{root_file}'")

    df = pd.DataFrame(results)
    
    plt.figure(figsize=(15, 10))

    for i, key in enumerate(keys_to_extract):
        keyrows = df[df['Key'].str.contains(key[0])]
        plt.subplot(3, 1, i + 1)
        
        x = keyrows['xVal']
        y = keyrows['Mean']
        if 'residuals' in key[0]:
            y = keyrows['StdDev']
        
        yerr = df[df['Key'].str.contains(key[0])]['StdErr'].values
        
        sns.lineplot(x=x, y=y, label='Data', marker='o')
        plt.errorbar(x, y, yerr=yerr, fmt='.', color='blue', capsize=5)
        
        plt.xlabel('')
        plt.ylabel(key[1])
        plt.title(key[1])
        annotate_points(df, key[0])
        plt.grid(True, which='both')
        plt.gca().set_yticks(plt.gca().get_yticks())
        ax = plt.gca()
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(ax.get_xticks())

    plt.xlabel(x_name)
    plt.tight_layout()
    plt.savefig(output_file)
    print('now showing')
    plt.show()

if __name__ == "__main__":
    main()

