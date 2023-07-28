import yaml
from yaml.loader import SafeLoader
import argparse
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor

tmp_corry_config = 'tmp_config'


def generate_tmp_config_file(config, run_nmb):
    # Function to generate a new temporary config file based on the provided config and run number

    new_config_file = tmp_corry_config + run_nmb
    sed_cmd = 'sed "s/' + config['global']['run_nmb_template_filename_pattern'] + (
            '/' + config['global']['run_nmb_replace_filename_pattern'] + '"' + run_nmb) + (
        f'/ {config["global"]["template_config"]} > {new_config_file}')

    print('sedding', sed_cmd)
    os.system(sed_cmd)
    return new_config_file


# Function to run the "corry" tool with specific parameters
def run_corry(config, params, scan_nmb):
    # Construct the command to execute the "corry" tool with specific parameters
    cmd = (f'{config["global"]["corry_bin"]} -c {tmp_corry_config} -l '
           f'{config["global"]["logfile"] + str(scan_nmb)} '
           f'-o histogram_file={config["global"]["output_file"] + str(scan_nmb)}')

    # Append custom parameters to the command
    for key, val in params.items():
        cmd += f' -o {key}={params[key]}'

    # Print the command that will be executed
    print('executing corry with: ', cmd)

    # Execute the "corry" tool with the constructed command using os.system
    os.system(cmd)


# Function to run multiple scans based on the configuration
def run_scans(config):
    scans = config['scans']
    num_threads = config['global']['nmb_threads']
    multi_threaded = config['global'].get('multi_threaded', True)

    def run_single_scan(scan, scan_index):
        if scan['type'] == 'range':
            n = 0
            for i in np.arange(scan['lo'], scan['hi'], scan['inc']):
                params = {scan['param']: i}
                run_corry(config, params, scan_index * num_threads + n)
                n += 1

    if multi_threaded:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            for scan_index, scan in enumerate(scans):
                executor.submit(run_single_scan, scan, scan_index)
    else:
        for scan_index, scan in enumerate(scans):
            run_single_scan(scan, scan_index)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run scans using the "corry" tool based on a configuration file.')
    parser.add_argument('-c', '--config', required=True, help='YAML configuration file containing the scan parameters')
    parser.add_argument('-r', '--run', help='Run number to be used in the temporary config file name')
    parser.add_argument('-d', '--delete', action='store_true', default=False,
                        help='Flag to delete the temporary config file after execution (default: False)')

    args = parser.parse_args()

    print(args.config)

    # Read the configuration from the YAML file provided as a command-line argument
    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.run is not None:
        tmp_corry_config = generate_tmp_config_file(config, args.run)
        config['global']['template_config'] = tmp_corry_config

    # Call the run_scans function to initiate the execution of the scans based on the loaded config
    run_scans(config)

    # Delete the temporary config file if the flag is set to True and the run number is provided
    if args.run is not None and args.delete:
        os.remove(tmp_corry_config)
