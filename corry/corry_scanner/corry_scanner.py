import yaml
import argparse
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import subprocess

tmp_corry_config = 'tmp_config'


def generate_tmp_config_file(config, run_nmb):
    # Function to generate a new temporary config file based on the provided config and run number

    new_config_file = tmp_corry_config + run_nmb
    sed_cmd = f'sed "s/{config["global"]["run_nmb_template_filename_pattern"]}/{config["global"]["run_nmb_replace_filename_pattern"]}{run_nmb}/" {config["global"]["template_config"]} > {new_config_file}'

    print('sedding', sed_cmd)

    try:
        subprocess.run(sed_cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f'Error occurred while executing sed command: {e}')

    return new_config_file


# Function to run the "corry" tool with specific parameters
def run_corry(config, params):
    print('run corry')

    # Get the parameter name and its value
    param_name, param_value = next(iter(params.items()))

    # Construct the command to execute the "corry" tool with specific parameters
    cmd = (f'{config["global"]["corry_bin"]} -c {config["global"]["template_config"]} -l '
           f'{config["global"]["logfile"]}_{param_name}_{param_value}.txt '
           f'-o histogram_file={config["global"]["output_file"]}_{param_name}_{param_value}.root')

    print('cmd = ', cmd)

    # Append custom parameters to the command
    for key, val in params.items():
        cmd += f' -o {key}={val}'

    # Print the command that will be executed
    print(f'executing corry with: {cmd}')

    # Execute the "corry" tool with the constructed command using subprocess.run
    try:
        result = subprocess.run(cmd, shell=True, check=True)
        if result.returncode != 0:
            print(f'Error occurred while running corry with parameters: {params}')
    except subprocess.CalledProcessError as e:
        print(f'Error occurred while running corry with parameters: {params}, Error: {e}')


# Function to run multiple scans based on the configuration
def run_scans(config):
    scans = config['scans']
    num_threads = config['global']['nmb_threads']
    multi_threaded = config['global'].get('multi_threaded', True)

    def run_single_scan(scan):
        if scan['type'] == 'range':
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                for i in np.arange(scan['lo'], scan['hi'], scan['inc']):
                    if multi_threaded:
                        print(f'submitting thread for scan parameter {scan["param"]} = {i}')
                        executor.submit(run_corry, config, {scan["param"]: i})
                        print('executor submitted')
                    else:
                        run_corry(config, {scan["param"]: i})

    def run_single_parameter(config, params):
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            print(params.items())
            for param_name, param_value in params.items():
                if multi_threaded:
                    print(f'submitting thread for scan parameter {param_name} = {param_value}')
                    executor.submit(run_corry, config, {param_name: param_value})
                    print('executor submitted', executor.running())
                else:
                    run_corry(config, {param_name: param_value})

    for scan_index, scan in enumerate(scans):
        print(f'running scan {scan_index + 1}')
        run_single_scan(scan)


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
