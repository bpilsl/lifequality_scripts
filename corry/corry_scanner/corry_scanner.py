import yaml
import argparse
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import subprocess

tmp_corry_config = 'tmp_config'


def generate_tmp_config_file(**kwargs):
    # Function to generate a new temporary config file based on the provided config and run number

    new_config_file = tmp_corry_config + str(kwargs['scan_nmb'])
    sed_cmd = f'sed "s/{kwargs["search_pattern"]}/{kwargs["replace_pattern"]}/" {kwargs["template_config"]} > {new_config_file}'

    print('sedding', sed_cmd)

    try:
        subprocess.run(sed_cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f'Error occurred while executing sed command: {e}')

    return new_config_file


# Function to run the "corry" tool with specific parameters
def run_corry(config, **kwargs):
    print('run corry')

    c_file = config["global"]["template_config"]
    if 'tmp_config' in kwargs:
        c_file = kwargs['tmp_config']

    cmd = f'{config["global"]["corry_bin"]} -c {c_file} '

    params = kwargs.get('params')
    if params:
        # Get the parameter name and its value
        param_name, param_value = next(iter(params.items()))
        cmd += (f'-l {config["global"]["logfile"]}_{param_name}_{param_value}.txt '
                f'-o histogram_file={config["global"]["output_file"]}_{param_name}_{param_value}.root ')
        # Append custom parameters to the command
        for key, val in params.items():
            cmd += f' -o {key}={val}'
    else:
        cmd += (f'-l {config["global"]["logfile"]}_{kwargs["output_modifier"]}.txt '
                f'-o histogram_file={config["global"]["output_file"]}_{kwargs["output_modifier"]}.root ')

    if 'output_dir' in kwargs and kwargs['output_dir'] is not None:
        cmd += f'-o output_directory={kwargs["output_dir"]}'

    print('cmd = ', cmd)

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

    def run_single_scan(scan, cfg_file):
        if scan['type'] == 'range':
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                for i in np.arange(scan['lo'], scan['hi'], scan['inc']):
                    if multi_threaded:
                        print(f'submitting thread for scan parameter {scan["param"]} = {i}')
                        executor.submit(run_corry, config, params={scan["param"]: i}, output_dir=scan.get('output_dir'),
                                        tmp_config=cfg_file)
                        print('executor submitted')
                    else:
                        run_corry(config, params={scan["param"]: i}, output_dir=scan['output_dir'], tmp_config=cfg_file)
        if scan['type'] == 'cfg_replace':
            run_corry(config,  output_dir=scan.get('output_dir'), tmp_config=cfg_file, output_modifier=scan.get('name'))

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for scan_index, scan in enumerate(scans):
            corry_config = config["global"]["template_config"]
            if 'template_config' in scan:
                corry_config = generate_tmp_config_file(template_config=scan['template_config'], scan_nmb=scan_index,
                                                        search_pattern=scan['search_pattern'],
                                                        replace_pattern=scan['replace_pattern'])

            if multi_threaded:
                executor.submit(run_single_scan, scan, corry_config)
            else:
                print(f'running scan {scan_index}')
                run_single_scan(scan, corry_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run scans using the "corry" tool based on a configuration file.')
    parser.add_argument('-c', '--config', required=True, help='YAML configuration file containing the scan parameters')
    parser.add_argument('-d', '--delete', action='store_true', default=False,
                        help='Flag to delete the temporary config file after execution (default: False)')

    args = parser.parse_args()

    print(args.config)

    # Read the configuration from the YAML file provided as a command-line argument
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Call the run_scans function to initiate the execution of the scans based on the loaded config
    run_scans(config)

    # Delete the temporary config file if the flag is set to True and the run number is provided
    if args.run is not None and args.delete:
        os.remove(tmp_corry_config)
