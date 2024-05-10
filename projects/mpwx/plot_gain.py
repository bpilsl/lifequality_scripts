import numpy as np
# from /home/bernhard/hephy/sw_dev/lifequality_scripts/projects/mpwx/mpwx_interpreter import *
import argparse
from scipy.optimize import OptimizeWarning
import warnings
import matplotlib.pyplot as plt
import re
from glob import glob
import os
import pandas as pd
import sys
# from home.bernhard.hephy.sw_dev.lifequality_script.projects.mpws.mpwx_interpreter import *
import matplotlib as mpl

sys.path.append('/home/bernhard/hephy/sw_dev/lifequality_scripts/projects/mpwx')
from mpwx_interpreter import *


pd.set_option('mode.chained_assignment', None)

def parse_args():
    parser = argparse.ArgumentParser(description='Analysis script for multiple directories.')
    parser.add_argument('dirs', nargs='+', help='List of directories to analyze')
    parser.add_argument('--bl', type=float, required=True, help='Baseline voltage')
    parser.add_argument('--iv', nargs=1, type=float, help='Specific injection voltage')
    parser.add_argument('--fit', nargs=2, type=float, required=False, help='Range to perform linear fit')
    return parser.parse_args()

def main():
    args = parse_args()

    data_colors = ['b', 'g', 'r', 'c', 'm']
    first = True
    for directory in args.dirs:
        files = glob(os.path.join(directory, '*.txt'))
        files.sort()
        single = False
        template = {'Voltage': [], 'Hits': [], 'Index': [], 'Injection': []}
        pixel_data = {'Pixel': [], 'Voltage': [], 'Hits': [], 'Index': []}
        dummy = readScurveData(files[0])
        df = pd.DataFrame(dummy)
        inj_voltages = df['Voltage'].unique()  
        grouped = {}      
        stats = {'VInj': [], 'VT50Err': [], 'VT50Mean': [], 'NoiseErr': [], 'NoiseMean': []}


        if args.iv:
            inj_voltages = args.iv
            single = True
        else:
            grouped = {}
            dummy = readScurveData(files[0])
            df = pd.DataFrame(dummy)
            inj_voltages = df['Voltage'].unique()

        for inj_voltage in inj_voltages:
            for idx, file in enumerate(files):
                df = readScurveData(file)
                thr = float(re.search(r'thr_(\d+\.*\d*)', file).group(1)) * 1e3
                effective_thr = thr - args.bl
                exploded_df = df.explode('Voltage').explode('Hits')
                selected_rows = exploded_df[exploded_df['Voltage'] == inj_voltage]
                selected_rows['Injection'] = selected_rows['Voltage'][:]
                selected_rows['Voltage'][:] = effective_thr

                for pixel, group in selected_rows.groupby('Pixel'):
                    pixel_data['Pixel'].append(pixel)
                    pixel_data['Voltage'].append(*group['Voltage'].tolist())
                    pixel_data['Hits'].append(*group['Hits'].tolist())
                    pixel_data['Index'].append(*group['Index'].tolist())
                    pixel_dict = {
                        'Voltage': group['Voltage'].tolist(),
                        'Hits': group['Hits'].tolist(),
                        'Index': group['Index'].tolist(),
                        'Injection': inj_voltage
                    }
                    grouped[pixel] = pixel_dict

            df = pd.DataFrame(pixel_data)
            for key in pixel_data.keys():
                pixel_data[key] = []
            s_data = interpretScurve(df, doPlot=single, xAxisLabel='Threshold voltage [mV]', title='Response at V$_{inj} = $' + str(inj_voltage) + 'mV')
            if not s_data['halfWayGaussFit']:
                continue

            vt50_mean = s_data['halfWayGaussFit'][0]
            vt50_err = s_data['halfWayGaussFit'][2]
            noise_mean = s_data['noiseGaussFit'][0]
            noise_err = s_data['noiseGaussFit'][2]
            stats['VT50Mean'].append(vt50_mean)
            stats['VT50Err'].append(vt50_err)
            stats['NoiseMean'].append(noise_mean)
            stats['NoiseErr'].append(noise_err)
            stats['VInj'].append(inj_voltage)

        if not single:
            fit_in_mean = np.array(stats['VT50Mean'])
            fit_in_inj = np.array(stats['VInj'])
            if args.fit:
                mask = np.bitwise_and(fit_in_inj > args.fit[0], fit_in_inj < args.fit[1])
            else:
                mask = np.bitwise_and(fit_in_mean > 1, fit_in_mean < 1500)

            fit_in_inj = fit_in_inj[mask]
            fit_in_mean = fit_in_mean[mask]
            if len(fit_in_inj) > 0:
                print(f'useable range : V_inj = ({fit_in_inj[0]}, {fit_in_inj[-1]})')
            else:
                print(os.path.basename(directory), 'totally broken')
                continue
            # plt.text(fit_in_inj[-1], max(fit_in_mean) * .8, f'$V_{"{inj}"} \in$ ({fit_in_inj[0]}mV, {fit_in_inj[-1]}mV)')
            fit = np.polyfit(fit_in_inj, fit_in_mean, deg=1)
            fit_q = np.polyfit(v_to_q(fit_in_inj), fit_in_mean, deg=1)
            fit_x = np.linspace(min(fit_in_inj), max(fit_in_inj), 100)
            fit_y = np.poly1d(fit)(fit_x)
            ax = plt.subplot(111)
            df = pd.DataFrame(stats)

            dac_val = re.search(r'\d+', os.path.basename(directory))
            if dac_val:  # if we plot a dac folder, get a color in the range from 0 to 63
                dac_val = int(dac_val.group(0))
                c_norm = mpl.colors.Normalize(vmin=0, vmax=63)

                c_map = mpl.cm.viridis
                s_map = mpl.cm.ScalarMappable(cmap=c_map, norm=c_norm)
                s_map.set_array([])
                color = s_map.to_rgba(dac_val)
            else:
                color = None


            ax.errorbar(df['VInj'], df['VT50Mean'], yerr=df['VT50Err'], fmt='o', markersize=2, capsize=5,
                        label=f'Data VT50 ({str.upper(os.path.basename(directory))})', c=color)
            # ax.errorbar(df['VInj'], df['NoiseMean'], yerr=df['NoiseErr'], fmt='o', markersize=2, capsize=5,
            #             label=f'Noise ({str.upper(os.path.basename(directory))})')
            # ax.errorbar(df['VInj'], df['VT50Mean'], yerr=df['VT50Err'], fmt='o', markersize=2, capsize=5,
            #             label=f'Data')
            ax.plot(fit_x, fit_y, linestyle='dashed', label=f'Fit ({str.upper(os.path.basename(directory))}): {fit_q[0]:.4f} * Q + {fit_q[1]:.2f}\n Gain: {fit_q[0] * 1e3:.2f} $\mu V / e^-$', c=color)
            # plt.title(f'Gain ({directory}): {fit_q[0] * 1e3:.2f} $\mu V / e^-$', fontsize=40)
            # plt.title(f'Gain: {fit_q[0] * 1e3:.2f} $\mu V / e^-$', fontsize=40)
            ax.set_xlabel('$V_{Inj}$ [mV]', fontsize=30)
            ax.set_ylabel('VT50 [mV]', fontsize=30)
            plt.legend(fontsize=12)
            plt.ylim(0, max(fit_in_mean) * 1.2)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            if first:
                # secax_y = ax.secondary_yaxis('right', functions=(v_to_q, q_to_v))
                # secax_y.set_ylabel('VT50 [$e^-$]')
                secax_x = ax.secondary_xaxis('top', functions=(v_to_q, q_to_v))
                secax_x.set_xlabel(r'$Q_{inj}$ [$e^-$]', fontsize=30)
                secax_x.tick_params(labelsize=20)
            first = False

    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
