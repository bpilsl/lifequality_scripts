import sys
import re
from matplotlib import pyplot as plt
import numpy as np
from cycler import cycler

in_file = sys.argv[1]
out_file = sys.argv[2]

in_file = open(in_file)
out_file = open(out_file, 'w')



currPix = None
vnfb = None
inj_volt = None
data = []
bla = {'pix': currPix, 'vinj': [], 'tot': []}

def v_to_q(v):
    return v * 2.8e-15 / 1.62e-19  # 2.8fF injection capacitance and convert to units of e-


def find_qinj(tot_value, coefficients):
    # Coefficients from np.polyfit
    k, d = coefficients
    return (tot_value - d) / k

# Create a color cycler
colors = plt.cm.tab10.colors
plt.gca().set_prop_cycle(cycler('color', colors))

for line in in_file:
    line = line.strip()
    pixel_match = re.search(r'Pixel: (\d+):(\d+)', line)
    if pixel_match:
        if len(bla['vinj']) > 0:
            bla['vinj'] = np.array(bla['vinj'])
            bla['qinj'] = v_to_q(bla['vinj'])
            data.append(bla)
        currPix = (int(pixel_match.group(1)), int(pixel_match.group(2)))
        bla = {'pix': currPix, 'vinj': [], 'totMean': [], 'totStdDev': []}
        continue
    if line.startswith('#') or len(line) == 0:
        continue
    if line.startswith('VNFB'):
        vnfb = int(line.split('=')[1])
        continue

    values = line.split(' ')
    inj_volt = float(values[0].replace(':', ''))
    ToTs = np.array(values[1:]).astype(int)
    bla['vinj'].append(inj_volt)
    if len(ToTs) > 0:
        bla['totMean'].append(np.mean(ToTs))
        bla['totStdDev'].append(np.std(ToTs))
    else:
        bla['totMean'].append(0)
        bla['totStdDev'].append(0)

if len(bla['vinj']) > 0:
    print('last ' , bla['pix'])
    bla['vinj'] = np.array(bla['vinj'])
    bla['qinj'] = v_to_q(bla['vinj'])
    data.append(bla)

# breakpoint()

ax1 = plt.subplot(311)
ax1.xaxis.set_label_coords(0.5, 0.15)
plt.xlabel('$Q_{Inj} [e^-]$', fontsize=20)
plt.ylabel('ToT [LSB]', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid()

fit_params = {'k': [], 'd': []}

for d in data:
    # Extract a color from the cycle
    color = next(plt.gca()._get_lines.prop_cycler)['color']

    # Plot the data points with error bars
    ax1.errorbar(d['qinj'], d['totMean'], yerr=d['totStdDev'], capsize=2, fmt='o', color=color, label=d['pix'])

    try:
        # breakpoint()
        fit = np.polyfit(d['qinj'], d['totMean'], 1)
        if fit[0] == .0:
            print('invalid fit for ', d['pix'])
            continue
        fit_params['k'].append(1 / fit[0])
        fit_params['d'].append(fit[1])
        out_file.write(f'{d["pix"][0]} {d["pix"][1]} {fit[0]} {fit[1]}\n')

        xfit = np.linspace(min(d['qinj']), max(d['qinj']), 50)
        yfit = np.poly1d(fit)(xfit)
        # breakpoint()
        ax1.plot(xfit, yfit, c=color)
    except Exception as e:
        print('fit for pix ', d['pix'], ' failed')

for key in fit_params:
    fit_params[key] = np.array(fit_params[key])

mean_fit_params = (np.mean(fit_params['k']), np.mean(fit_params['d']))
std_dev_fit_params = (np.std(fit_params['k']), np.std(fit_params['d']))
print(f'mean fit params: {mean_fit_params}, std.dev. = {std_dev_fit_params}')

ax2 = plt.subplot(312)
plt.xlabel('Slope [$e^- / ToT$]', fontsize=20)
plt.ylabel('Counts', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax2.hist(fit_params['k'], bins=50, label='Histogram of "slope"')
plt.grid()
ax2.xaxis.set_label_coords(0.5, 0.15)


ax3 = plt.subplot(313)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Offset [$ToT$]', fontsize=20 )
plt.ylabel('Counts', fontsize=20)
ax3.xaxis.set_label_coords(0.5, 0.15)
# ax3.yaxis.set_label_coords(0.5, 1.05)
plt.grid()
ax3.hist(fit_params['d'], bins=50, label='Histogram of "d"')



plt.show()
