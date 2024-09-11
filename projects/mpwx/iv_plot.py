import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

labels = ['Backside biased', 'Topside biased']
min_u = 120
font_small = 25
font_large = 30


def read_data(file):
    f = open(file)

    u = []
    i = []
    data = []

    for line in f:
        if line.startswith('New run'):
            if len(u) > 0:
                stats = [np.mean(u), np.std(u), np.mean(i), np.std(i)]
                u = []
                i = []
                data.append(stats)
            continue
        if len(line.strip()) == 0 or line.startswith('#'):
            continue
        splitted = line.split(',')
        if abs(float(splitted[0])) < min_u:
            continue
        u.append(abs(float(splitted[0])))
        i.append(abs(float(splitted[1])))

    if len(u) > 0:
        stats = [np.mean(u), np.std(u), np.mean(i), np.std(i)]
        u = []
        i = []
        data.append(stats)

    data = np.array(data)
    data = data[data[:, 0].argsort()]
    return data


fig, ax = plt.subplots(figsize=(16, 9))
colors = sns.color_palette('tab10', len(labels))  # Generate a color palette
for i, label in enumerate(labels):
    data = read_data(sys.argv[i + 1])
    d = {'U': data[:, 0], 'I': data[:, 2]}
    df = pd.DataFrame(d)
    color = colors[i]  # Assign color from the palette
    sns.lineplot(data=df, x='U', y='I', color=color, linewidth=2)
    # ax.errorbar(data[:, 0], data[:, 2], yerr=data[:, 3], fmt='o', capsize=6, markersize=5, label=label, color=color)
    ax.scatter(data[:, 0], data[:, 2], label=label, color=color, marker='X', s=70)
    ax.set_xlabel('Reverse Bias Voltage [V]', fontsize=font_small)
    ax.set_ylabel('Current [A]', fontsize=font_small)
    ax.set_title('IV characteristics', fontsize=font_large)
    ax.set_yscale('log')
    plt.grid(True)
    plt.legend(fontsize=font_small)
    plt.xticks(fontsize=font_small)
    plt.yticks(fontsize=font_small)

plt.show()
