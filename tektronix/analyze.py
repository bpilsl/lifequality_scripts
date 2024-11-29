import pandas as pd
import matplotlib.pyplot as plt
import sys
import seaborn as sns
from glob import glob
from reportlab.lib.colors import transparent
from matplotlib.ticker import EngFormatter

def analyzeFile(file):
    result = {}
    df = pd.read_csv(file, dtype=float)
    result['low'] = min(df['Ch2 (V)'])
    result['high'] = max(df['Ch2 (V)'])
    result['p2p'] = result['high'] - result['low']

    return  result, df

file = []
fontsize = 28
plt.rcParams.update({
'font.size': fontsize,             # Base font size
'axes.titlesize': fontsize,        # Title size
'axes.labelsize': fontsize,        # X/Y label size
'xtick.labelsize': fontsize,       # X tick label size
'ytick.labelsize': fontsize,       # Y tick label size
'legend.fontsize': fontsize,       # Legend text size
'legend.title_fontsize': fontsize  # Legend title size
})
fig = plt.figure(figsize=(16, 12))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
for i, path in enumerate(sys.argv):
    color = None
    label = ''
    if i == 0:
        continue
    if i == 1:
        color = 'b'
        label = ''
    elif i == 2:
        color = 'r'
    elif i == 3:
        color = 'g'

    files = glob(f'{path}/*.csv')   
    for j, file in enumerate(files):
        label = ''
        if j == 0:
            label = path
        result, df = analyzeFile(file)
        # df.plot()
        # sns.lineplot(df, x='Time (s)', y='Ch1 (V)', label=f'{file}:Ch1')
        plt.subplot(2,1,1)
        # plt.xlim(3.35e-5, 3.45e-5)
        sns.lineplot(df, x='Time (s)', y='Ch2 (V)', color=color, label = label, ax=ax1)
        sns.lineplot(df, x='Time (s)', y='Ch1 (V)', color= color, label = label, ax=ax2)
        print('p2p = ', result['p2p'], 'V')


formatter = EngFormatter(unit='s')
ax1.xaxis.set_major_formatter(formatter)            
ax2.xaxis.set_major_formatter(formatter)                    
formatter = EngFormatter(unit='V')
ax1.yaxis.set_major_formatter(formatter)                    
ax2.yaxis.set_major_formatter(formatter)                    
ax1.grid()
ax1.set_ylabel('SFOUT')
ax1.set_xlabel('Time')
ax2.grid()
ax2.set_ylabel('HB')
ax2.set_xlabel('Time')
ax2.set_xlim(3.2e-6, 4.8e-6)

plt.savefig('tektronix.png', transparent=True)
plt.show()

