# Analyze data from speed radar
# Display statistics
# J.Beale 5/6/2025

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.stats import norm
import time

# replace element with average of N elements
def moving_avg(arr, N):
    # Ensure N is odd to have a symmetric window
    if N % 2 == 0:
        raise ValueError("N should be an odd number for symmetric averaging")

    pad_width = N // 2
    padded = np.pad(arr, pad_width, mode='edge')
    kernel = np.ones(N) / N

    # Convolve with 'valid' to get the same size as input
    smoothed = np.convolve(padded, kernel, mode='valid')
    return smoothed

def find_groups_df(dfRaw, T, N):
    kmh = dfRaw['kmh'].to_numpy()  # get just the speeds
    epoch = dfRaw['epoch'].to_numpy()  # get just the epoch timestamp
   
    nkmh = np.abs(kmh)
    mask = nkmh > T     # Boolean mask where condition is met

    padded = np.pad(mask.astype(int), (1, 1), constant_values=0)
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    #print("starts:", *starts[-2000:].astype(int))

    # Collect group data
    group_data = []
    for start, end in zip(starts, ends):
        start_epoch = epoch[start]  # event start time in epoch seconds
        end_epoch = epoch[end-1]
        duration = end_epoch - start_epoch
        if duration >= N:
            group = kmh[start:end]
            avg = moving_avg(group, 7)
            group = np.abs(avg)
            dir = np.sign(np.mean(avg)).astype(int)
            group_data.append({
                'start_time': start_epoch,
                'start_index': start,
                'dir': dir,
                'duration': duration,
                'max': group.max(),
                'avg': group.mean()
            })
    #print("Group length: %d" % len(group_data))    
    return pd.DataFrame(group_data)

def showStats(note, dfg):
    print("%s events: %d " % (note,len(dfg)),end="")
    going_left = (dfg['dir'] > 0).sum()
    going_right = (dfg['dir'] < 0).sum()
    print("Left: %d  Right: %d" % (going_left, going_right))
    print("Max speed Avg: %.2f std: %.2f" % ( dfg['max'].mean(), dfg['max'].std() ) )
    print("Avg speed Avg: %.2f std: %.2f" % ( dfg['avg'].mean(), dfg['avg'].std() ) )
    print("Duration Avg: %.2f std: %.2f" % ( dfg['duration'].mean(), dfg['duration'].std() ) )
    kmh_max = dfg['max'].max()
    mph_max = 0.621371 * kmh_max
    print("Peak speed: %.2f km/h (%.2f mph)" % (kmh_max, mph_max), end='')

    max_index = dfg['max'].idxmax() # index of max-speed event    
    print("  at %s PDT  dir: %d" % (dfg.at[max_index,'datetime'], dfg.at[max_index,'dir']))

# Plot histogram ===================
def doPlot(dfg1):
    plt.hist(dfg1['max'], bins=12, range=(20, 80), edgecolor='black')
    plt.xlabel('km/h')
    plt.ylabel('events')
    plt.title('Speeds in km/h  '+hr_string+lastDate )
    plt.grid('both')
    plt.show()

# ===============================================

# input data file from radar
in_dir = r"C:\Users\beale\Documents\doppler"

fname = r"20250504_202105_SerialLog.csv"
#fname = "20250505_212916_SerialLog.csv"

in_path = os.path.join(in_dir, fname)

dfRaw = pd.read_csv(in_path)

kmh_speed = dfRaw['kmh'].to_numpy()  # get just the speeds
epoch = dfRaw['epoch'].to_numpy()  # get just the seconds timestamp

dur = (epoch[-1] - epoch[0])/(60*60.0) # duration in hours
hr_string = ("%.1f hours " % dur)
print(hr_string)

T = 20.0 # threshold in km/h for interesting event
N = 2.5 # duration in seconds
# result = get_groups(kmh_speed, T, N)

fraction = np.mean(np.abs(kmh_speed) < T)

print("File: %s" % fname)
print("Readings: %d  frac below %.1f: %.3f" % (len(dfRaw),T,fraction))


dfg1 = find_groups_df(dfRaw, T, N)
pd.set_option('display.max_columns', None)

dfg1['datetime'] = pd.to_datetime(dfg1['start_time'], unit='s', utc=True
        ).dt.tz_convert('US/Pacific').dt.strftime('%Y-%m-%d %H:%M:%S.%f')
dfg1['datetime'] = dfg1['datetime'].str[:-4] # remove excess digits



lastDate = dfg1['datetime'].iloc[-1]
firstDate = dfg1['datetime'].iloc[0]
firstDateStr = "Start: " +str(firstDate)[0:-3]
print("%s" % firstDateStr)
print("Last event: %s" % lastDate)
# print(dfg1)
pd.set_option('display.float_format', '{:.2f}'.format)
#print(dfg1[['datetime', 'dir', 'duration', 'max', 'avg']])

showStats("Summary", dfg1)

# =======================
# Enable interactive plot mode
# Interactive plotting

plt.ion()
fig, ax = plt.subplots()

data_so_far = []
bin_count = 20
bin_range = (20, 80)
bin_edges = np.linspace(bin_range[0], bin_range[1], bin_count + 1)
bin_width = bin_edges[1] - bin_edges[0]
bin_stats = [] # Store bin stats for final summary

for index, row in dfg1.iterrows():
    data_so_far.append(row['max'])

    ax.clear()

    mu = np.mean(data_so_far)
    sigma = np.std(data_so_far)
    N = len(data_so_far)

    # Histogram with counts
    counts, bins, patches = ax.hist(data_so_far,
                                     bins=bin_edges,
                                     edgecolor='black',
                                     color='skyblue')

    # Scale y-axis so labels don't overflow top edge of graph
    max_count = max(counts)
    ax.set_ylim(0, max_count * 1.2)

    # Gaussian curve scaled to counts
    x = np.linspace(bin_range[0], bin_range[1], 500)
    y = norm.pdf(x, mu, sigma) * N * bin_width
    ax.plot(x, y, 'g--', label=f'Normal($\mu$={mu:.1f}, $\sigma$={sigma:.1f})')

    bin_stats.clear()

    for i in range(len(bin_edges) - 1):
        p_bin = norm.cdf(bin_edges[i + 1], mu, sigma) - norm.cdf(bin_edges[i], mu, sigma)
        expected = N * p_bin
        observed = counts[i]

        if expected > 0:
            z = (observed - expected) / np.sqrt(expected)
            p_val = 2 * norm.sf(abs(z))
        else:
            p_val = 1.0

        # Save for summary
        bin_stats.append({
            'bin_range': f'{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}',
            'observed': int(observed),
            'expected': expected,
            'p_value': p_val
        })

        # Label formatting
        label = f'p={p_val:.1e}' if p_val < 0.01 else f'p={p_val:.2f}'

        # Highlight if surprising
        if p_val < 0.01:
            patches[i].set_edgecolor('red')
            patches[i].set_linewidth(2)

        ax.text((bin_edges[i] + bin_edges[i + 1]) / 2,
                observed + 0.5,
                label,
                ha='center',
                va='bottom',
                fontsize=10)


    ax.set_title(f'Histogram with p-values for {index+1} vehicles')
    ax.set_xlabel('speed, km/h', fontsize = 14)
    ax.set_ylabel('vehicle count', fontsize = 14)
    # Show x-axis ticks every 10 units
    xmin = bin_range[0]
    xmax = bin_range[1]+1
    step = 5
    ax.set_xticks(ticks=range(xmin, xmax, step))
    ax.tick_params(axis='y', labelsize=14)    
    ax.tick_params(axis='x', labelsize=14)
    ax.grid(True, axis='y', linestyle=':', linewidth=1)
    ax.legend()
    ax.annotate(firstDateStr, 
            xy=(0, 1.01), xycoords='axes fraction',
            ha='left', va='bottom', fontsize=11)


    plt.pause(0.01)

plt.ioff()

ax.annotate((hr_string + " ending " + lastDate[:-3]), 
            xy=(1, 1.01), xycoords='axes fraction',
            ha='right', va='bottom', fontsize=11)

plt.show()


# Final summary of most surprising bins
print("\n%d bins : most surprising:" % bin_count)
bin_stats_sorted = sorted(bin_stats, key=lambda x: x['p_value'])

for stat in bin_stats_sorted:
    if (stat['p_value'] > 0.01):
        continue
    print(f"Bin {stat['bin_range']}: Observed = {stat['observed']:3d}, "
          f"Expected = {stat['expected']:5.2f}, p-value = {stat['p_value']:.4f}")
