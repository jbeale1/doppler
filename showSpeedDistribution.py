# Analyze data from speed radar
# Display statistics
# J.Beale 5/6/2025

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.stats import norm, shapiro, probplot
import time
from scipy.ndimage import binary_dilation
from filterpy.kalman import KalmanFilter
# ==============================================================

# input data file from radar
#in_dir = r"C:\Users\beale\Documents\doppler"
in_dir = "/home/john/Documents/doppler"

#fname = r"20250504_202105_SerialLog.csv"
#fname = r"20250505_212916_SerialLog.csv"
#fname = r"20250506_220329_SerialLog.csv"
#fname = r"20250507_222129_SerialLog.csv"
#fname = r"20250508_000003_SerialLog.csv"
fname = r"20250509_000003_SerialLog.csv"

# ==============================================================
#  return indices where discrete difference is above threshold, or empty array
def get_spikes(arr, threshold):
    diffs = np.diff(arr)
    indices = np.where(np.abs(diffs) > threshold)[0]
    max = np.max(diffs)
    return indices, max

# replace element with average of N elements
def moving_avg(arr, N):
    if N % 2 == 0:
        raise ValueError("N should be an odd number for symmetric averaging")

    pad_width = N // 2
    padded = np.pad(arr, pad_width, mode='edge')
    kernel = np.ones(N) / N

    # Convolve with 'valid' to get the same size as input
    smoothed = np.convolve(padded, kernel, mode='valid')
    return smoothed

# remove spikes faster than a_max in m/s^2
def clean_spikes(speed_kmh, a_max = 5):
    speed_ms = speed_kmh / 3.6   # convert to m/s
    dt = 0.09  # 90 ms
    acceleration = np.diff(speed_ms) / dt
    acceleration = np.insert(acceleration, 0, 0)
    # print(acceleration)
    valid_mask = np.abs(acceleration) <= a_max
    expanded_mask = binary_dilation(valid_mask, structure=np.ones(3))
    cleaned_speed_ms = speed_ms.copy()
    # Use linear interpolation for invalid segments
    invalid_indices = np.where(~expanded_mask)[0]
    valid_indices = np.where(expanded_mask)[0]
    cleaned_speed_ms[invalid_indices] = np.interp(invalid_indices, valid_indices, cleaned_speed_ms[valid_indices])
    cleaned_speed_kmh = cleaned_speed_ms * 3.6
    return cleaned_speed_kmh

def kalman_filter(data, dir, dt=0.09):
    if (dir < 0): # going away from sensor, better to reverse time
        speed_kmh = data[::-1]
    else:
        speed_kmh = data        
    n = len(speed_kmh)
    speed_measurements = speed_kmh / 3.6  # convert to m/s
    
    # State: [position, velocity, acceleration]
    kf = KalmanFilter(dim_x=3, dim_z=1)
    kf.x = np.array([[0.], [speed_measurements[0]], [0.]])
    kf.F = np.array([[1, dt, 0.5*dt**2],
                    [0, 1, dt],
                    [0, 0, 1]])
    kf.H = np.array([[0, 1, 0]])  # still only measuring velocity

    kf.P *= 1000.
    kf.R = 1.0
    kf.Q = np.array([
        [dt**4/4, dt**3/2, dt**2/2],
        [dt**3/2, dt**2,   dt],
        [dt**2/2, dt,      1]
    ]) * 0.5  # Tune this

    filtered_speed = []

    for z in speed_measurements:
        kf.predict()
        kf.update(z)
        filtered_speed.append(kf.x[1, 0])  # velocity component

    if (dir < 0):
        out = filtered_speed[::-1]
    else:
        out = filtered_speed

    return np.array(out) * 3.6  # back to km/h

# find the largest segment without jumps larger than T
def longest_stable_segment(arr, T):
    arr = np.asarray(arr)
    if len(arr) < 2:
        return (0, len(arr))  # Edge case

    diffs = np.abs(np.diff(arr))
    breakpoints = np.where(diffs > T)[0]
    segment_ends = np.concatenate(([ -1 ], breakpoints, [ len(arr) - 1 ]))
    starts = segment_ends[:-1] + 1
    ends = segment_ends[1:] + 1
    lengths = ends - starts
    max_idx = np.argmax(lengths)
    return starts[max_idx], ends[max_idx]


# plot speed of fastest vehicle in dataset    
def doPlotOne(times, speeds):    
    plt.title("Vehicle event")    
    plt.scatter(times, speeds, s=4)
    plt.ylabel('speed, km/h')
    plt.xlabel('sample number')
    plt.grid('both')
    plt.show(block=False)

def find_groups_df(dfRaw, T, N):
    kmh = dfRaw['kmh'].to_numpy()  # get just the speeds
    epoch = dfRaw['epoch'].to_numpy()  # get just the epoch timestamp
   
    nkmh = np.abs(kmh)
    mask = nkmh > T     # Boolean mask where condition is met

    padded = np.pad(mask.astype(int), (1, 1), constant_values=0)
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    print("Starts count: %d" % len(starts))

    #print("starts:", *starts[-2000:].astype(int))

    # Collect group data
    group_data = []
    for start, end in zip(starts, ends):
        start_epoch = epoch[start]  # event start time in epoch seconds
        end_epoch = epoch[end-1]
        duration = end_epoch - start_epoch
        if duration >= N:
            group = kmh[start:end]
            dStart,dEnd = longest_stable_segment(group, 5) # stability requirement
            dStart += start
            dEnd += start
            #if (dStart != start) or (dEnd != end):
            if False:                
                print("%d,%d,%d,%d " % (start,end,dStart,dEnd),end="")            
                input(" <enter>")

            start_epoch = epoch[dStart]  # event start time in epoch seconds
            end_epoch = epoch[dEnd-1]
            duration = end_epoch - start_epoch
            if (duration) >= N:                
                groupA = kmh[dStart:dEnd] # reassign the group to avoid noise   
                # cleaner = clean_spikes(groupA)      
                dir = np.sign(np.mean(groupA)).astype(int)
                smoothed = kalman_filter(groupA, dir)       
                avg = moving_avg(smoothed, 7)
                group = np.absolute(avg)
                size = len(group)        
                # doPlotOne(range(size), group)
                max = group.max()
                if (max > 70):
                    print(start_epoch, max)
                group_data.append({
                    'start_time': start_epoch,
                    'start_index': dStart,
                    'end_index': dEnd,
                    'dir': dir,
                    'duration': duration,
                    'max': max,
                    'avg': group.mean()
                })
            else:
                print("Skip duration ",duration)
    print("Group length: %d" % len(group_data))    
    return pd.DataFrame(group_data)

# display Q-Q plot
def showQQ(dfg, dfg1, dfRaw):
    speeds = dfg1['max'].to_numpy()
    stat, p = shapiro(speeds)
    print("Shapiro-Wilk statistic = %.4f, p-value = %0.3e" % (stat, p))

    count = len(dfg)
    firstDate = dfg['datetime'].iloc[0][:-6] # only hh:mm
    epoch0 = dfRaw['epoch'].iloc[0]
    epoch1 = dfRaw['epoch'].iloc[-1]
    dur = (epoch1 - epoch0)/(60*60.0) # duration in hours
    probplot(speeds, dist="norm", plot=plt)
    title = ("Probability Plot   [%d in %.1f h] %s" % (count, dur, firstDate))
    plt.title(title)
    plt.ylabel('speed, km/h')
    plt.grid('both')
    plt.show()


# plot speed of fastest vehicle in dataset    
def showFast(speeds, dir, label):    
    plt.title("Fastest vehicle  %s" % label)    
    plt.plot(speeds, 'x')
    cleaner = kalman_filter(speeds, dir)    
    smooth = moving_avg(cleaner, 7)
    plt.plot(smooth, linewidth = 1, color='#40B000')
    # plt.plot(cleaner, linewidth = 1, color='#B04000')
    plt.ylabel('speed, km/h')
    plt.xlabel('sample number')
    plt.grid('both')
    plt.show()

def showStats(note, dfg):
    print("%s events: %d " % (note,len(dfg)),end="")
    going_left = (dfg['dir'] > 0).sum()
    going_right = (dfg['dir'] < 0).sum()
    print("Left: %d  Right: %d" % (going_left, going_right))
    print("Max speed Avg: %.2f std: %.2f" % ( dfg['max'].mean(), dfg['max'].std() ) )
    print("Avg speed Avg: %.2f std: %.2f" % ( dfg['avg'].mean(), dfg['avg'].std() ) )
    print("Duration Avg: %.2f std: %.2f" % ( dfg['duration'].mean(), dfg['duration'].std() ) )
    index_max = dfg['max'].idxmax()
    kmh_max = dfg.at[index_max, 'max']
    duration = dfg.at[index_max, 'duration']
    avg = dfg.at[index_max, 'avg']
    kmh_max = dfg['max'].max()
    mph_max = 0.621371 * kmh_max
    print("Max: %.2f km/h (%.2f mph) %.1f avg %.1f sec" % 
          (kmh_max, mph_max, avg, duration), end='')

    dtime = dfg.at[index_max,'datetime']
    print("  at %s PDT  dir: %d" % (dtime, dfg.at[index_max,'dir']))
    start = dfg.at[index_max,'start_index']
    end = dfg.at[index_max,'end_index']
    
    fast_data = np.abs(kmh_speed[start:end])
    # fast_data = clean_spikes(fast_data_raw)
    dir = dfg.at[index_max,'dir']
    label = "[%d] %s PDT" % (dir, dtime)
    showFast(fast_data, dir, label) # plot speed of fastest vehicle in dataset    
    showQQ(dfg, dfg1, dfRaw) # display Probability (~ Quantile-Quantile) plot


# Plot histogram ===================
def doHistPlot(dfg1):
    plt.hist(dfg1['max'], bins=12, range=(20, 80), edgecolor='black')
    plt.xlabel('km/h')
    plt.ylabel('events')
    plt.title('Speeds in km/h  '+hr_string+lastDate )
    plt.grid('both')
    plt.show()

# ===============================================


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

outPath = os.path.join(in_dir, 'CarSpeeds.csv')
dfg1.to_csv(outPath, index=True)

# =======================
# Enable interactive plot mode
# Interactive plotting

plt.ion()
fig, ax = plt.subplots(figsize=(12, 6))

data_so_far = []
bin_count = 12
bin_range = (20, 80)
bin_edges = np.linspace(bin_range[0], bin_range[1], bin_count + 1)
bin_width = bin_edges[1] - bin_edges[0]
bin_stats = [] # Store bin stats for final summary

if True:
#for index, row in dfg1.iterrows():
    #data_so_far.append(row['max'])
    data_so_far = dfg1['max'].tolist()
    index = len(dfg1)-1

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
    ax.plot(x, y, 'g--', label=f'Normal ($\mu$={mu:.1f}, $\sigma$={sigma:.1f})')

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
        if p_val > 0.054:
            label = ""
        elif p_val > 0.01:
            label = f'p={p_val:.2f}'
        else:
            label = f'p={p_val:.1e}'

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


    ax.set_title(f'Speeds for {index+1} vehicles')
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
print()
bin_stats_sorted = sorted(bin_stats, key=lambda x: x['p_value'])

leastP = bin_stats_sorted[0].get('p_value')

if (leastP > 0.01):
    print("No histogram bins are that surprising- distribution looks normal.")
else:
    print("Of %d bins, the most surprising:" % bin_count)

    for stat in bin_stats_sorted:
        if (stat['p_value'] > 0.01):
            continue
        print(f"Bin {stat['bin_range']}: Observed = {stat['observed']:3d}, "
            f"Expected = {stat['expected']:5.2e}, p-value = {stat['p_value']:.4f}")
        
