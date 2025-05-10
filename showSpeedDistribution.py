# Analyze data from speed radar
# Display vehicle & pedestrian stats
# J.Beale 5/10/2025

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.stats import norm, shapiro, probplot
import time
from scipy.ndimage import binary_dilation, label
from filterpy.kalman import KalmanFilter
from datetime import datetime
from zoneinfo import ZoneInfo

# ==============================================================

# input data file from radar
in_dir = r"C:\Users\beale\Documents\doppler"
#in_dir = "/home/john/Documents/doppler"

#fname = r"20250504_202105_SerialLog.csv"
#fname = r"20250505_212916_SerialLog.csv"
#fname = r"20250506_220329_SerialLog.csv"
#fname = r"20250507_222129_SerialLog.csv" # 9 total
#fname = r"20250508_000003_SerialLog.csv"
#fname = r"20250509_000003_SerialLog.csv"
fname = r"20250510_000003_SerialLog.csv"

# ==============================================================

def count_people(times, speeds, speed_threshold=1.5, 
                 max_abs_speed=20, min_group_size=150, time_gap=10, plot=True):
    """
    Count people walking past radar, filtering each time group to retain only the dominant direction.
    
    Parameters:
    - times: np.ndarray of shape (N,) with Unix timestamps
    - speeds: np.ndarray of shape (N,) with Doppler speed measurements
    - speed_threshold: minimum absolute speed to consider valid movement (km/h)
    - max_abs_speed: maximum absolute speed for valid movement (km/h)
    - min_group_size: minimum number of points for a group to be valid
    - time_gap: max gap (in seconds) to treat as same person
    - plot: whether to generate a plot of the data

    Returns:
    - people: list of dicts with direction and average speed
    """
    # Valid points
    valid_mask = (np.abs(speeds) > speed_threshold) & (np.abs(speeds) < max_abs_speed)
    valid_times = times[valid_mask]
    valid_speeds = speeds[valid_mask]

    # Grouping by time gaps
    time_diffs = np.diff(valid_times)
    group_breaks = np.where(time_diffs > time_gap)[0] + 1
    group_indices = np.split(np.arange(len(valid_times)), group_breaks)

    people = []
    slow_cars = []
    for idx_group in group_indices:
        if len(idx_group) < min_group_size:
            continue

        group_times = valid_times[idx_group]
        group_speeds = valid_speeds[idx_group]

        # Determine dominant direction
        num_positive = np.sum(group_speeds > 0)
        num_negative = np.sum(group_speeds < 0)
        if num_positive >= num_negative:
            dominant_mask = group_speeds > 0
            direction = "towards"
        else:
            dominant_mask = group_speeds < 0
            direction = "away"

        dominant_times = group_times[dominant_mask]
        dominant_speeds = group_speeds[dominant_mask]
        accel = np.diff(dominant_speeds)
        jerk = np.diff(accel)
        jerkstd = np.std(jerk)
        # accelstd = np.std(accel)

        # jerkstd <= 0.3 is very likely a slow car

        avg_velocity = np.mean(dominant_speeds)  
        max_speed = np.max(abs(dominant_speeds))
        max_speed_s = np.max(moving_avg(abs(dominant_speeds), 25))
        as_round = round(abs(avg_velocity), 2)        

        cat = "unknown"  
        if (max_speed_s > 17) or (jerkstd < 0.36):    
            cat = "car"
        elif (len(dominant_times) >= min_group_size) and (jerkstd >= 0.36):
            cat = "person"

        if cat == "person":
            people.append({
                "start_time": dominant_times[0],
                "end_time": dominant_times[-1],
                "direction": direction,
                "avg_speed_kmh": as_round,
                "jerk_std": round(jerkstd, 2)
            })
        if cat == "car":
            slow_cars.append({
                "start_time": dominant_times[0],
                "end_time": dominant_times[-1],
                "direction": direction,
                "avg_speed_kmh": as_round,
                "jerk_std": round(jerkstd, 2)
            })

        #if (abs(avg_velocity) > 10):
        #    plt.title("%s  js=%.3f v=%.1f" % (cat, jerkstd, max_speed_s))
        #    plt.plot(dominant_speeds)
        #    plt.show()


    if plot:
        # Plot all data
        plt.figure(figsize=(16, 6))
        plt.scatter(times, speeds, color='green', s=10, label='Radar speed')
        plt.ylim(-20, 20)
        plt.xlabel("Unix Epoch Time (s)")
        plt.ylabel("Speed (km/h)")
        plt.title("Radar Speeds with Detected Walking Intervals")

        # Overlay detected people
        for person in people:
            color = 'blue' if person['direction'] == 'towards' else 'red'
            plt.axvspan(person['start_time'], person['end_time'], color=color, alpha=0.2,
                        label=f"{person['direction'].capitalize()} (avg {person['avg_speed_kmh']} km/h)")

        # Remove duplicate legend entries
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.grid(True)
        plt.show()

    return slow_cars, people

def plot_hours(hour_counts, s):
    # Plot using matplotlib
    plt.figure(figsize=(10, 5))
    plt.bar(range(24), hour_counts, color='skyblue', edgecolor='black')
    plt.xlabel("hour of day (PDT)")
    plt.ylabel("vehicle count")
    plt.title("Vehicles per Hour  %s" % s)
    plt.xticks(range(24))
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    total_events = hour_counts.sum()
    plt.gca().text(
        0.98, 0.95, f"Total: {total_events}", 
        transform=plt.gca().transAxes, 
        ha='right', va='top', 
        fontsize=12, fontweight='normal',
        bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.5')
    )

    plt.tight_layout()
    plt.show()

# Extract hour-of-day (0–23) directly from datetime
def hour_count(event_times):
    # Convert to datetime and extract hours
    PDT = ZoneInfo("America/Los_Angeles")
    hours = np.array([datetime.fromtimestamp(ts, tz=PDT).hour for ts in event_times])
    # Count how many events fall into each hour
    hour_counts = np.bincount(hours, minlength=24)

    dt = datetime.fromtimestamp(event_times[0], tz=PDT)    
    date_string = dt.strftime("%a %#m/%#d/%y")

    plot_hours(hour_counts, date_string)
    peak_hour = np.argmax(hour_counts)
    peak_count = hour_counts[peak_hour]
    total = hour_counts.sum()
    print("Peak traffic at hour %02d with %d vehicles (%.1f%% of total)" % 
          (peak_hour, peak_count, 100.0 * peak_count/total ))

    #print("Hours, traffic for %s" % date_string)
    #for i in range(24):
    #    print(i, hour_counts[i])


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
    # print("Starts count: %d" % len(starts))

    #print("starts:", *starts[-2000:].astype(int))

    #plt.ion()
    #fig, ax = plt.subplots(figsize=(12, 6))

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
                accelR = np.diff(group)/(3.6 * 0.09) # in m/s^2
                accel = moving_avg(accelR,21)
                if (dir > 0):
                    accel = accel[0:int(size*.7)]
                else:
                    accel = accel[int(size*.3):]
                # ax.clear()
                accelMax = np.max(accel)
                accelMin = np.min(accel)
                #if (dir < 0) and (np.max(np.absolute(accel)) > 0.5):
                #    ax.plot(accel)
                #    ax.set_xticks(ticks=range(0, 200, 50))
                #    #ax.set_yticks(ticks=range(20, 65, 10))


                # doPlotOne(range(size), group)
                max = group.max()
                #if (max > 70):
                #    print(start_epoch, max)
                group_data.append({
                    'start_time': start_epoch,
                    'start_index': dStart,
                    'end_index': dEnd,
                    'dir': dir,
                    'duration': duration,
                    'max': max,
                    'avg': group.mean(),
                    'amax' : accelMax,
                    'amin' : accelMin
                })
            #else:
            #    print("Skip duration ",duration)
            #plt.pause(0.01)

    #plt.ioff()
    #plt.show()

    print("Cars: %d" % len(group_data))    
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
    
    fast_data = np.abs(speed_kmh[start:end])
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

def summarize_slow_events(events, label):
    if not events:
        print("No %s detected." % label)
        return

    speeds = [p["avg_speed_kmh"] for p in events]
    num_events = len(events)
    avg_speed = np.mean(speeds)
    min_speed = np.min(speeds)
    max_speed = np.max(speeds)

    print(f"%s: %d  Avg: %.2f km/h  Min/Max: %.2f / %.2f" % 
          (label, num_events, avg_speed, min_speed, max_speed))


# ===============================================


in_path = os.path.join(in_dir, fname)

dfRaw = pd.read_csv(in_path)

speed_kmh = dfRaw['kmh'].to_numpy()  # get just the speeds
epoch = dfRaw['epoch'].to_numpy()  # get just the seconds timestamp

dur = (epoch[-1] - epoch[0])/(60*60.0) # duration in hours
hr_string = ("%s   %.1f hours " % (fname,dur))
print(hr_string)



T = 20.0 # threshold in km/h for interesting event
N = 2.5 # duration in seconds
# result = get_groups(kmh_speed, T, N)

fraction = np.mean(np.abs(speed_kmh) < T)

#print("File: %s" % fname)
print("Readings: %d  frac below %.1f: %.3f" % (len(dfRaw),T,fraction))

slow_cars, people = count_people(epoch, speed_kmh, plot = False)
summarize_slow_events(people, "People")
summarize_slow_events(slow_cars, "Slow cars")

#avg_speeds = np.array([p["avg_speed_kmh"] for p in people])
#jerks = np.array([p["jerk_std"] for p in people])
#for s,j in zip(avg_speeds, jerks):
#    print(s,j)
#assert False, "done"

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


parts = fname.split("_")
outname = parts[0]+"_CarSpeeds.csv"
outPath = os.path.join(in_dir, outname)
dfg1.to_csv(outPath, index=False, float_format='%.2f')

event_times = dfg1['start_time'].to_numpy() # array of all start time epochs
hour_count(event_times)  # find how many each hour

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


print("Peak accel: %.2f  %.2f m/s^2" % (dfg1['amax'].max(), dfg1['amin'].min()))

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

print("# ==================================================")
