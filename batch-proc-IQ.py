#!/home/john/anaconda3/envs/cv/bin/python

# batch-process set of .mp3 files, one by one
# IQ-plot1.py will append results to logfile
# 12-Jan-2023 J.Beale

import os       # pathname composition
import sys      # exit()
import datetime # current date
import time     # current time
import subprocess # run scp, sox
import glob     # list of files in directory


gdir="/home/john/Audio/images/old/2023/"  # guide directory, list of .png files
# path to remote host directory with .mp3 files
rdir="john@john-Z83-4.local:/media/john/Seagate4GB/MINIX-John/Doppler1/old/"
ldir="/dev/shm/"  # local working directory
resultFile = "/home/john/Audio/images/DopplerD-Jan.csv"
proc="/home/john/Audio/images/doppler/IQ-plot1.py" # process one .mp3 file

# header for output CSV table
cheader = "epoch, dir, max(mph), avg(mph), min(mph), std(px), area(px), "
cheader += "length(ft), distance(ft), duration, kind"

with open(resultFile, 'w') as f:
    dstring = time.strftime('%Y-%b-%d %H:%M:%S')

    f.write(cheader+"\n")  # start output file with column header line
    f.write("# Run at %s\n" % dstring)

flist = glob.glob(gdir + "DpD_*.png")  # list of all known mp3 files
flist.sort() # let's do them in ascending order

for fpath in flist:
    fpath1 = os.path.splitext(fpath)[0]
    froot = os.path.basename(fpath1) # base filename from full path
    #print(froot)  # of form: "DpD_2023-01-11_14-05-00"

    # froot = "DpD_2023-01-11_20-55-00"
    fname_mp3 = froot + ".mp3"

    rpath3 = rdir + fname_mp3
    lpath3 = ldir + fname_mp3
    lpathW = ldir + froot + ".wav"

    subprocess.run(["scp", rpath3, lpath3]) # get the .mp3 from remote host
    subprocess.run(["sox", lpath3, lpathW]) # convert it to .wav format

    subprocess.run([proc, lpathW]) # convert it to .wav format
    subprocess.run(["rm", lpath3, lpathW]) # remove files from tmp folder when done
    # sys.exit()
    
# ========================================================
dstring = time.strftime('%Y-%b-%d %H:%M:%S')
print("Run complete at %s." % dstring)

