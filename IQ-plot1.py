#!/home/john/anaconda3/envs/cv/bin/python

"""
Plot Radar I/Q data on spectrogram
Python3 code with scipy, numpy, matplotlib
J.Beale, Jan.11 2023
"""


import sys         # command-line arguments
import os          # find basename from full file path
import subprocess # run scp, sox
import glob     # list of files in directory
import gc         # memory getting filled up

import numpy as np
from scipy import signal
from scipy.io import wavfile
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd  # dataframes for output
# import librosa # only for reading .mp3 files

import datetime # to show today's date
import time     # epoch timestamp from string conversion
import cv2  # OpenCV for processing detected events
from skimage import morphology
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.color import label2rgb

# Convert filename in form "xxx_YYYY-MM-DD_hh-mm-ss.xxx"
# to Unix epoch time, assuming filename has local time (YMD hms)
def string2epoch(fname):
    try:
        dateS,timeS = fname[4:23].split('_')
        (Y,M,D) = [int(i) for i in dateS.split('-')]
        (h,m,s) = [int(i) for i in timeS.split('-')]
        date_time = datetime.datetime(Y,M,D,h,m,s)
        epoch = (time.mktime(date_time.timetuple()))
        if (s == 59):  # assume was just a few ms before :00
            epoch += 1
    except:
        epoch=0
    return epoch

# Use spectrogram of doppler I,Q data to find objects and speeds
def doOneImage(fname_in):

    # Radar Ft = 24.15 GHz  Fd=2*Ft*(v/c) v=Fd * c/(2*Ft)   (c/2Ft) = 6.205 (m/s)/kHz = 13.88 mph/kHz
    mphPerHz = 1.0/72.05 # mph/Hz
    mpsPerHz = 6.205E-3  # m/s per Hz
    fpm = 3.28084  # how many feet in a meter

    # audio files saved with 24000 Hz sample rate
    #datraw,fs = librosa.load(fname_in, sr=None, mono=False) # preserve sample rate

    fbase = os.path.basename(fname_in) # base filename from full path
    epoch = string2epoch(fbase)  # Unix epoch time from filename

    fs, datraw = scipy.io.wavfile.read(fname_in) # load file with scipy
    N,ch = datraw.shape
    xR = datraw[:,:].astype(np.float32) / 65535.0
    #xR = datraw[int(N*.75):,:].astype(np.float32) / 65535.0
    #xR = datraw[int(N*0.1):int(N*0.3),:].astype(np.float32) / 65535.0
    x = xR[:,0] + 1j * xR[:,1]  # make complex from 2 reals

    fig, ax =  plt.subplots(3)
    ax[0].set_title('YH-24G01    %s' % (fname_in))
    ax[0].set(xlabel='time (s)', ylabel='frequency (Hz)')
    ax[0].grid()

    FFTsize=int(4096*1)
    fRange = 4000  # spectrogram displayed frequency range, in Hz
    Pxx, freqs, bins, im = ax[0].specgram(x, NFFT=FFTsize, Fs=fs, 
                                        noverlap=3000, scale='dB',
                                        cmap='magma')
    A = 3  # paper size (affects plot dpi)
    plt.rc('figure', figsize=[46.82 * .5**(.5 * A), 33.11 * .5**(.5 * A)])
    ax[0].axis((None, None, -fRange, fRange))  # freq limits, Hz

    c=int(FFTsize/2) # center frequency (f=0)
    fRange=int(300*2)  # frequency bins on either size of f=0
    lm = -180 # clamp signal below this level in dB
    fMask = 5    # low-freq bin range to clamp to 0

    # print("Pxx 0,0 value: ", Pxx[0,0], Pxx[4095,0])
    # max freq = fs/2 => +FFTsize in matrix
    # 0 freq = 0 => FFTsize/2
    # -fs/2 => 0 in matrix
    #print("Pxx size ", Pxx.shape) # (4096, 6566) 4096 => fs/2
    p1 = (Pxx[c-fRange:c+fRange,:]) # selected frequency region

    pMin = np.amin(p1)
    pMax = np.amax(p1)
    pRange = pMax - pMin
    p1[fRange-fMask:fRange+fMask,:]=pMin  # mask off low frequencies to minimum value

    p1 = (p1 - pMin) / pRange # normalize range to (0..1)
    #print("p1 Min = %5.3f  Max = %5.3f" % (pMin, pMax))

    minV = 0.004  # minimum value for data array
    minT = 1.5E-5   # clamp to this minimum threshold
    p1f = np.flip(p1,0)  # flip array along 1st axis
    pL = p1f - (p1*minV) # subtract off residual from inexact phase shift
    pL = np.maximum(minT,pL) # clamp to positive definite

    pMin = np.amin(pL)
    pMax = np.amax(pL)
    # print("Min = %5.3f  Max = %5.3f" % (pMin, pMax))

    pdif = 20*np.log10(pL) # units of dB
    p2 = np.maximum(lm,pdif)
    # ax[1].grid()
    # ax[1].imshow(p2,cmap='magma')
    # ax[1].set_title('spectrogram with residual opposite phase subtracted')

    # p2.shape = (1000,1311)
    pMin = np.amin(p2)
    pMax = np.amax(p2)
    pRange = pMax - pMin
    img = uint_img = np.array((p2-pMin)*255.0/pRange).astype('uint8')

    # print('Image Dimensions : ', img.shape) # (fbins 1200, timebins 6566)

    Vsize = 7  # vertical motion blur length
    Hsize = 5  # horiz. motion blur
    kernel_motion_blur = np.zeros((Vsize, Vsize))
    kernel_Hmotion_blur = np.zeros((Hsize, Hsize))

    kernel_Hmotion_blur[int((Hsize-1)/2),:] = np.ones(Hsize) # H motion
    kernel_Hmotion_blur /= Hsize

    kernel_motion_blur[:,int((Vsize-1)/2)] = np.ones(Vsize) # V motion
    kernel_motion_blur /= Vsize
    imgT = cv2.filter2D(img, -1, kernel_motion_blur)
    imgB = cv2.filter2D(imgT, -1, kernel_Hmotion_blur)

    image = imgB # size = (1200, 6567)  (freq x time)

    gap = 150  # don't sum near f=zero (rain noise)
    Pstack = np.sum(image[0:(fRange-gap),:], axis=0) # + freq, sum vertically 
    Nstack = np.sum(image[(fRange+gap):,:], axis=0) # - freq, sum vertically 
    # ax[2].plot(Pstack)  # vertical sums
    # ax[2].plot(Nstack)  # vertical sums

    #thresh = 22  # if there is no rain
    thresh = 45  # if there is rain (was 32)

    bw = morphology.closing(image > thresh, morphology.square(3))
    #cleared = bw
    mask = morphology.remove_small_objects(bw, 800, connectivity=2)

    # label image regions
    label_image = label(mask, background=0)
    props1 = regionprops(label_image)
    ecount = len(props1)
    #print("Found %d events" % ecount)
    image_label_overlay = label2rgb(label_image, image=image, bg_label=0)

    ax[1].imshow(image_label_overlay)
    ax[1].set_title('labelled image')
    # ax[1].axis('off')


    maskImg = (mask * 255).astype('uint8')
    imgOut = image * (mask > 0)  # image with non-event background masked off

    (fTotal, colTotal) = image.shape  # dimensions of image
    # Time Scale Factor =  0.045689  = 300 sec / 6565 pixel columns
    tScaleFac = (N/fs)/colTotal       # convert horizontal image pixels to time (sec)
    eCount = 0
    #print("n, mph, time, duration")

    df = pd.DataFrame(columns = 
         ['epoch', 'mphmax','mphavg','mphmin', 'stdAvg', 'area', 'len',
           'dist', 'dur', 'type'])
    mphScale = (fs/FFTsize) * mphPerHz  # to get units of mph
    mpsScale = (fs/FFTsize) * mpsPerHz  # to get units of m/s    
    typeDict = {'ped':0, 'car':1, 'van':2, 'bus':3, 'odd':9} # all the events we know

    for region in props1:
        #print(region.area)
        area = region.area
        areaS = (area/10)  # scaled down for easier display
        if area >= 700:  # was 1200 draw a bounding rectangle
            eType = 'car' # by default, unless found otherwise
            fCenter = region.centroid[0]                                  
            minr, minc, maxr, maxc = region.bbox
            imgS = imgOut[minr:maxr,minc:maxc] # image selected region

            eDur = (maxc-minc) * tScaleFac  # event duration in units of seconds
            # apt = area / eDur # total pixel area per unit time
            peaks = np.argmax(imgS, axis=0) # max along 1st axis
            vVec = (fRange-(minr + peaks)) * mpsScale # velocity in m/s at each moment in time
            mDist = np.trapz(vVec) * tScaleFac * fpm  # total distance travelled (in feet)
            avgP = minr + np.average(peaks) # average of all values
            pkSz = peaks.size  # number of elements in vector
            pk1 = peaks[0:int(pkSz/3)]
            pk2 = peaks[int(pkSz/3):2*int(pkSz/3)]
            pk3 = peaks[2*int(pkSz/3):]
            std1 = np.std(np.diff(pk1[0::2]))    # measure of stability of velocity
            std2 = np.std(np.diff(pk2[0::2]))    # measure of stability of velocity
            std3 = np.std(np.diff(pk3[0::2]))    # measure of stability of velocity
            #peakP = minr + np.amin(peaks) # index of highest frequency (if pos.)
            Svec = abs(fRange - (minr + peaks)) * mphScale
            posFreq = False # just a default
            
            # print(peaks.size, peaks)
            if (minr < fRange): # positive frequency half of plot
              maxP = minr + np.amin(peaks) # index of highest frequency (if pos.)                  
              minP = minr + np.amax(peaks) # index of lowest frequency (if pos.)                  
              eTime = maxc * tScaleFac  # time in seconds
              stdAvg = (std1+std2)/2
              posFreq = True
            if (maxr > fRange): # negative frequency, bottom half of plot
              maxP = minr + np.amax(peaks) # index of highest frequency (if neg.)                            
              minP = minr + np.amin(peaks) # index of highest frequency (if neg.)                            
              eTime = minc * tScaleFac  # start time in seconds
              stdAvg = (std2+std3)/2
              posFreq = False

            mphMax = (fRange - maxP) * mphScale
            mpsMax = (fRange - maxP) * mpsScale
            mphAvg = (fRange - avgP) * mphScale
            mpsAvg = (fRange - avgP) * mpsScale
            mphMin = (fRange - minP) * mphScale
            #print(mpsAvg)
            stdAvg *= 5.0/abs(mphAvg) # scaled by avg speed
            #mphStd = std * mphScale  # stdAvg was 8
            if ((eDur > 2) and (stdAvg < 8) and (abs(mphMax) > 2) and (abs(mphMax) > 1.4) and 
                  ((abs(mphMax) > 5) or (eDur > 5))): # skip any slow events if too short
              if ( (abs(mphMax) < 5) and (stdAvg > 2) ):
                  eType = 'ped' # by default, unless found otherwise

              # ax[2].plot(Svec)  # plot V vs T profile
              
              if (posFreq):
                  sig = Pstack[minc:maxc]
              else:
                  sig = Nstack[minc:maxc]
              sig = sig - (np.amax(sig)*0.7)
              pkCount = np.sum(sig > 0) # width of peak ~ length of vehicle
              length = int(pkCount * tScaleFac * abs(mpsAvg) * fpm) # in feet
              if (mphAvg < 0):
                  length *= 0.75  # fudge factor: vehicles in far lane look longer
              if (length > 18):
                  eType = 'van'
              if (length > 40):  
                  eType = 'bus'
              if ( (abs(mphMax) > 5) and ((stdAvg > 0.75) or (mDist > 300)) ):
                  eType = 'odd' # probably combined events of some kind

              ax[2].plot(sig)  # vertical sums
              ax[2].grid()              
              tIndex = typeDict[eType]
              aTime = epoch + eTime  # absolute epoch time = file start + offset
              #print(aTime)

              # add this event to dataframe
              df.loc[eCount] = [aTime, mphMax, mphAvg, mphMin, stdAvg, areaS, length,
                                mDist, eDur, tIndex]
              eCount += 1
              # add visible box around detected event on graph
              rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=1)
              ax[1].add_patch(rect)

              

    df = df.sort_values(by=['epoch'])  # events in time order of appearance
    df = df.reset_index(drop=True)  # reset the index to be in sorted time order

    if (showPlot):
        plt.show()

    if (savePlot):
        maskImg = (mask * 255).astype('uint8')
        out = image * (mask > 0)
        if ( fbase[-4:] == '.wav'):  # remove the .wav extension
            fbase = fbase[:-4]
        fname_out1= fdirOut + fbase + "_1.png"
        fname_out2= fdirOut + fbase + "_2.png"
        cv2.imwrite(fname_out1,imgOut) # detected image
        cv2.imwrite(fname_out2,maskImg) # peaks mask

    #del datraw,Pxx,xR,x,fig,ax,p1,freqs,bins,im
    #gc.collect()  # done with all the intermediate files

    return df

# ======================================================================
# Main program here

# fdir = "C:/Users/beale/Documents/Audio/"

fdirOut = "./"
showPlot = False  # show spectrogram graphs
savePlot = False  # save thresholded spectrogram images

n = len(sys.argv)
#print("Total arguments passed:", n)
if (n < 2):
    print("%s Version 0.1" % sys.argv[0])
    print("%s: Missing argument. Must supply a filename to work on." % sys.argv[0])
    sys.exit()
    
fname1 = sys.argv[1]

#fname1 = "DpD_2023-01-07_16-30-00"
#fname1 = "DpD_2023-01-10_18-40-00"
#fname1 = "DpD_2023-01-08_20-15-00"  # rain
#fname1 = "DpD_2023-01-08_17-20-00"  # rain

#fname1 = fdir + fname1
#if ( fname1[-4:] != '.wav'):
#    fname1 += '.wav'

resultFile = "/home/john/Audio/images/DopplerD-Jan.csv"

"""
gdir="/home/john/Audio/images/old/2023/"  # guide directory, list of .png files
# path to remote host directory with .mp3 files
rdir="john@john-Z83-4.local:/media/john/Seagate4GB/MINIX-John/Doppler1/old/"
ldir="/dev/shm/"  # local working directory

cheader = "epoch, max(mph), avg(mph), min(mph), std(px), area(px), "
cheader += "length(ft), distance(ft), duration, kind"

flist = glob.glob(gdir + "DpD_*.png")  # list of all known mp3 files
flist.sort() # let's do them in ascending order
"""
with open(resultFile, 'a') as f:
    #f.write(cheader+"\n")  # start output file with column header line
    df = doOneImage(fname1) # returns events in Pandas DataFrame
    
    eCount = len(df.index)  # count of all events
    today = datetime.date.today()
    dstring = today.strftime('%Y-%b-%d')

    f.write("# FILE, %s, %s, %d\n" % (fname1, dstring, eCount))
    print("# FILE, %s, %s, %d" % (fname1, dstring, eCount))
    print(df.to_csv(sep=',', float_format =
                    '{: 6.1f}'.format, index=False, header=False))
    f.write(df.to_csv(sep=',', float_format =
                    '{: 6.1f}'.format, index=False, header=False))

    #f.flush()  # because we're impatient to check results

"""
    for fpath in flist:
        fpath1 = os.path.splitext(fpath)[0]
        froot = os.path.basename(fpath1) # base filename from full path
        #print(froot)  # of form: "DpD_2023-01-11_14-05-00"

        # froot = "DpD_2023-01-11_20-55-00"
        fname_mp3 = froot + ".mp3"

        rpath3 = rdir + fname_mp3
        lpath3 = ldir + fname_mp3
        lpathW = ldir + froot + ".wav"

        # print(rpath3, lpath3, lpathW)
        subprocess.run(["scp", rpath3, lpath3]) # get the .mp3 from remote host
        subprocess.run(["sox", lpath3, lpathW]) # convert it to .wav format
        #sys.exit()  # just for testing here
        
        # process this audio file, find events
        fname1 = froot  # no path, no extension
        df = doOneImage(lpathW) # returns events in Pandas DataFrame
        
        eCount = len(df.index)  # count of all events
        today = datetime.date.today()
        dstring = today.strftime('%Y-%b-%d')
        f.write("# FILE, %s, %s, %d\n" % (fname1, dstring, eCount))
        print("# FILE, %s, %s, %d" % (fname1, dstring, eCount))
        print(df.to_csv(sep=',', float_format =
                        '{: 6.1f}'.format, index=False, header=False))
        f.write(df.to_csv(sep=',', float_format =
                        '{: 6.1f}'.format, index=False, header=False))

        f.flush()  # because we're impatient to check results
        subprocess.run(["rm", lpath3, lpathW]) # remove input files from ramdisk

        del df
        gc.collect()  # done with all the intermediate files

"""

"""
pd.set_option('display.max_columns', None) # show all columns in dataframe
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', None)
pd.options.display.float_format = '{:,.1f}'.format
print(df)
"""

