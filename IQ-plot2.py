#!/home/john/anaconda3/envs/cv/bin/python

"""
Interpret event properties from spectrogram
Python3 code with scipy, numpy


J.Beale, Jan.27 2023
"""


import sys         # command-line arguments
import os
import subprocess # run scp, sox
import glob     # list of files in directory

from scipy.fft import fft, fftfreq, fftshift
from scipy.signal import cosine
from scipy import interpolate  # spline fit
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import cv2  # OpenCV for processing detected events
import matplotlib.patches as mpatches
import pandas as pd  # dataframes for output

import datetime # to show today's date
import time     # program run time
from skimage import morphology
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.color import label2rgb

# --------------------------------------------------------
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
# --------------------------------------------------------


N = 4096       # size of FFT
slices = 6566  # how many separate segments we do FFT on, in input function
specDur = 300  # duration of spectrogram in seconds
sliceDt = specDur / slices  # seconds per value (pixel) in spectrogram

mphPerHz = 1.0/72.05 # mph/Hz
mpsPerHz = 6.205E-3  # m/s per Hz
mphPermps = 2.23694  # mph per m/s
fpm = 3.28084  # how many feet in a meter
fpsPerMph = 1.46667  # convert mph to fps
# --------------------------------------------------------

# Calculate a spline fit to function y (units: mph) and graph it
def doSpline(yraw,ax,aTime):
    dirS = "going right"
    col="b" # blue
    if (np.average(yraw)<0):
        dirS = "going left" # eg. toward speed bump
        col="g" # green
        cols="b" # blue
    else:
        return (0,0,0,0,0,0)  # don't plot cars going right
    yraw = np.abs(yraw)        # units are mph
    y = np.concatenate([[yraw[0]*0.75],yraw]) # add a new first data point
    yDiff = np.diff(y)

    #nPoints = 10  # how many points on low-point curve

    x = range(0, len(y))
    #xs = range(0, len(y), int(len(y)/nPoints))
        
    knot_numbers = 5 # (was 4) how many interior knot points in our spline fit
    x_new = np.linspace(0, 1, knot_numbers+2)[1:-1] # not the endpoints
    q_knots = np.quantile(x, x_new) # evenly spaced x values 
    
    #print("length = %d" % len(x))
    w = np.ones(len(x)) # weights for spline fit
    w[1:20] = 0.3  # beginning has low weight
    results = interpolate.splrep(x, y, w, t=q_knots, s=1, full_output=True)
    t,c,k = results[0]
    #print("Fit results = %5.3f" % results[1])
    yfit = interpolate.BSpline(t,c,k)(x) # vector same length as 'y'
    #yfits = interpolate.BSpline(t,c,k)(xs)

    yfps = yfit * fpsPerMph # spline-fit. units are feet per second
    #yfps = y * fpsPerMph # raw data. units are feet per second
    yDist = np.cumsum(yfps * sliceDt)  # total feet travelled

    """
    col = next(ax._get_lines.prop_cycler)['color']
    ax.scatter(yDist, y, s=1, c=col)  # raw data of mph vs ft travelled
    ax.plot(yDist, yfit, '-', c=col)  # spline fit for mph vs ft travelled
    """
    
    # find max diff between (y - yfit) for all i>Ti where yDist[Ti] = 50 ft.
    # find Vmax on d=(50..135), Vmax on d=(135-200), and Vmin in range d=70..200
    iD1 = np.nonzero(yDist > 50)[0][0] # first index i where yDist[i] > 50
    iD1a = np.nonzero(yDist > 70)[0][0]
    iD2 = np.nonzero(yDist > 135)[0][0]
    iD3 = np.nonzero(yDist > 200)[0][0]
    #print("ID1: %d,%d,%d,%d" % (iD1,iD1a,iD2,iD3))
    maxyDif = np.abs(yDiff[iD1:iD3]).max()
    maxDif = np.abs(y[iD1:iD3] - yfit[iD1:iD3]).max()
    maxV1 = yfit[iD1:iD2].max()
    maxV2 = yfit[iD2:iD3].max()
    minV = yfit[iD1a:iD3].min()
    minVi = np.argmin(yfit[iD1a:iD3]) + iD1a  # index of minimum speed
    minVdist = yDist[minVi]  # distance travelled at time of min. speed
    dV = (maxV1+maxV2)/2 - minV
    """
    print("#V, T,D,V1,V2,minV, %.1f, %5.2f, %5.2f, %5.2f, %5.2f, %5.2f, %4.1f" %
          (aTime, maxyDif, maxDif, maxV1, maxV2, dV, minVdist))
    f.write("#V, T,D,V1,V2,minV, %.1f, %5.2f, %5.2f, %5.2f, %5.2f, %5.2f, %4.1f\n" %
          (aTime, maxyDif, maxDif, maxV1, maxV2, dV, minVdist))
    """
    # plt.show()
    return (maxyDif, maxDif, maxV1, maxV2, dV, minVdist)

# --------------------------------------------------------

def doOneImage(fname_in, ax):
    
    fbase = os.path.basename(fname_in) # base filename from full path
    epoch = string2epoch(fbase)  # Unix epoch time from filename
    image = cv2.imread(fname_in, cv2.IMREAD_GRAYSCALE) # get IQ spectrogram image
    #print("image array shape: ",np.shape(image)) # (1200, 6566, 3)
    

    """
    scale_percent = 25 # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    cv2.imshow("spectrogram", resized)
    cv2.waitKey(1)
    """
    
    # image size = (1200, 6567)  (freq x time)
    fRange = 600  # y coordinate of f=0 center frequency

    
    fs = 24000  # files recorded at 24000 sps
    T = 1.0 / fs # sample interval, (s)
    pMin0 = 0  # we don't know anymore the minimum intensity in spectrum
    pMax0 = 0  # we don't know anymore the minimum intensity in spectrum

    #image = imgB # size = (1200, 6567)  (freq x time)
    
    # =====================================================
    

    #fig, ax =  plt.subplots(2)  # set up matplotlib plot


    gap = 150  # don't sum near f=zero (rain noise)
    Pstack = np.sum(image[0:(fRange-gap),:], axis=0) # + freq, sum vertically 
    Nstack = np.sum(image[(fRange+gap):,:], axis=0) # - freq, sum vertically 
    # ax[2].plot(Pstack)  # vertical sums
    # ax[2].plot(Nstack)  # vertical sums

    #thresh = 22  # if there is no rain
    thresh = 45  # if there is rain (was 32)  # *Is this correct? *

    bw = morphology.closing(image > thresh, morphology.square(3)) # <= problem here?
    
    #cleared = bw
    mask = morphology.remove_small_objects(bw, 800, connectivity=2)

    # label image regions
    label_image = label(mask, background=0)
    props1 = regionprops(label_image)
    ecount = len(props1)
    #print("Found %d events" % ecount)
    image_label_overlay = label2rgb(label_image, image=image, bg_label=0)

    #ax[0].imshow(image_label_overlay)
    #ax[0].set_title('labelled image')
    # ax[1].axis('off')


    maskImg = (mask * 255).astype('uint8')
    imgOut = image * (mask > 0)  # image with non-event background masked off

    (fTotal, colTotal) = image.shape  # dimensions of image
    # Time Scale Factor =  0.045689  = 300 sec / 6565 pixel columns
    tScaleFac = specDur/colTotal       # convert horizontal image pixels to time (sec)
    eCount = 0
    #print("n, mph, time, duration")

    pd.options.display.float_format = '{:,.1f}'.format
    df = pd.DataFrame(columns = 
         ['epoch', 'dir', 'mphmax','mphavg','mphmin', 'stdAvg', 'area', 'len',
           'dist', 'dur', 'type',
           'xyD', 'fitD', 'maxV1', 'maxV2', 'dV', 'ftMinV'
          ])
    mphScale = (fs/N) * mphPerHz  # to get units of mph
    mpsScale = (fs/N) * mpsPerHz  # to get units of m/s    
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
            peaks = np.argmax(imgS, axis=0) # strongest freq. at each time step
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

              if ( (abs(mphMax) < 9) and (stdAvg > 2) ):
                  eType = 'ped' # by default, unless found otherwise

              # ax[2].plot(Svec)  # plot V vs T profile
              
              if (posFreq):
                  sig0 = Pstack[minc:maxc]
              else:
                  sig0 = Nstack[minc:maxc]
              sig = sig0 - (np.amax(sig0)*0.7)
              pkCount = np.sum(sig > 0) # width of peak ~ length of vehicle
              length = int(pkCount * tScaleFac * abs(mpsAvg) * fpm) # in feet
              if (mphAvg < 0):
                  length *= 0.75  # fudge factor: vehicles in far lane look longer
              if (length > 18):
                  eType = 'van'
              if (length > 40):  
                  eType = 'bus'
              if ( (abs(mphMax) > 9) and ((stdAvg > 0.75) or (mDist > 300)) ):
                  eType = 'odd' # probably combined events of some kind

              vMph = vVec * mphPermps  # vehicle speed vs time in mph
              #ax[1].plot(vMph) # show speed in mph
              #ax[1].plot(sig0)  # show vertical sums amplitude (sig.strength)
              #ax[1].grid("on")
              #print("mphAvg = %5.3f" % mphAvg)
              
              tIndex = int(typeDict[eType])
              aTime = epoch + eTime  # absolute epoch time = file start + offset
              direction = 0
              if (mphAvg > 0):
                  direction = 1
              mphMax = abs(mphMax)
              mphAvg = abs(mphAvg)
              mphMin = abs(mphMin)
              mDist = abs(mDist)
              #print(aTime)          

              if (mphAvg > 12.0) and (mDist > 200): # not slow or too brief
                  (maxyDif, maxDif, maxV1, maxV2, dV, minVdist) = doSpline(vMph,ax,aTime)
              else:
                  (maxyDif, maxDif, maxV1, maxV2, dV, minVdist) = (0, 0, 0, 0, 0, 0)

              # add this event to dataframe
              df.loc[eCount] = [aTime, direction, mphMax, mphAvg, mphMin, 
                                stdAvg, areaS, length, mDist, eDur, tIndex,
                                maxyDif, maxDif, maxV1, maxV2, dV, minVdist]

              eCount += 1



              # add visible box around detected event on graph
              #rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
              #                        fill=False, edgecolor='red', linewidth=1)
              # ax[0].add_patch(rect)


              
    df = df.sort_values(by=['epoch'])  # events in time order of appearance
    df = df.reset_index(drop=True)  # reset the index to be in sorted time order
            
    if (savePlot):        
        #out = image * (mask > 0)
        fbase = os.path.basename(fname_in)
        if ( fbase[-4:] == '.wav'):  # remove the .wav extension
            fbase = fbase[:-4]
        fname_out1= fdirOut + fbase + ".png"
        #fname_out1= fdirOut + fbase + ".jpg"  # jpeg saves space?
        #fname_out2= fdirOut + fbase + "_mask.png"
        cv2.imwrite(fname_out1,image) # detected image
        #cv2.imwrite(fname_out1,imgOut) # detected image
        #cv2.imwrite(fname_out2,maskImg) # peaks mask
    
    
    #cv2.imshow("spectrogram", image)
    #cv2.waitKey(0)
    return (pMin0,pMax0,df)

# -------------------------------------------------------------------

def findEvents(fname1, ax):
        
    (pMin0,pMax0,df) = doOneImage(fname1,ax) # returns Pandas DataFrame
    
    eCount = len(df.index)  # count of all events
    pedCount = ((df['type']==0)).sum()  # how many pedestrians?
    badCount = ((df['type']==9)).sum()  # how many bad-looking events?
    
    dstring = time.strftime('%H:%M:%S')

    f.write("# FILE, %s, %s, %.2E, %5.1f, %d, %d, %d\n" %
        (fname1, dstring, pMin0, pMax0, pedCount, badCount, eCount))
    print("# FILE, %s, %s, %.2E, %5.1f, %d, %d, %d" %
        (fname1, dstring, pMin0, pMax0, pedCount, badCount, eCount))
    #print("# FILE, %s, %s, %d" % (fname1, dstring, eCount))
    print(df.to_csv(sep=',', float_format =
                    '{: 6.2f}'.format, index=False, header=False))
    f.write(df.to_csv(sep=',', float_format =
                    '{: 6.2f}'.format, index=False, header=False))
    
    if (showPlot):
        fig.canvas.draw()
        fig.canvas.flush_events()
        #plt.show(block=False)

# ===================================================================    
# Main program starts here  
  
showPlot = True  # show spectrogram graphs
#savePlot = True
#showPlot = False  # show spectrogram graphs
savePlot = False


"""
n = len(sys.argv)
if (n < 2):
    print("%s Version 0.1" % sys.argv[0])
    print("%s: Missing argument. Must supply a filename to work on." % sys.argv[0])
    sys.exit()
    
fname1 = sys.argv[1]
"""

#fdir="C:/Users/beale/Documents/Audio/"
fdir="/home/john/Audio/images/"

    
#resultFile = "./DopplerD-Jan.csv"
resultFile = "/home/john/Audio/images/DLog12.csv"

fdirOut = "/home/john/Audio/images/doppler/outdir/" # save processed images

indir="/home/john/Audio/images/doppler/outdir/"  # input spectrogram images

gdir="/home/john/Audio/images/old/2023/"  # guide directory, list of .png files
# path to remote host directory with .mp3 files
rdir="john@john-Z83-4.local:/media/john/Seagate4GB/MINIX-John/Doppler1/old/"
ldir="/dev/shm/"  # local working directory


# plt.ion()
fig, ax = plt.subplots()
"""
ax.set_xlabel("calculated distance (ft)")
ax.set_ylabel("vehicle speed (mph)")
ax.set_title("Vehicle Speed Trajectory   (Raw + Fitted)  18-Jan-2023")
ax.set_xlim(left=50, right=250)
ax.set_ylim(bottom=5, top=35)
ax.grid("on")
"""

# header for output CSV table
cheader = "epoch, dir, max(mph), avg(mph), min(mph), std(px), area(px), "
cheader += "length(ft), distance(ft), duration, kind"

f = open(resultFile, 'w')
dstring = time.strftime('%Y-%b-%d %H:%M:%S')
f.write(cheader+"\n")  # start output file with column header line
f.write("# Run at %s\n" % dstring)

flist = glob.glob(indir + "DpD_*.png")  # list of input spectrogram image files
flist.sort() # let's do them in ascending order

#lastF = "/home/john/Audio/images/old/2023/DpD_2023-01-26_05-10-00.png"
#iStart = flist.index(lastF)
#print(len(flist), iStart)
#flist = flist[iStart:]  # truncate list to just this point

print("New length: ",len(flist))
#flist = flist[-1200:] # only the most recent N files
#print(len(flist))
print("First and last files:")
print("%s\n%s" % (flist[0],flist[-1]))

# sys.exit()  # DEBUG stop here

for fpath in flist:
    fpath1 = os.path.splitext(fpath)[0]
    froot = os.path.basename(fpath1) # base filename from full path

    findEvents(fpath, ax)  # find events in this spectrogram

    #sys.exit()

    
if (showPlot):
            plt.ioff()
            #plt.pause(0.0001)
            plt.show()

# ---------------------------------------------------------
# rough plot: velocity max around 70 ft and 200 ft, minimum around 135 ft
# based on 1200 files, DLog8.csv  20-Jan-2023

