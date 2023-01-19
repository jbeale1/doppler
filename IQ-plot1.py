#!/home/john/anaconda3/envs/cv/bin/python

"""
Plot Radar I/Q data on spectrogram
Python3 code with scipy, numpy, matplotlib
Based on FFT example at
https://docs.scipy.org/doc/scipy/tutorial/fft.html

J.Beale, Jan.18 2023
"""


import sys         # command-line arguments
import os
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
sliceDt = 300 / slices  # seconds per value (pixel) in spectrogram

mphPerHz = 1.0/72.05 # mph/Hz
mpsPerHz = 6.205E-3  # m/s per Hz
mphPermps = 2.23694  # mph per m/s
fpm = 3.28084  # how many feet in a meter
fpsPerMph = 1.46667  # convert mph to fps
# --------------------------------------------------------

# Calculate a spline fit to function y (units: mph) and graph it
def doSpline(yraw,ax):
    dirS = "going right"
    col="b" # blue
    if (np.average(yraw)<0):
        dirS = "going left" # eg. toward speed bump
        col="g" # green
        cols="b" # blue
    else:
        return  # don't plot cars going right
    yraw = np.abs(yraw)        # units are mph
    y = np.concatenate([[yraw[0]*0.75],yraw])

    #nPoints = 10  # how many points on low-point curve

    x = range(0, len(y))
    #xs = range(0, len(y), int(len(y)/nPoints))
        
    knot_numbers = 5 # (was 4) how many interior knot points in our spline fit
    x_new = np.linspace(0, 1, knot_numbers+2)[1:-1] # not the endpoints
    q_knots = np.quantile(x, x_new) # evenly spaced x values 
    
    print("length = %d" % len(x))
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

    # ax.scatter(x,y,c=cols, s=1, label="raw data")
    #ax.plot(x, yfit, '-', c=col, label="spline fit")
    col = next(ax._get_lines.prop_cycler)['color']
    ax.scatter(yDist, y, s=1, c=col)
    ax.plot(yDist, yfit, '-', c=col)    
    
    # plt.show() 

# --------------------------------------------------------

def doOneImage(fname_in, ax):

    fbase = os.path.basename(fname_in) # base filename from full path
    epoch = string2epoch(fbase)  # Unix epoch time from filename

    fs, datraw = wavfile.read(fname_in) # load file with scipy
    T = 1.0 / fs # sample interval, (s)
    Nsamp,ch = datraw.shape
    
    xR = datraw[:,:].astype(np.float32) / 65535.0
    #xR = datraw[int(Nsamp*0.03):int(Nsamp*0.06),:].astype(np.float32) / 65535.0
    y = xR[:,0] + 1j * xR[:,1]  # complex from 2 reals
    
    Ntot = y.size
    # x = np.linspace(0.0, T*Ntot, Ntot, endpoint=False) # full time axis
    
    xf = fftfreq(N, T)  # calculate X axis vector in time units
    xf = fftshift(xf)   # shift FFT output to be symmetric around 0
    w = cosine(N) # windowing function
    
    sgram = np.zeros((N,slices))
    for i in range(slices):
        a = int(i * ((Ntot - N)/slices))
        b = a+N
        ySlice = y[a:b]  # segment of data to do this FFT on
        ywf = fft(ySlice*w)  # windowed FFT
        ywf = fftshift(ywf)  # put zero freq on center instead of edge
    
        #ywf = np.maximum(0.01,ywf)  # clamp small noise levels to fixed value
        #ylog = 20*np.log10(np.abs(ywf)) # convert to dB, 20log10()
        #sgram[:,i] = ylog  # add this line to spectrogram image
        sgram[:,i] = np.abs(ywf)  # add this line to spectrogram image
    
    # sgram is (N x slices) in size
    # need to smooth over time and freq for reasonable plot
    #fRange = int(N/5)   # what part of full (0..N/2) frequency range to show
    fRange = 600
    a=int((N/2)-fRange)
    b=int((N/2)+fRange)
    p1 = sgram[a:b,:]
    
    """
    plt.imshow(20*np.log10(p1), interpolation='none')
    #plt.imshow(sgram, interpolation='bicubic')
    plt.show()
    """
    
    # convert matrix to OpenCV plot
    
    fMask = 3
    pMin0 = np.amin(p1)
    pMax0 = np.amax(p1)
    # pMax = np.maximum(5,pMax0) # don't autoscale noise up too high
    pMax = 6    # typical value
    pMin = 1E-5 # somewhat higher than usual
    pRange = pMax - pMin
    p1[fRange-fMask:fRange+fMask,:]=pMin  # mask off low frequencies to min value
    
    minV = 0.06
    minT = 4.0E-3   # clamp to this minimum threshold
    
    p1 = (p1 - pMin) / pRange # scale to reasonable values
    p1 = np.clip(p1,0.0,1.0)   # clamp to range (0..1)
    
    p1f = np.flip(p1,0)  # flip array along 1st axis
    pL = p1f - (p1*minV) # subtract off residual from inexact phase shift
    pL = np.maximum(minT,pL) # clamp to positive definite
    
    p2 = 20*np.log10(pL) # units of dB
    
    #plt.imshow(p2, interpolation='none')
    #plt.imshow(sgram, interpolation='bicubic')
    #plt.show()
    
    pMin = np.amin(p2)
    pMax = np.amax(p2)
    pRange = pMax - pMin
    
    # plotIQ_1 setup: (fbins 1200, timebins 6566)
    
    # image is 6566 x 1638 pixels
    img = np.array((p2-pMin)*255.0/pRange).astype('uint8')
    
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
    
    # =====================================================
    

    #fig, ax =  plt.subplots(2)  # set up matplotlib plot


    gap = 150  # don't sum near f=zero (rain noise)
    Pstack = np.sum(image[0:(fRange-gap),:], axis=0) # + freq, sum vertically 
    Nstack = np.sum(image[(fRange+gap):,:], axis=0) # - freq, sum vertically 
    # ax[2].plot(Pstack)  # vertical sums
    # ax[2].plot(Nstack)  # vertical sums

    #thresh = 22  # if there is no rain
    thresh = 45  # if there is rain (was 32)  # *Is this correct? *

    bw = morphology.closing(image > thresh, morphology.square(3))
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
    tScaleFac = (Nsamp/fs)/colTotal       # convert horizontal image pixels to time (sec)
    eCount = 0
    #print("n, mph, time, duration")

    pd.options.display.float_format = '{:,.1f}'.format
    df = pd.DataFrame(columns = 
         ['epoch', 'dir', 'mphmax','mphavg','mphmin', 'stdAvg', 'area', 'len',
           'dist', 'dur', 'type'])
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
              print("mphAvg = %5.3f" % mphAvg)
              
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
              if (mphAvg > 12.0) and (mDist > 150) and (mDist < 280): # not slow, brief, or multiple
                  doSpline(vMph,ax)
              
              # add this event to dataframe
              df.loc[eCount] = [aTime, direction, mphMax, mphAvg, mphMin, 
                                stdAvg, areaS, length, mDist, eDur, tIndex]
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
        fname_out1= fdirOut + fbase + "_3.png"
        fname_out2= fdirOut + fbase + "_mask.png"
        cv2.imwrite(fname_out1,image) # detected image
        #cv2.imwrite(fname_out1,imgOut) # detected image
        cv2.imwrite(fname_out2,maskImg) # peaks mask
    
    
    #cv2.imshow("spectrogram", image)
    #cv2.waitKey(0)
    return (pMin0,pMax0,df)

# ===================================================================    
# Main program starts here  
  
showPlot = True  # show spectrogram graphs
#savePlot = True
#showPlot = False  # show spectrogram graphs
savePlot = False

fdirOut = "./"
#wdir="C:/Users/beale/Documents/Audio/"
#fname_in = wdir + "DpD_2023-01-14_16-55-00.wav"
#fname_in = wdir + "DpD_2023-01-14_12-35-00.wav"  
#doOneImage(fname_in)

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
#fname1 = "DpD_2023-01-14_16-55-00"  # 9 events
#fname1 = "DpD_2023-01-14_12-35-00"  # 6 events
#fname1 = "DpD_2023-01-02_11-05-00" # 11 events (reflector-knee?)
#fname1 = "DpD_2023-01-03_03-45-00" # 0 (noise prob.)
#fname1 = "DpD_2023-01-03_04-10-00"  # big signal, no events
# fname1 = "DpD_2023-01-06_14-25-00" # 5 events, 3 odd; big peak
#fname1 = "DpD_2023-01-16_17-14-59" # 6 ev: JPB 2 walk, 2 jog
#fname1 = "DpD_2023-01-16_19-45-00"  # 7 events, JPB 2 walk, 2 jog
#fname1 = "DpD_2023-01-16_20-45-00"  # 5 events, JPB 2 walk, 2 jog
#fname1 = "DpD_2023-01-17_16-29-59" # 7 events
#fnames = ["DpD_2023-01-17_15-30-00", # 7 events
#          "DpD_2023-01-17_16-29-59", # 7 events
#          "DpD_2023-01-14_16-55-00",  # 9 events
#          "DpD_2023-01-14_12-35-00",  # 6 events
#          ]
"""
"""

fnames = [
          "DpD_2023-01-10_17-15-00",
          "DpD_2023-01-18_07-55-00",
          "DpD_2023-01-18_08-30-00",
          "DpD_2023-01-18_08-35-00",
          "DpD_2023-01-18_08-40-00",
          "DpD_2023-01-18_08-45-00",
          "DpD_2023-01-18_09-10-00",
          "DpD_2023-01-18_09-34-59",
          "DpD_2023-01-18_12-10-00",
          "DpD_2023-01-18_12-15-00",
          "DpD_2023-01-18_12-35-00",
          "DpD_2023-01-18_12-40-00",
          "DpD_2023-01-18_14-15-00",
          "DpD_2023-01-18_14-20-00",
          "DpD_2023-01-18_17-09-59"
        ]

    
#resultFile = "./DopplerD-Jan.csv"
resultFile = "/home/john/Audio/images/DLog7.csv"

plt.ion()
fig, ax = plt.subplots()
ax.set_xlabel("calculated distance (ft)")
ax.set_ylabel("vehicle speed (mph)")
ax.set_title("Vehicle Speed Trajectory   (Raw + Fitted)  18-Jan-2023")
ax.set_xlim(left=50, right=250)
ax.set_ylim(bottom=5, top=35)

ax.grid("on")

for fname1 in fnames:
    fname1 = fdir + fname1
    if ( fname1[-4:] != '.wav'):
        fname1 += '.wav'
    
    with open(resultFile, 'a') as f:
        # df = doOneImage(fname1) 
        
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
                        '{: 6.1f}'.format, index=False, header=False))
        f.write(df.to_csv(sep=',', float_format =
                        '{: 6.1f}'.format, index=False, header=False))
        
        if (showPlot):
            fig.canvas.draw()
            fig.canvas.flush_events()
            #plt.show(block=False)

    
if (showPlot):
            plt.ioff()
            #plt.pause(0.0001)
            plt.show()

# ---------------------------------------------------------
