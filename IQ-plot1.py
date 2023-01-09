"""
Plot Radar I/Q data on spectrogram
Python3 code with scipy, numpy, matplotlib
J.Beale, Jan.8 2023
"""

import numpy as np
from scipy import signal
from scipy.io import wavfile
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import datetime # to show today's date
import cv2  # OpenCV for processing detected events
from skimage import morphology
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.color import label2rgb

# stereo wav file is Nx2 array of signed INT16

fdir = "C:/Users/beale/Documents/Audio/"

fname1 = "DpD_2023-01-07_16-30-00.wav"  # 6.5 cars, 1 ped
#fname1 = "DpD_2023-01-07_22-55-00.wav"   # 2 cars
#fname1 = "DpD_2023-01-08_05-45-00.wav"  # only rain

#fname1 = "DpD_2023-01-08_07-34-59.wav"   # 1 car
#fname1 = "DpD_2023-01-08_09-45-00.wav"  # lots of rain
#fname1 = "DpD_2023-01-08_18-00-00.wav"  # 5 cars
#fname1 = "DpD_2023-01-08_17-20-00.wav"  # 8 cars
#fname1 = "DpD_2023-01-08_20-15-00.wav"   # 5 cars + rain

fname_in= fdir + fname1

DopFreqScale = 1.0/72.05 # mph/Hz  if radar is 24.15 GHz

fs, datraw = scipy.io.wavfile.read(fname_in) # fs = 24000 kHz
N,ch = datraw.shape
xR = datraw[:,:].astype(np.float32) / 65535.0
#xR = datraw[int(N*.75):,:].astype(np.float32) / 65535.0
#xR = datraw[int(N*0.1):int(N*0.3),:].astype(np.float32) / 65535.0
x = xR[:,0] + 1j * xR[:,1]  # make complex from 2 reals

today = datetime.date.today()
dstring = today.strftime('%Y-%b-%d')
print("%s read at %s" % (fname1,dstring))

fig, ax =  plt.subplots(2)
ax[0].set_title('YH-24G01    %s    plot: %s' % (fname1,dstring))
ax[0].set(xlabel='time (s)', ylabel='frequency (Hz)')
ax[0].grid()
#FFTsize=2048
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
fMask = 30    # low-freq bin range to clamp 0

# print("Pxx 0,0 value: ", Pxx[0,0], Pxx[4095,0])
# max freq = fs/2 => +FFTsize in matrix
# 0 freq = 0 => FFTsize/2
# -fs/2 => 0 in matrix
#print("Pxx size ", Pxx.shape) # (4096, 6566) 4096 => fs/2
p1 = (Pxx[c-fRange:c+fRange,:]) # selected frequency region

pMin = np.amin(p1)
pMax = np.amax(p1)
pRange = pMax - pMin
p1[fRange-fMask:fRange+fMask,:]=pMin  # force low frequencies to minimum value

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
Hsize = 3  # horiz. motion blur
kernel_motion_blur = np.zeros((Vsize, Vsize))
kernel_Hmotion_blur = np.zeros((Hsize, Hsize))

kernel_Hmotion_blur[int((Hsize-1)/2),:] = np.ones(Hsize) # H motion
kernel_Hmotion_blur /= Hsize

kernel_motion_blur[:,int((Vsize-1)/2)] = np.ones(Vsize) # V motion
kernel_motion_blur /= Vsize
imgT = cv2.filter2D(img, -1, kernel_motion_blur)
imgB = cv2.filter2D(imgT, -1, kernel_Hmotion_blur)


fname_out1= fdir + fname1 + "_1.png"
fname_out2= fdir + fname1 + "_2.png"

image = imgB # size = (1200, 6567)  (freq x time)

#thresh = threshold_otsu(image)
#print("otsu threshold = %d" % thresh)
#if (thresh < 25):  # reasonable minimum
thresh = 25

bw = morphology.closing(image > thresh, morphology.square(3))
#cleared = clear_border(bw)  # clear artifacts at border
cleared = bw
mask = morphology.remove_small_objects(cleared, 800, connectivity=2)

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
tScaleFac = (N/fs)/colTotal       # convert image pixels to time (sec)
eCount = 0
for region in props1:
    if region.area >= 2000:  # draw a bounding rectangle
        eCount += 1
        fCenter = region.centroid[0]                                  
        minr, minc, maxr, maxc = region.bbox
        imgS = imgOut[minr:maxr,minc:maxc] # image selected region
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=1)
        ax[1].add_patch(rect)
        if (maxr < fRange): # positive frequency half of plot
          peaks = np.argmax(imgS, axis=0) # max along 1st axis
          # print(peaks.size, peaks)
          peakP = minr + np.amin(peaks) # index of highest frequency (if positive)
          fPk = (fRange - peakP) * ((fs/2)/(FFTsize/2))
          mph = DopFreqScale * fPk
          eTime = maxc * tScaleFac  # time in seconds
          eDur = (maxc-minc) * tScaleFac
          print("%d, %5.2f mph, at %5.1f s, dur= %5.1f s" 
                % (eCount, mph, eTime, eDur))
          
plt.show()
print("Total events: %d" % eCount)

maskImg = (mask * 255).astype('uint8')

out = image * (mask > 0)

cv2.imwrite(fname_out1,imgOut) # detected image
cv2.imwrite(fname_out2,maskImg) # peaks mask

# colormap names   magma plasma  bone
# stackoverflow.com/questions/66539861/where-is-the-list-of-available-built-in-colormap-names
