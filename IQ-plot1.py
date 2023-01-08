"""
Plot Radar I/Q data on spectrogram
Python3 code with scipy, numpy, matplotlib
J.Beale, Jan.7 2023
"""

import numpy as np
from scipy import signal
from scipy.io import wavfile
import scipy.io
import matplotlib.pyplot as plt
import datetime # to show today's date
import cv2  # OpenCV for processing detected events
from skimage import morphology

# stereo wav file is Nx2 array of signed INT16

fdir = "C:/Users/beale/Documents/Audio/"

#fname1 = "DpD_2023-01-07_16-59-59.wav"
#fname1 = "DpD_2023-01-07_13-55-00.wav"
fname1 = "DpD_2023-01-07_16-30-00.wav"
#fname1 = "DpD_2023-01-07_22-55-00.wav"

fname_in= fdir + fname1

fs, datraw = scipy.io.wavfile.read(fname_in)
N,ch = datraw.shape
xR = datraw[:,:].astype(np.float32) / 65535.0
#xR = datraw[int(N*.75):,:].astype(np.float32) / 65535.0
#xR = datraw[int(N*0.1):int(N*0.3),:].astype(np.float32) / 65535.0
x = xR[:,0] + 1j * xR[:,1]  # make complex from 2 reals

today = datetime.date.today()
dstring = today.strftime('%Y-%b-%d')
fig, ax =  plt.subplots(2)
ax[0].set_title('YH-24G01    %s    plot: %s' % (fname1,dstring))
ax[0].set(xlabel='time (s)', ylabel='frequency (Hz)')
ax[0].grid()
#FFTsize=2048
FFTsize=int(4096*1)
fRange = 3000  # spectrogram displayed frequency range, in Hz
Pxx, freqs, bins, im = ax[0].specgram(x, NFFT=FFTsize, Fs=fs, 
                                    noverlap=3000, scale='dB',
                                    cmap='magma')
A = 3  # paper size (affects plot dpi)
plt.rc('figure', figsize=[46.82 * .5**(.5 * A), 33.11 * .5**(.5 * A)])
ax[0].axis((None, None, -fRange, fRange))  # freq limits, Hz

c=int(FFTsize/2) # center frequency (f=0)
r=int(250*2)  # bins on either size of f=0
lm = -180 # clamp signal below this level in dB
fMask = 12    # low-freq bin range to clamp 0

p1 = (Pxx[c-r:c+r,:]) # selected frequency region

pMin = np.amin(p1)
pMax = np.amax(p1)
pRange = pMax - pMin
p1[r-fMask:r+fMask,:]=pMin  # force low frequencies to minimum value

p1 = (p1 - pMin) / pRange # normalize range to (0..1)
#print("p1 Min = %5.3f  Max = %5.3f" % (pMin, pMax))

minV = 0.004  # minimum value for data array
minT = 1.5E-5   # clamp to this minimum threshold
p1f = np.flip(p1,0)  # flip array along 1st axis
pL = p1f - (p1*minV) # subtract off residual from inexact phase shift
pL = np.maximum(minT,pL) # clamp to positive definite

pMin = np.amin(pL)
pMax = np.amax(pL)
print("Min = %5.3f  Max = %5.3f" % (pMin, pMax))

pdif = 20*np.log10(pL) # units of dB
p2 = np.maximum(lm,pdif)
ax[1].grid()
ax[1].imshow(p2,cmap='magma')
ax[1].set_title('spectrogram with residual opposite phase subtracted')
plt.show()
# p2.shape = (1000,1311)
pMin = np.amin(p2)
pMax = np.amax(p2)
pRange = pMax - pMin
img = uint_img = np.array((p2-pMin)*255.0/pRange).astype('uint8')
#print(p2.shape)

print('Image Dimensions : ', img.shape)

size = 9  # horizontal motion blur
kernel_motion_blur = np.zeros((size, size))
kernel_motion_blur[int((size-1)/2),:] = np.ones(size)
kernel_motion_blur = kernel_motion_blur / size
imgB = cv2.filter2D(img, -1, kernel_motion_blur)

img = cv2.medianBlur(imgB,3)
sf = 0.5 # image display scale factor
dim = (int(img.shape[1]*sf) , int(img.shape[0]*sf) )
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
print('Resized Dimensions : ', resized.shape)
fname_out= fdir + fname1 + "_1.png"

imgTh = (resized > 30) # thresholded image as boolean array
mask = morphology.remove_small_objects(imgTh, 200, connectivity=5)

maskImg = (mask * 255).astype('uint8')
#cv2.imshow('mask',maskImg)
out = resized * (mask > 0)

cv2.imwrite(fname_out,out)
#cv2.imshow('image',out)
cv2.waitKey(0)

# colormap names   magma plasma  bone
# stackoverflow.com/questions/66539861/where-is-the-list-of-available-built-in-colormap-names
