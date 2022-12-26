# -*- coding: utf-8 -*-
"""
Plot Radar I/Q data on spectrogram
Python code with scipy, numpy, matplotlib
Dec 25 18:39:25 2022
@author: jbeale
"""

import numpy as np
from scipy import signal
from scipy.io import wavfile
import scipy.io
from scipy.fft import fftshift
import matplotlib.pyplot as plt

# wav file imports as Nx2 array of signed INT16

#fname_in="C:/Users/beale/Documents/Audio/Cornercube-run1.wav"
#fname_in="C:/Users/beale/Documents/Audio/Cornercube-run2.wav"
#fname_in="C:/Users/beale/Documents/Audio/DopplerIQ-walk3.wav"
fname_in="C:/Users/beale/Documents/Audio/DopplerIQ-walk2.wav"

fs, datraw = scipy.io.wavfile.read(fname_in)
N,ch = datraw.shape
xR = datraw.astype(np.float32) / 65535.0
x = xR[:,0] + 1j * xR[:,1]  # make complex from 2 reals

NFFT=2048
Pxx, freqs, bins, im = plt.specgram(x, NFFT=NFFT, Fs=fs, 
                                    noverlap=128, scale='dB',
                                    cmap='magma')
plt.axis((None, None, -3000, 3000))  # freq limits, Hz
plt.show()

c=int(NFFT/2) # center frequency (f=0)
r=100  # range on either size of f=0
A = 3
lm = -1.5 # clamp values below this level
plt.rc('figure', figsize=[46.82 * .5**(.5 * A), 33.11 * .5**(.5 * A)])
#s1 = np.log10(Pxx[c-r:c+r,38])
sum = np.sum(np.log10(Pxx[c-r:c+r,37:43]),axis=1)/6.0
sumf = np.flip(sum)

plt.grid()

p1 = np.log10(Pxx[c-r:c+r,:])
p1f = np.flip(p1,0)  # flip array along 1st axis
pdif = p1f - (p1*0.8)
p2 = np.maximum(lm,pdif)
plt.imshow(p2,cmap='magma')
plt.show

# colormap names   magma plasma  bone
# stackoverflow.com/questions/66539861/where-is-the-list-of-available-built-in-colormap-names
