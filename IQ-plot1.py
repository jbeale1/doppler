"""
Plot Radar I/Q data on spectrogram
Python3 code with scipy, numpy, matplotlib
J.Beale, Dec 26 2022
"""

import numpy as np
from scipy import signal
from scipy.io import wavfile
import scipy.io
import matplotlib.pyplot as plt

# stereo wav file is Nx2 array of signed INT16

#fname_in="C:/Users/beale/Documents/Audio/Cornercube-run1.wav"
#fname_in="C:/Users/beale/Documents/Audio/Cornercube-run2.wav"
#fname_in="C:/Users/beale/Documents/Audio/DopplerIQ-walk3.wav"
fname_in="C:/Users/beale/Documents/Audio/DopplerIQ-walk2.wav"
#fname_in="C:/Users/beale/Documents/Audio/Robotwalk1.wav"

fs, datraw = scipy.io.wavfile.read(fname_in)
N,ch = datraw.shape
xR = datraw.astype(np.float32) / 65535.0
x = xR[:,0] + 1j * xR[:,1]  # make complex from 2 reals

fig, ax =  plt.subplots(2)
ax[0].set_title('YH-24G01 24GHz IQ doppler   Walking 14 steps away (11 m), then back')
ax[0].set(xlabel='time (s)', ylabel='frequency (Hz)')
FFTsize=2048
fRange = 2000  # spectrogram displayed frequency range, in Hz
Pxx, freqs, bins, im = ax[0].specgram(x, NFFT=FFTsize, Fs=fs, 
                                    noverlap=700, scale='dB',
                                    cmap='magma')
A = 3  # paper size (affects plot dpi)
plt.rc('figure', figsize=[46.82 * .5**(.5 * A), 33.11 * .5**(.5 * A)])
ax[0].axis((None, None, -fRange, fRange))  # freq limits, Hz

c=int(FFTsize/2) # center frequency (f=0)
r=80  # bins on either size of f=0
lm = -60 # clamp signal below this level in dB

p1 = 20*np.log10(Pxx[c-r:c+r,:]) # units of dB
p1f = np.flip(p1,0)  # flip array along 1st axis
pdif = p1f - (p1*0.65) # subtract off residual from inexact phase shift
p2 = np.maximum(lm,pdif)
ax[1].grid()
ax[1].imshow(p2,cmap='magma')
ax[1].set_title('spectrogram with residual opposite phase subtracted')
plt.show()

# colormap names   magma plasma  bone
# stackoverflow.com/questions/66539861/where-is-the-list-of-available-built-in-colormap-names
