"""
Plot Radar I/Q data on spectrogram
Python3 code with scipy, numpy, matplotlib
J.Beale, Jan.1 2023
"""

import sys
import numpy as np
from scipy import signal
from scipy.io import wavfile
import scipy.io
import matplotlib.pyplot as plt
import datetime # to show today's date


if (len(sys.argv) < 2):
    print("%s: Missing argument. Supply input.wav filename" % sys.argv[0])
    sys.exit()
    
fname_in = sys.argv[1]  # read in this wav file to numpy array

fs, datraw = scipy.io.wavfile.read(fname_in)
N,ch = datraw.shape # stereo wav file is Nx2 array of signed INT16
xR = datraw[:,:].astype(np.float32) / 65535.0
x = xR[:,0] + 1j * xR[:,1]  # make complex from 2 reals

today = datetime.date.today()
dstring = today.strftime('%Y-%b-%d')
fig, ax =  plt.subplots(2)
ax[0].set_title('YH-24G01    %s    plot: %s' % (fname_in,dstring))
ax[0].set(xlabel='time (s)', ylabel='frequency (Hz)')
ax[0].grid()

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
lm = -105 # clamp signal below this level in dB

p1 = 20*np.log10(Pxx[c-r:c+r,:]) # units of dB
p1f = np.flip(p1,0)  # flip array along 1st axis
pdif = p1f - (p1*0.55) # subtract off residual from inexact phase shift
p2 = np.maximum(lm,pdif)
ax[1].grid()
ax[1].imshow(p2,cmap='magma')
ax[1].set_title('spectrogram with residual opposite phase subtracted')
plt.show()
