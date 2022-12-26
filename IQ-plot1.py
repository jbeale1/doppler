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
fname_in="C:/Users/beale/Documents/Audio/DopplerIQ-walk3.wav"

fs, datraw = scipy.io.wavfile.read(fname_in)
N,ch = datraw.shape
xR = datraw.astype(np.float32) / 65535.0
x = xR[:,0] + 1j * xR[:,1]  # make complex from 2 reals

NFFT=4096
Pxx, freqs, bins, im = plt.specgram(x, NFFT=NFFT, Fs=fs, 
                                    noverlap=256, scale='dB',
                                    cmap='magma')
plt.axis((None, None, -1500, 1500))
plt.show()
