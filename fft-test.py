# Based on FFT example at
# https://docs.scipy.org/doc/scipy/tutorial/fft.html

from scipy.fft import fft, fftfreq, fftshift
from scipy.signal import cosine
from scipy.io import wavfile

import numpy as np
import matplotlib.pyplot as plt

N = 4096       # size of FFT
slices = 6000  # how many separate segments we do FFT on, in input function

fname_in = "../DpD_2023-01-10_17-15-00.wav"
fs, datraw = wavfile.read(fname_in) # load file with scipy
T = 1.0 / fs # sample interval, (s)
Nsamp,ch = datraw.shape

xR = datraw[:,:].astype(np.float32) / 65535.0
#xR = datraw[int(Nsamp*0.03):int(Nsamp*0.06),:].astype(np.float32) / 65535.0
y = xR[:,0] + 1j * xR[:,1]  # complex from 2 reals

Ntot = y.size
x = np.linspace(0.0, T*Ntot, Ntot, endpoint=False) # full time axis

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

    ywf = np.maximum(0.01,ywf)  # clamp small noise levels to fixed value
    ylog = 20*np.log10(np.abs(ywf)) # convert to dB, 20log10()
    #plt.semilogy(xf, np.abs(ywf), linewidth=0.7)
    #plt.plot(xf, ylog, linewidth=0.7)
    sgram[:,i] = ylog  # add this line to spectrogram image

# sgram is (N x slices) in size
# need to smooth over time and freq for reasonable plot
range = int(N/5)   # what part of full (0..N/2) frequency range to show
a=int((N/2)-range)
b=int((N/2)+range)
sWin = sgram[a:b,:]
plt.imshow(sWin, interpolation='none')
#plt.imshow(sgram, interpolation='bicubic')
plt.show()

