import sys
from scipy import *
from numpy import *
from pylab import *
from ipywidgets import *
import scipy.io.wavfile as wav
import soundfile as sf
import scipy.signal as s
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import stem

def parabolic(f, x):
    xv = 0.5 * (f[x-1]-f[x+1]) / (f[x-1] - 2*f[x] + f[x+1]) + x
    yv = f[x] - 0.25 * (f[x-1] - f[x+1]) * (xv - x)
    return (xv, yv)

def cut(w, signal):
    mid = (len(signal)/w)/2.0
    if mid <= 0.5:
        return signal
    else:
        end = len(signal)/2.0 + w/2.0
        start = end - w
        out = []
        for i in range(0,len(signal)):
            if i > start and i <= end:
                out.append(signal[i])
        return out

if __name__ == "__main__":
    w, signal = wav.read(sys.argv[1])
    channel = 1
    if type(signal[0]) in (tuple, list, np.ndarray):
        signal = signal[:,0]/2.0 + signal[:,1]/2.0 #Pozbycie sie drugiego kanalu
        channel = 2
    signal = cut(w, signal)
    count = len(signal)  

    window = s.kaiser(len(signal), 100) #Okno
    signal = signal * window
    spectrum = log(abs(np.fft.rfft(signal))) #FFT
    hps = copy(spectrum)

    for i in range(2,7):
        dec = s.decimate(spectrum, i) #Decimate
        hps[:len(dec)] += dec

    peak_start = 50
    hps2 = hps[peak_start:]
    i_peak = argmax(hps2[:len(dec)])
    peak = parabolic(hps2, i_peak)[0]
    frequency = i_peak + peak_start #Czestotliwosc 
    #print(frequency)

    if frequency > 170:
        print('K')
    else:
        print('M')

    #fig = plt.figure(figsize=(15, 6), dpi=80)
    #plt.plot(spectrum, 'o')
    #plt.show()
