import sys
from scipy import *
from numpy import *
from pylab import *
from ipywidgets import *
import scipy.io.wavfile as wav
import scipy.signal as s
import matplotlib.pyplot as plt
from matplotlib.pyplot import stem

def cut(w, signal):
    mid = (len(signal)/w)/2.0
    if mid <= 0.5:
        return signal
    else:
        start = len(signal)/2.0 - w/2.0
        end = start + w
        out = []
        for i in range(0,len(signal)):
            if i > start and i <= end:
                out.append(signal[i])
        return out

if __name__ == "__main__":
    w, signal = wav.read(sys.argv[1])
    signal = signal[:,0]/2.0 + signal[:,1]/2.0
    signal = signal / max(signal)
    signal = cut(w, signal)  
    
    window = s.hann(len(signal)/2)
    zeros = zeros(len(signal)/2)
    fun = hstack([window,zeros])
    signal = signal*fun
    spectrum = abs(fft(signal))
    hps = copy(spectrum)

    for i in range(2, 7):
        dec = s.decimate(spectrum, i)
        hps[:len(dec)] += dec
    peak = np.argmax(hps[50:])
    fundamental = 50 + peak
    print(fundamental)
    fig = plt.figure(figsize=(15, 6), dpi=80)
    plt.plot(signal, linestyle='-', color='red')
    plt.show()
