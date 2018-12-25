import sys
from scipy import *
from numpy import *
from pylab import *
from ipywidgets import *
import scipy.io.wavfile as wav
import soundfile as sf
import scipy.signal as s
import matplotlib.pyplot as plt
from matplotlib.pyplot import stem

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
        signal = signal[:,0]/2.0 + signal[:,1]/2.0
        channel = 2
    signal = cut(w, signal)
    count = len(signal)  

    window = s.kaiser(count, 100)
    signal = signal * window
    spectrum = np.log(abs(np.fft.rfft(signal)))
    spectrum2 = copy(spectrum)

    for i in range(2,7):
        dec = s.decimate(spectrum, i)
        spectrum2[:len(dec)] += dec

    peak_start = 50
    peak = np.argmax(spectrum2[peak_start:])
    frequency = peak_start + peak 
    #print(frequency)

    if frequency > 170:
        print('K')
    else:
        print('M')

    #fig = plt.figure(figsize=(15, 6), dpi=80)
    #plt.plot(spectrum, 'o')
    #plt.show()
