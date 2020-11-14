# DT2119, Lab 1 Feature Extraction
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, fftpack
from lab1_tools import *
from scipy.fftpack.realtransforms import dct
from scipy.spatial.distance import cdist
# Function given by the exercise ----------------------------------

def mspec(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, samplingrate=20000):
    
    frames = enframe(samples,samplingrate, winlen, winshift)
    preemph = preemp(frames, preempcoeff)
    windowed = windowing(preemph)
    spec = powerSpectrum(windowed, nfft)
    return logMelSpectrum(spec, samplingrate)

def mfcc(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, nceps=13, samplingrate=20000, liftercoeff=22):
    
    mspecs = mspec(samples, winlen, winshift, preempcoeff, nfft, samplingrate)
    ceps = cepstrum(mspecs, nceps)
    return lifter(ceps, liftercoeff)

# Functions to be implemented ----------------------------------

def enframe(samples,samplingrate, winlen, winshift):
    """
    Slices the input samples into overlapping windows.
    Args:
        winlen: window length in samples.
        winshift: shift of consecutive windows in samples
    Returns:
        numpy array [N x winlen], where N is the number of windows that fit
        in the input signal
    """
    N = len(samples)
    result = np.array(samples[0:winlen],dtype='float64')
    i = winshift
    while True:
        if i+winlen>=N:
            break
        result=np.vstack((result,samples[i:i+winlen]))
        i = i + winshift
    return result


def preemp(input, p=0.97):
   
    N = input.shape[0]
    result = np.copy(input)

    # FIR Filter
    b = [1,-p]
    a = [1, 0]
    for i in range(N):
        result[i,:] = signal.lfilter(b,a, input[i,:])
    return result

def windowing(input):
    
    window = signal.hamming(input.shape[1], sym=False)
    for idx, row in enumerate(input):
        input[idx] = window * input[idx]
    return input

def powerSpectrum(input, nfft):
    
    return np.power(np.abs(fftpack.fft(input, nfft)), 2)

def logMelSpectrum(input, samplingrate):
    
    T = trfbank(samplingrate, 512, lowfreq=133.33, linsc=200 / 3., logsc=1.0711703, nlinfilt=13, nlogfilt=27,
                equalareas=False)
    # Plot filter bank
    # for i in range(len(T)):
    # plt.plot(np.transpose(T[i]))
    # plt.show()
    Spec = np.dot(input, T.T)
    print(Spec)
    Spec = np.where(Spec == 0.0, np.finfo(float).eps, Spec)  # Numerical Stability
    print(Spec)
    return np.log(Spec)

def cepstrum(input, nceps):
    
    mfcc = dct(input)
    mfcc = mfcc[:,0:nceps]
    #return lifter(mfcc)
    return mfcc


    
