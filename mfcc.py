import numpy as np
import matplotlib.pyplot as plt
import os
import librosa
from IPython.display import Audio
from scipy.fftpack import dct

def stft(y, window_ms = 10, overlap_pct = 0.25):
    window_size = int(window_ms * sr / 1000)
    hop = int(window_size * (1 - overlap_pct))
    w = np.hanning(window_size + 1)[:-1]
    return np.array([np.fft.rfft(w * y[i:i + window_size]) for i in range(0, len(y) - window_size, hop)])

def mel_filter_bank(sr, n_fft, n_mels):
    mel_min = 0
    mel_max = 1127 * np.log(1 + (sr / 2) / 700) # Maximum frequency is half the sampling rate
    mels = np.linspace(mel_min, mel_max, num=n_mels + 2) #Equally spaced in Mel scale, but add two for the endpoints
    freqs = 700 * (np.exp(mels / 1127) - 1) # Convert back into frequency domain
    total_n_fft = (n_fft - 1) * 2 #The total number of FFT bins before we cut in half
    bins = np.floor(total_n_fft * freqs / sr).astype(int) # Which bin does each frequency correspond to? bin*sr/n_fft = freq

    fbank = np.zeros([n_mels, n_fft]) #Initialize empty filter bank matrix with rows = number of mel filters, columns = total number of FFT bins

    for m in range(1, n_mels + 1): #For each mel filter
        #Mark start, middle and end frequency bins
        f_m_minus = bins[m - 1]
        f_m = bins[m]
        f_m_plus = bins[m + 1]

        #Linear interpolation between each bin for the mel filter
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bins[m - 1]) / (bins[m] - bins[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bains[m + 1] - k) / (bins[m + 1] - bins[m])

    return fbank

def get_mfcc(y, sr, window_ms = 10, overlap_pct = 0.25, mel_banks = 20, n_mfcc = 12):
    stft_result = stft(y, window_ms, overlap_pct)
    magnitude_spectrogram = np.abs(stft_result)
    power_spectrogram = magnitude_spectrogram ** 2 #Power = amplitude^2 is often taken as it aligns more with human hearing
    mel_filterbank  = mel_filter_bank(sr, power_spectrogram.shape[1], mel_banks)
    mel_spectrogram = np.dot(power_spectrogram, mel_filterbank.T)
    log_mel_spectrogram = np.log(mel_spectrogram)
    return dct(log_mel_spectrogram, type=2, axis=1, norm='ortho')[:, 1:(n_mfcc+1)] # Keep first few coefficients, except the first, as others are usually not informative