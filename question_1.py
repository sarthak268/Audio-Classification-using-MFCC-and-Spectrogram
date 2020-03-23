# All images must be saved in the same format as given in question i.e. in Dataset folder

import numpy as np
import glob
from scipy.io import wavfile
from scipy.fftpack import dct
import matplotlib.pyplot as plt
import pickle
import random

def hz2mel(freq):

    return (2595 * np.log10(1 + freq / 1400))

def mel2hz(freq):

    return (700 * (10 ** (freq / 2595) - 1))

def db(signal):

    return 20 * np.log10(signal, where=signal>0)

def get_hz_scale(s, fs, num_of_points):

    f = s * fs / num_of_points
    
    f1 = []
    for i in range(len(f)):
        f1.append(int(f[i]))

    return f1

def mfcc(signal, fs, preemphasis_coeff=0.95, frame_size=0.025, frame_stride=0.01, nfft=512, num_triang_filters=40, number_of_ceps=12, ceps_lifting=10):

    # step 1 : preemphasis
    emp_signal = np.append(signal[0], signal[1:] - preemphasis_coeff * signal[:-1])

    # step 2 : framing
    f_l = int(round(frame_size * fs))
    f_s = int(round(frame_stride * fs))
    s_l = len(emp_signal)
    n_f = int(np.ceil(float(np.abs(s_l - f_l)) / f_s)) # dont remove all frames

    s_l_after_padding = n_f * f_s + f_l
    zero = np.zeros((s_l_after_padding - s_l))
    padded_signal = np.append(emp_signal, zero)

    indices = np.tile(np.arange(0, f_l), (n_f, 1)) + np.tile(np.arange(0, n_f * f_s, f_s), (f_l, 1)).T
    frames = padded_signal[indices.astype(np.int32, copy=False)]

    # step 3 : window
    frames = frames * np.hamming(f_l)

    # step 4 : FT and power spectrum
    magnitude = np.absolute(np.fft.rfft(frames, nfft))
    power_spectrum_frames = (1. / nfft) * (magnitude ** 2)

    # step 5 : filter banks
    low_mel_frequency = 0
    high_mel_frequency = hz2mel(fs)
    mel_scale_low2high = np.linspace(low_mel_frequency, high_mel_frequency, num_triang_filters + 2)
    hz_scale_low2high = mel2hz(mel_scale_low2high)

    bins = np.floor((nfft + 1) * hz_scale_low2high / fs)
    filter_bank = np.zeros((num_triang_filters, int(np.floor((nfft / 2) + 1))))
    for i in range(1, num_triang_filters + 1):
        f_left, f_right, f = int(bins[i-1]), int(bins[i+1]), int(bins[i])

        for j in range(f_left, f):
            filter_bank[i-1, j] = (j - bins[i-1]) / (bins[i] - bins[i-1])
        
        for k in range(f, f_right):
            filter_bank[i-1, j] = (bins[i+1] - k) / (bins[i+1] - bins[i])

    filter_bank = np.dot(power_spectrum_frames, filter_bank.T)
    filter_bank = db(filter_bank)

    # step 6 : mfcc
    mfcc = dct(filter_bank)[:, 1 : (number_of_ceps + 1)]

    # step 7 : sinusoidal lifting (additional step)
    n = np.arange(mfcc.shape[1])
    lifting = 1 + (ceps_lifting / 2) * np.sin(np.pi * n / ceps_lifting)
    mfcc_after_lifting = mfcc * lifting

    # step 8 : mean normalisation 
    mfcc_mean_normalised = mfcc_after_lifting - np.mean(mfcc_after_lifting, axis=0)

    return mfcc_mean_normalised, len(signal)

def plot_mfcc(mfcc_f, fs, len_sig):

    plt_spectro = plt.imshow(mfcc_f, origin='lower')

    xticks, yticks =  10, 10
    c = np.linspace(0, mfcc_f.shape[0], yticks)
    c_in_hz = get_hz_scale(c, fs, len_sig)
    
    plt.xlabel("time")
    plt.yticks(c, c_in_hz)
    plt.ylabel("frequency")

    plt.title("Spectrogram")
    plt.show()


if (__name__ == '__main__'):
    
    training = False
    noise = False
    
    classes = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    noise_data = './Dataset/_background_noise_/'
    all_noises = glob.glob(noise_data + '*.wav')
    all_data = []

    if (training):
        mode = 'training'
    else:
        mode = 'validation'

    if (noise):
        noise_str = '_noise'
    else:
        noise_str = ''
    
    for cl_id in range(len(classes)):
        cl = classes[cl_id]

        count = 0
        per_class_data = []
        classwise_files = glob.glob('./Dataset/' + mode + '/' + cl + '/*wav') 

        for f in classwise_files:
            print (f)
            sampling_rate, data = wavfile.read(f)

            zero = np.zeros((sampling_rate - len(data)))
            data = np.append(data, zero)

            mfcc_feature, l_sig = mfcc(data, sampling_rate)
            #plot_mfcc(mfcc_feature, sampling_rate, l_sig)
            per_class_data.append(mfcc_feature)

            if (noise):
                f = random.choice(all_noises)
                sampling_rate, noise_data = wavfile.read(f)

                noise_data = noise_data * 0.005
                indx = random.randint(0,noise_data.shape[0]-sampling_rate-1)
                noise_data = noise_data[indx:indx+sampling_rate]
                data = noise_data + data

                mfcc_feature, l_sig = mfcc(data, sampling_rate)
                #plot_mfcc(mfcc_feature, sampling_rate, l_sig)
                per_class_data.append(mfcc_feature)
            
        all_data.append(per_class_data)

    with open('mfcc_' + mode + noise_str + '.txt', 'wb') as fp:
        pickle.dump(all_data, fp)
    



