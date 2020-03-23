# All images must be saved in the same format as given in question i.e. in Dataset folder

import numpy as np
import glob
from scipy.io import wavfile
import matplotlib.pyplot as plt
import pickle
import random

def get_fourier_coeff(signal, i):

    a = np.arange(0, len(signal))
    coeff = np.sum(signal * np.exp((1j * 2 * np.pi * a * i) / len(signal))) / len(signal)
    return coeff

def get_fourier_coeff_abs(signal):

    coeff = []
    for i in range(int(len(signal) / 2)):
        coeff.append(np.abs(get_fourier_coeff(signal, i)) * 2)

    return coeff

def db(signal):

    return 20 * np.log10(signal, where=signal>0)

def get_hz_scale(s, fs, num_of_points):

    f = s * fs / num_of_points
    
    f1 = []
    for i in range(len(f)):
        f1.append(int(f[i]))
    
    return f1

def spectrogram(signal, fs, nfft=512):

    #num_pts_overlap = nfft / 2
    num_pts_overlap = 0
    s = np.arange(0, len(signal), nfft - num_pts_overlap, dtype=int)
    s = s[s + nfft < len(signal)]

    dft_coeff = []
    for s1 in s:
        # find the ST DFT
        t_window = get_fourier_coeff_abs(signal[s1 : s1 + nfft])
        dft_coeff.append(t_window)

    spectro_hz = np.array(dft_coeff).T
    spectro = db(spectro_hz)

    return spectro, len(signal)

def plot_spectrogram(spectro, fs, len_sig):

    plt_spectro = plt.imshow(spectro, origin='lower')

    xticks, yticks =  10, 10
    c = np.linspace(0, spectro.shape[0], yticks)
    c_in_hz = get_hz_scale(c, fs, len_sig)
    
    plt.xlabel("time")
    plt.yticks(c, c_in_hz)
    plt.ylabel("frequency")

    plt.title("Spectrogram")
    plt.show()

if (__name__ == '__main__'):

    training = True
    noise = False
    
    classes = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    noise_data = './Dataset/_background_noise_/'
    all_noises = glob.glob(noise_data + '*.wav')
    #classes = ['zero', 'one']
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

            spectro_feature, l_sig = spectrogram(data, sampling_rate)
            #plot_spectro(spectro_feature, sampling_rate, l_sig)
            per_class_data.append(spectro_feature)

            if (noise):
                f = random.choice(all_noises)
                sampling_rate, noise_data = wavfile.read(f)

                noise_data = noise_data * 0.005
                indx = random.randint(0,noise_data.shape[0]-sampling_rate-1)
                noise_data = noise_data[indx:indx+sampling_rate]
                data = noise_data + data

                spectro_feature, l_sig = spectrogram(data, sampling_rate)
                #plot_spectro(spectro_feature, sampling_rate, l_sig)
                per_class_data.append(spectro_feature)
            
        all_data.append(per_class_data)

    with open('spectro_' + mode + noise_str + '.txt', 'wb') as fp:
        pickle.dump(all_data, fp)
    

