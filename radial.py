# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 18:07:03 2022

@author: Amin Abdipour
"""

### Import required packages
import csv
import math
import pandas as pd
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
# Helps to obtain the FFT
import scipy.fftpack    
# Various operations on signals (waveforms)
import scipy.signal as signal
                                                    
                                                    ###Obtain ecg sample from file###
mat = scipy.io.loadmat('sample_data.mat')
resultList = list(mat['sample_data'])
y = list(resultList[0]);

# Number of samplepoints
N = len(y)
# sample spacing
Fs = 500
T = 1 / Fs
#Compute x-axis

x = np.linspace(0, 20, 10000)

#Compute FFT
yf = scipy.fftpack.fft(y)
#Compute frequency x-axis
xf = np.linspace(0, 250, 5000)

                                                                ##Declare plots for time-domain and frequency-domain plots##
fig_td = plt.figure()
fig_td.canvas.set_window_title('Time domain signals')
fig_fd = plt.figure()
fig_fd.canvas.set_window_title('Frequency domain signals')
ax1 = fig_td.add_subplot(211)
ax1.set_title('Before filtering')
ax2 = fig_td.add_subplot(212)
ax2.set_title('After filtering')
ax3 = fig_fd.add_subplot(211)
ax3.set_title('Before filtering')
ax4 = fig_fd.add_subplot(212)
ax4.set_title('After filtering')     

#Plot non-filtered inputs
ax1.plot(x,y, color='r', linewidth=0.7)
ax3.plot(xf, 2.0/N * np.abs(yf[:N//2]), color='r', linewidth=0.7, label='raw')
ax3.set_ylim([0 , 300])

                                                               ###Compute filtering - Bandpass 0.1 to 40 Hz###

b, a = signal.butter(6, [1, 40], fs=Fs, btype='band')
freq, h = signal.freqz(b, a, fs=Fs)
fig, ax = plt.subplots(2, 1, figsize=(8, 6))
ax[0].plot(freq, 20*np.log10(abs(h)), color='blue')
ax[0].set_title("Frequency Response")
ax[0].set_ylabel("Amplitude (dB)", color='blue')
ax[0].set_xlim([0, 100])
ax[0].set_ylim([-25, 10])
ax[0].grid(True)
ax[1].plot(freq, np.unwrap(np.angle(h))*180/np.pi, color='green')
ax[1].set_ylabel("Angle (degrees)", color='green')
ax[1].set_xlabel("Frequency (Hz)")
ax[1].set_xlim([0, 100])
ax[1].set_yticks([-90, -60, -30, 0, 30, 60, 90])
ax[1].set_ylim([-90, 90])
ax[1].grid(True)


tempf = signal.filtfilt(b,a, y)
yff = scipy.fftpack.fft(tempf)


"""
                                                                ### Design and plot filter to remove the 50 Hz component from a signal
f0 = 50.0  # Frequency to be removed from signal (Hz)
Q = 30.0  # Quality factor
# Design notch filter
b, a = signal.iirnotch(f0, Q, Fs)
freq, h = signal.freqz(b, a, fs=Fs)
fig, ax = plt.subplots(2, 1, figsize=(8, 6))
ax[0].plot(freq, 20*np.log10(abs(h)), color='blue')
ax[0].set_title("Frequency Response")
ax[0].set_ylabel("Amplitude (dB)", color='blue')
ax[0].set_xlim([0, 100])
ax[0].set_ylim([-25, 10])
ax[0].grid(True)
ax[1].plot(freq, np.unwrap(np.angle(h))*180/np.pi, color='green')
ax[1].set_ylabel("Angle (degrees)", color='green')
ax[1].set_xlabel("Frequency (Hz)")
ax[1].set_xlim([0, 100])
ax[1].set_yticks([-90, -60, -30, 0, 30, 60, 90])
ax[1].set_ylim([-90, 90])
ax[1].grid(True)

tempf = signal.filtfilt(b,a, tempf1)
yff = scipy.fftpack.fft(tempf)
"""

# Plot filtered outputs
ax4.plot(xf, 2.0/N * np.abs(yff[:N//2]), color='g', linewidth=0.7)
ax4.set_ylim([0 , 300.2])
# ax1.set_xlim([10 , 14])
# ax2.set_xlim([10 , 14])
ax2.plot(x,tempf, color='g', linewidth=0.7);

#method 1 for find peaks
plt.figure();
peaks, _ = signal.find_peaks(tempf, height=33000)
plt.plot(tempf)
plt.plot(peaks, tempf[peaks])
plt.plot(np.zeros_like(x), "--", color="gray")
plt.show()

#method 2 for find peaks
plt.figure();
peaks, _ = signal.find_peaks(tempf, distance=200)
np.diff(peaks)
plt.plot(tempf)
plt.plot(peaks, tempf[peaks])
plt.show()


                                     #Compute heart rate
RR_list = []
cnt = 0
while (cnt < (len(peaks)-1)):
    RR_interval = (peaks[cnt+1] - peaks[cnt]) #Calculate distance between beats in # of samples
    ms_dist = ((RR_interval / Fs) * 1000.0) #Convert sample distances to ms distances
    RR_list.append(ms_dist) #Append to list
    cnt += 1

bpm = 60000 / np.mean(RR_list) #60000 ms (1 minute) / average R-R interval of signal
print("\n\n\nAverage Heart Beat is: %.01f\n" %(bpm)) #Round off to 1 decimal and print
print("mean deviation of R to R intervals in the signal: %.01f\n" %(np.mean(RR_list)))
print("standard deviation of R to R intervals in the signal: %.01f\n" %(np.std(RR_list)))
print("No of peaks in sample are: {0}".format(len(peaks)))

plt.show()
