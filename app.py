# SELIM SAHIN - 18253071 @ PAU 2023-2024
# İşaret İşleme - High Pass Filter ve Low Pass Filter Uygulaması

import librosa
import librosa.effects
import soundfile as sf
import datetime
import matplotlib.pyplot as plt
import numpy as np    
from scipy import signal
from scipy.signal import butter, lfilter, freqz

#### --------------------- FILTERING FUNCTIONS --------------------- ####

def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

#### --------------------- APPLY THE FILTERS --------------------- ####

# Path to the audio file
sourceFile = 'audio/SampleAudio.mp3'

# Load the audio file using librosa
audio, samplerate = librosa.load(sourceFile, sr=None)

# Path and name of the .wav file to be exported
date_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
highpass_export_path = date_time + '_Audio_HighPass.wav'
lowpass_export_path = date_time + '_Audio_LowPass.wav'

# Filter requirements
order = 6
fs = 30.0       # sample rate, Hz
cutoff = 3.667  # desired cutoff frequency of the filter, Hz

# Filter the audio signal. Create lowpass and highpass audio files.
highpass_audio = butter_highpass_filter(audio, cutoff, fs, order)
lowpass_audio = butter_lowpass_filter(audio, cutoff, fs, order)

# Export the audio files in .wav format.
sf.write(highpass_export_path, highpass_audio, samplerate)
sf.write(lowpass_export_path, lowpass_audio, samplerate)

#### --------------------- PLOTTING FUNCTIONS --------------------- ####

# Time axis in seconds
t = np.arange(0, len(audio)) / samplerate

# Plot the frequency response (high-pass filter)
b, a = butter_highpass(cutoff, fs, order)
w, h = freqz(b, a, worN=8000)

plt.figure(figsize=(10, 8))

# Plot the frequency response of the high-pass filter.
plt.subplot(2, 2, 1)
plt.plot(0.5 * fs * w / np.pi, np.abs(h), 'b')
plt.plot(cutoff, 0.5 * np.sqrt(2), 'ko')
plt.axvline(cutoff, color='k')
plt.xlim(0, 0.5 * fs)
plt.title("High-Pass Filter Frequency Response")
plt.xlabel('Frequency [Hz]')
plt.grid()

# Plot the original audio signal and the high-pass filtered signal.
plt.subplot(2, 2, 3)
plt.plot(t, audio, 'b-', label='Original Audio')
plt.plot(t, highpass_audio, 'g-', linewidth=2, label='Highpass Filtered Audio')
plt.xlabel('Time [sec]')
plt.grid()
plt.legend()

# Plot the frequency response (low-pass filter)
b, a = butter_lowpass(cutoff, fs, order)
w, h = freqz(b, a, worN=8000)

# Plot the frequency response of the low-pass filter.
plt.subplot(2, 2, 2)
plt.plot(0.5 * fs * w / np.pi, np.abs(h), 'b')
plt.plot(cutoff, 0.5 * np.sqrt(2), 'ko')
plt.axvline(cutoff, color='k')
plt.xlim(0, 0.5 * fs)
plt.title("Low-Pass Filter Frequency Response")
plt.xlabel('Frequency [Hz]')
plt.grid()

# Plot the original audio signal and the low-pass filtered signal.
plt.subplot(2, 2, 4)
plt.plot(t, audio, 'b-', label='Original Audio')
plt.plot(t, lowpass_audio, 'g-', linewidth=2, label='Lowpass Filtered Audio')
plt.xlabel('Time [sec]')
plt.grid()
plt.legend()

plt.tight_layout()  # Adjust the layout for better spacing
plt.show()
