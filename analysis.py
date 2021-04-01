from scipy.io import wavfile as wav
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import scipy
import numpy as np
import librosa

audio_data = 'prueba_hard.wav'
x , sr = librosa.load(audio_data, sr=48000)
# print("=== X & SR ===")
# print(x)
# print(sr)
# print(len(x))

fft_out = fft(x)
# print("=== FFT OUT ===")
# print(fft_out)
# print(len(fft_out))

freqs = scipy.fft.fftfreq(len(fft_out), (1/sr))
# print("=== FREQS ===")
# print(freqs)
# print(len(freqs))

# print("=== PLOT ===")
# print(freqs[range(len(fft_out)//2)])
# print(range(len(fft_out)//2))
# print(len(fft_out))
# print(len(fft_out)//2)
plt.plot(freqs[range(len(fft_out)//2)], np.abs(fft_out[range(len(fft_out)//2)]))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude (dB)")
plt.show()