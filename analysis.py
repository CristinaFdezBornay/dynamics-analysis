from scipy.io import wavfile as wav
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import scipy
import numpy as np
import librosa

print("=== X & SR ===")
audio_data = 'prueba_hard.wav'
x , sr = librosa.load(audio_data, sr=48000)
print(x)
print(sr)
print(len(x))

# rate= wav.read(audio_data) --> NOT WORKING READ

print("=== FFT OUT ===")
fft_out = fft(x)
print(fft_out)
print(len(fft_out))

# %matplotlib inline
# plt.plot(x)
# plt.plot(fft_out)
# plt.plot(x, np.abs(fft_out))

print("=== FREQS ===")
freqs = scipy.fft.fftfreq(len(fft_out), (1/sr))
print(freqs)
print(len(freqs))


print("=== PLOT ===")
print(freqs[range(len(fft_out)//2)])
print(range(len(fft_out)//2))
print(len(fft_out))
print(len(fft_out)//2)
plt.plot(freqs[range(len(fft_out)//2)], fft_out[range(len(fft_out)//2)])
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude (dB)")
plt.show()