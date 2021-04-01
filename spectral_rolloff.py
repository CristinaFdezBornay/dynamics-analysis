import matplotlib.pyplot as plt
import librosa
import librosa.display
import sklearn
import numpy as np

audio_data = 'prueba_hard.wav'
x , sr = librosa.load(audio_data, sr=48000)

# Calculate spectral rolloff
spectral_rolloff = librosa.feature.spectral_rolloff(x+0.01, sr=sr)[0]

# Computing the time variable for visualization
plt.figure(figsize=(12, 4))
frames = range(len(spectral_rolloff))
t = librosa.frames_to_time(frames)

# Normalising the spectral rolloff for visualisation
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_rolloff), color='r')
plt.xlabel("Time (s)")
plt.title("Spectral Rolloff")
plt.show()

mean = np.sum(normalize(spectral_rolloff)) / len(normalize(spectral_rolloff))
print(mean)