import matplotlib.pyplot as plt
import scipy
import numpy as np
import librosa
import librosa.display
import sklearn

audio_data = 'prueba_hard.wav'
x , sr = librosa.load(audio_data, sr=48000)

spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
spectral_centroids.shape
(775,)

# Computing the time variable for visualization
plt.figure(figsize=(12, 4))
frames = range(len(spectral_centroids))
t = librosa.frames_to_time(frames)

# Normalising the spectral centroid for visualisation
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

#Plotting the Spectral Centroid along the waveform
librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_centroids), color='b')
plt.xlabel("Time (s)")
plt.title("Spectral Centroid")
plt.show()

mean = np.sum(normalize(spectral_centroids)) / len(normalize(spectral_centroids))
print(mean)