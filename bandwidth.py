import matplotlib.pyplot as plt
import librosa
import librosa.display
import sklearn
import numpy as np

audio_data = 'prueba_hard.wav'
x , sr = librosa.load(audio_data, sr=48000)

# Computing the time variable for visualization
frames = range(178)
t = librosa.frames_to_time(frames)

# Normalising the spectral rolloff for visualisation
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

# Calculate bandwidth
spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(x+0.01, sr=sr)[0]
spectral_bandwidth_3 = librosa.feature.spectral_bandwidth(x+0.01, sr=sr, p=3)[0]
spectral_bandwidth_4 = librosa.feature.spectral_bandwidth(x+0.01, sr=sr, p=4)[0]

#  Plot
plt.figure(figsize=(15, 9))
librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_bandwidth_2), color='r')
plt.plot(t, normalize(spectral_bandwidth_3), color='g')
plt.plot(t, normalize(spectral_bandwidth_4), color='y')
plt.legend(('p = 2', 'p = 3', 'p = 4'))
plt.xlabel("Time (s)")
plt.show()

# mean = np.sum(normalize(spectral_centroids)) / len(normalize(spectral_centroids))
mean = np.mean(normalize(spectral_bandwidth_2))
print('P2 mean: {}'.format(mean))
mean = np.mean(normalize(spectral_bandwidth_3))
print('P3 mean: {}'.format(mean))
mean = np.mean(normalize(spectral_bandwidth_4))
print('P4 mean: {}'.format(mean))