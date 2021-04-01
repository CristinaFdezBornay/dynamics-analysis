import matplotlib.pyplot as plt
import librosa
import librosa.display
import sklearn
import numpy as np

audio_data = 'prueba_hard.wav'
x , sr = librosa.load(audio_data, sr=48000)

#Plot the signal:
plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr)

# Zooming in
n0 = 10
n1 = 300
n1 = len(x)
plt.figure(figsize=(14, 5))
plt.plot(x[n0:n1])
plt.grid()
plt.show()

zero_crossings = librosa.zero_crossings(x[n0:n1], pad=False)
print(sum(zero_crossings))