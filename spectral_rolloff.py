try:
    from classes.wav import Wav
    import matplotlib.pyplot as plt
    import numpy as np
    import librosa.display
except NameError as e:
    print(e)
    print('[Import error] Please run <pip install -r requirements.txt>')
    exit()

def main():
    try:
        model = Wav()
        spectral_rolloffs = model.calculate_spectral_rolloff()
        print("Mean spectral Rolloff: {}".format(np.mean(spectral_rolloffs)))
        time = model.get_time(spectral_rolloffs)
        # range_to_plot = model.get_range_to_plot(0, len(time))
        plt.figure(figsize=(12, 4))
        librosa.display.waveplot(model.dataset, sr=model.sr, alpha=0.4)
        plt.plot(time, model.normalize(spectral_rolloffs), color='b')
        plt.xlabel("Time (s)")
        plt.title("Spectral Rolloff")
        plt.show()
    except NameError as e:
        print(e)

if __name__ == '__main__':
    main()










# import matplotlib.pyplot as plt
# import librosa
# import librosa.display
# import sklearn
# import numpy as np

# audio_data = 'prueba_hard.wav'
# x , sr = librosa.load(audio_data, sr=48000)

# # Calculate spectral rolloff
# spectral_rolloff = librosa.feature.spectral_rolloff(x+0.01, sr=sr)[0]

# # Computing the time variable for visualization
# plt.figure(figsize=(12, 4))
# frames = range(len(spectral_rolloff))
# t = librosa.frames_to_time(frames)

# # Normalising the spectral rolloff for visualisation
# def normalize(x, axis=0):
#     return sklearn.preprocessing.minmax_scale(x, axis=axis)

# librosa.display.waveplot(x, sr=sr, alpha=0.4)
# plt.plot(t, normalize(spectral_rolloff), color='r')
# plt.xlabel("Time (s)")
# plt.title("Spectral Rolloff")
# plt.show()

# mean = np.sum(normalize(spectral_rolloff)) / len(normalize(spectral_rolloff))
# print(mean)