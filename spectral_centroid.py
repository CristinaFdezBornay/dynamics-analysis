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
        spectral_centroids = model.calculate_spectral_centroid()
        print("Mean spectral Centroid: {}".format(np.mean(spectral_centroids)))
        time = model.get_time(spectral_centroids)
        # range_to_plot = model.get_range_to_plot(0, len(time))
        plt.figure(figsize=(12, 4))
        librosa.display.waveplot(model.dataset, sr=model.sr, alpha=0.4)
        plt.plot(time, model.normalize(spectral_centroids), color='b')
        plt.xlabel("Time (s)")
        plt.title("Spectral Centroid")
        plt.show()
    except NameError as e:
        print(e)

if __name__ == '__main__':
    main()
