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
        spectral_bandwidth_2 = model.calculate_spectral_bandwidth(2)
        normalized_sb_2 = model.normalize(spectral_bandwidth_2)
        spectral_bandwidth_3 = model.calculate_spectral_bandwidth(3)
        normalized_sb_3 = model.normalize(spectral_bandwidth_3)
        spectral_bandwidth_4 = model.calculate_spectral_bandwidth(4)
        normalized_sb_4 = model.normalize(spectral_bandwidth_4)

        print('P2 mean : {}'.format(np.mean(spectral_bandwidth_2)))
        print('P2 mean normalized: {}'.format(np.mean(normalized_sb_2)))
        print('P3 mean : {}'.format(np.mean(spectral_bandwidth_3)))
        print('P3 mean normalized: {}'.format(np.mean(normalized_sb_3)))
        print('P4 mean : {}'.format(np.mean(spectral_bandwidth_4)))
        print('P4 mean normalized: {}'.format(np.mean(normalized_sb_4)))
        
        time = model.get_time(spectral_bandwidth_2)
        # range_to_plot = model.get_range_to_plot(0, len(time))

        plt.figure(figsize=(15, 9))
        librosa.display.waveplot(model.dataset, sr=model.sr, alpha=0.4)
        plt.plot(time, normalized_sb_2, color='r')
        plt.plot(time, normalized_sb_3, color='g')
        plt.plot(time, normalized_sb_4, color='y')
        plt.legend(('p = 2', 'p = 3', 'p = 4'))
        plt.xlabel("Time (s)")
        plt.show()
    except NameError as e:
        print(e)

if __name__ == '__main__':
    main()
