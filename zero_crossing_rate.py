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
        plt.figure(figsize=(14, 5))
        librosa.display.waveplot(model.dataset, sr=model.sr)
        range_to_plot = model.get_range_to_plot(0, len(model.dataset))
        print("Zero crossing: {}".format(model.calculate_zero_crossing(range_to_plot)))
        plt.figure(figsize=(14, 5))
        plt.plot(model.dataset[range_to_plot])
        plt.grid()
        plt.show()
    except NameError as e:
        print(e)

if __name__ == '__main__':
    main()