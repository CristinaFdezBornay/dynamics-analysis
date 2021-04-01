try:
    from classes.wav import Wav
    import matplotlib.pyplot as plt
    import numpy as np
except NameError as e:
    print(e)
    print('[Import error] Please run <pip install -r requirements.txt>')
    exit()

def main():
    try:
        model = Wav()
        fft = model.calculate_fft()
        freqs = model.get_frequency_array(len(fft))
        range_to_plot = model.get_range_to_plot(0, len(fft)//2)
        plt.plot(freqs[range_to_plot], np.abs(fft[range_to_plot]))
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude (dB)")
        plt.title("FFT")
        plt.show()
    except NameError as e:
        print(e)

if __name__ == '__main__':
    main()