try:
    import argparse
    import librosa
    import sklearn
    import scipy
except NameError as e:
    print(e)
    print('[Import error] Please run <pip install -r requirements.txt>')
    exit()

class Wav:
    def __init__(self):
        self.sr = 48000
        self.args = self.parse_arg()
        self.datafile = self.args.datafile
        self.dataset = self.extract_dataset()
    
    def parse_arg(self):
        try:
            parser = argparse.ArgumentParser(prog='program', usage='%(prog)s [-h][-cf]datafile.wav')
            parser.add_argument('datafile', help='the .wav file containing the dataset.')
            parser.add_argument('-cr', '--choose_range', help='choose the range to plot.', action='store_true')
            args = parser.parse_args()
            return args
        except:
            raise NameError('[Parse error] There has been an error while parsing the arguments.')
    
    def extract_dataset(self):
        try:
            return librosa.load(self.datafile, sr=self.sr)[0]
        except:
            raise NameError('[Reading error] There has been an error while extracting the dataset.')

    def normalize(self, dataset):
        try:
            axis_to_normalize = 0
            return sklearn.preprocessing.minmax_scale(dataset, axis=axis_to_normalize)
        except:
            raise NameError('[Process error] There has been an error while normalizing the dataset.')

    def calculate_fft(self):
        try:
            return scipy.fftpack.fft(self.dataset)
        except:
            raise NameError('[Process error] There has been an error while calculating the fft the dataset.')

    def calculate_spectral_centroid(self):
        try:
            return librosa.feature.spectral_centroid(self.dataset, sr=self.sr)[0]
        except:        
            raise NameError('[Process error] There has been an error while calculating the spectral centroid.')

    def calculate_spectral_rolloff(self):
        try:
            return librosa.feature.spectral_rolloff(self.dataset+0.01, sr=self.sr)[0]
        except:        
            raise NameError('[Process error] There has been an error while calculating the spectral rolloff.')

    def calculate_spectral_bandwidth(self, p):
        try:
            return librosa.feature.spectral_bandwidth(self.dataset+0.01, sr=self.sr, p=p)[0]
        except:
            raise NameError('[Process error] There has been an error while calculating the spectral bandwidth.')

    def calculate_zero_crossing(self, range):
        try:
            return sum(librosa.zero_crossings(self.dataset[range], pad=False))
        except:
            raise NameError('[Process error] There has been an error while calculating the zero crossing.')

    def get_frequency_array(self, fft_len):
        try:
            return scipy.fft.fftfreq(fft_len, (1/self.sr))
        except:
            raise NameError('[Process error] There has been an error while generating the frquency array.')

    def get_range_to_plot(self, n0, n1):
        try:
            if self.args.choose_range:
                n0 = int(input("Introduce min [ {} - {} ]: ".format(n0, n1)))
                n1 = int(input("Introduce max [ {} - {} ]: ".format(n0, n1)))
            return range(n0, n1)
        except:
            raise NameError('[Process error] There has been an error while generating the range to plot.')

    def get_time(self, signal):
        try:
            frames = range(len(signal))
            return librosa.frames_to_time(frames)
        except:
            raise NameError('[Process error] There has been an error while generating the time to plot.')
