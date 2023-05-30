import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter
from torchaudio.functional import filtfilt
import librosa

from torchaudio import transforms, load
from torch.nn.functional import normalize
from torch import from_numpy, zeros, cat

import random


class lungSoundUtil():
    @staticmethod
    def wavToArray(file):
        # sr, data = wavfile.read(f"{audio_file_path}/{file}")
        # Loading with librosa resamples the data to the specified
        # frequencey (4000Hz) and normalizes the data between 1 and -1
        signal, sr = load(file, normalize=True)
        # Some of the recordings are multi-channel. Convert to a single channel
        signal = signal[:1, :]

        # Normalize from 1 to -1
        # normalize(signal, )

        return (sr, signal)

    @staticmethod
    # High pass filter function to remove heart sounds
    def filterNoise(sampling_rate, signal):
        # Butterworth band pass filter with cutoff frequencies
        # as a fraction of nquist frequency (1/2 the sampling rate)
        b, a = butter(3, [300/(0.5*sampling_rate), 1000/(0.5*sampling_rate)], 'band')
        signal = from_numpy(lfilter(b, a, signal)).float()        

        # signal = filtfilt(signal, from_numpy(b).float(), from_numpy(a).float())

        # Downsample result to 4000 Hz
        signal = transforms.Resample(sampling_rate, 4000)(signal)

        return signal

    @staticmethod
    def createMelSpectogram(signal):
        signal = signal.numpy()
        spec = librosa.feature.melspectrogram(y=signal, sr=4000, n_mels=256, fmax=1500, n_fft=8192)
        # n_fft is the window size -> We make it 2X the sr, because we assume a complete
        # respiratory phase takes 2 seconds -> 1 second inhale, 1 second exhale
        spec_dec = from_numpy(librosa.power_to_db(spec, ref=np.max))

        return spec_dec

    @staticmethod
    def augmentSpectogram(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
        _, n_mels, n_steps = spec.shape
        mask_value = spec.mean()
        aug_spec = spec

        freq_mask_param = max_mask_pct * n_mels
        for _ in range(n_freq_masks):
          aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_masks):
          aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

        return aug_spec

    @staticmethod
    def pad_trunc(signal):
        sig = signal
        num_rows, sig_len = sig.shape
        max_len = 4000//1000 * 20000 # 20 seconds

        if (sig_len > max_len):
          # Truncate the signal to the given length
          sig = sig[:,:max_len]

        elif (sig_len < max_len):
          # Length of padding to add at the beginning and end of the signal
          pad_begin_len = random.randint(0, max_len - sig_len)
          pad_end_len = max_len - sig_len - pad_begin_len

          # Pad with 0s
          pad_begin = zeros((num_rows, pad_begin_len))
          pad_end = zeros((num_rows, pad_end_len))

          sig = cat((pad_begin, sig, pad_end), 1)
          
        return sig
