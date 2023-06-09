from data_loader import df3 as df
import audio_util
import matplotlib.pyplot as plt
import numpy as np
from torch import arange
import librosa

plot_colors = {
    "Tc": "#32F2F5",
    "Al": "#FA9A0A",
    "Ll": "#F53532",
    "Pl": "#2AC126",
    "Ar": "#0A6AFA",
    "Pr": "#BD26C1",
    "Lr": "#32F2F5"
}

def plotSpectogram(spec, axs, row, column):
    spec = spec.numpy()
    img = librosa.display.specshow(spec[0], x_axis='time', y_axis='mel', sr=4000, ax=axs[row][column])
    # fig.colorbar(img, ax=axs[row][column], format='%+2.0f dB')
    return

def plotWaveform(sig, sr, axs, row, column):
    sig = sig.numpy()
    num_channels, num_frames = sig.shape
    time_axis = arange(0, num_frames) / sr
    axs[row][column].plot(time_axis, sig[0])
    return axs

def createPlots(o):#patient_number, files, diagnosis, chest_location=''):
    # Time domain
    fig, axs = plt.subplots(2,2, figsize=(15, 6))
    

    axs[0][0].set_xlabel("Time [s]")
    axs[0][1].set_xlabel("Time [s]")
    axs[1][0].set_xlabel("Time [s]")
    axs[0][0].set_ylabel("Amplitude")
    axs[0][1].set_ylabel("Amplitude")
    axs[1][0].set_ylabel("Frequency")
    axs[0][0].set_title(f"a)")
    axs[0][1].set_title(f"b)")
    axs[1][0].set_title(f"c)")

    sr, raw_sig = audio_util.lungSoundUtil.wavToArray(o['file'])
    plotWaveform(raw_sig, sr, axs, 0, 0)

    filt_sig = audio_util.lungSoundUtil.filterNoise(sr, raw_sig)
    plotWaveform(filt_sig, 4000, axs, 0, 1)


    duration_sig = audio_util.lungSoundUtil.pad_trunc(filt_sig)
    spectogram = audio_util.lungSoundUtil.createMelSpectogram(duration_sig)
    plotSpectogram(spectogram, axs, 1, 0)

    # filt_time = np.linspace(0, filt_sig.shape[0]/4000, filt_sig.shape[0])

    # axs[0][0].plot(time_axis, raw_sig[0])
    # axs[0][1].plot(filt_time, signal)
    # axs[1][0].specgram(spectogram, 4000)

    # axs[0][1].legend()
    
    fig.tight_layout()
    plt.savefig(f"waveforms/{o['patient_no']}_{o['diagnosis']}_timedom.png")
    plt.close()

    return

for i, d in df.iterrows():
    createPlots(d)
