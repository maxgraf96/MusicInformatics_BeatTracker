import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import matplotlib as mpl

# Increse DPI (resolution) of plots
mpl.rcParams['figure.dpi'] = 300

import Globals


def plot_mel_spectrogram(mel_spectrogram, sr):
    plt.figure(figsize=(10, 4))
    S_dB = librosa.power_to_db(mel_spectrogram, ref=np.max)
    librosa.display.specshow(S_dB, x_axis='time',
                             y_axis='mel',
                             sr=sr,
                             fmax=sr,
                             hop_length=32)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    plt.show()

def plot_OSE(ose, beats):
    plt.figure(figsize=(10, 4))
    x = np.arange(ose.size)
    plt.plot(x, ose)
    # Plot beats
    for i in range(len(beats)):
        plt.axvline(x=beats[i])
    plt.title('Onset Strength Envelope')
    plt.tight_layout()
    plt.grid()
    plt.show()

def plot_evaluation(original, found, original_down, found_down, ose, ellis=False):
    plt.figure(figsize=(10, 4))
    x = np.arange(ose.size)
    to_seconds = Globals.FFT_HOP / Globals.OSE_SAMPLE_RATE
    if ellis:
        # Convert index data to seconds
        found = np.multiply(found, to_seconds)
        found_down = np.multiply(found_down, to_seconds)
        original = np.multiply(original, to_seconds)
        original_down = np.multiply(original_down, to_seconds)
    # Convert x-axis to seconds
    x = x * to_seconds
    plt.plot(x, ose)
    for i in range(len(found)):
        if found[i] in found_down:
            plt.axvline(x=found[i], ymax=1, ymin=0.5, c='orange')
        else:
            plt.axvline(x=found[i], ymax=1, ymin=0.5, c='brown')
    for i in range(len(original)):
        if original[i] in original_down:
            plt.axvline(x=original[i], ymax=0.5, ymin=0, c='orange')
        else:
            plt.axvline(x=original[i], ymax=0.5, ymin=0, c='g')
    plt.title('Found vs. original beats')
    plt.tight_layout()
    plt.xlabel("Seconds")
    plt.ylabel("Normalised OSE values")
    plt.grid()
    plt.show()