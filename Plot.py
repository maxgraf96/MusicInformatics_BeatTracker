import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display

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

def plot_evaluation(original, found, original_down, found_down, ose):
    plt.figure(figsize=(10, 4))
    x = np.arange(ose.size)
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
    plt.title('Found vs. Original beats')
    plt.tight_layout()
    plt.grid()
    plt.show()