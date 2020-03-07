import librosa
from scipy.signal import butter, filtfilt
from pathlib import Path
import numpy as np
import os
import IPython.display as ipd
from scipy.signal import find_peaks
import Globals
import Main

def estimate_tempo(ose):
    """
    This function uses the precomputed global tempo information parameters to estimate the tempo
    of one piece
    :param ose: The onset strength envelope
    :return: The tactus estimate, the estimated tempo expressed in terms of OSE frames
                and whether duple (True) or triple (False) tempo is assumed
    """
    # The list of tempo period strengths
    TPS = []

    # Calculate autocorrelation of onset strength envelope
    ac = librosa.autocorrelate(ose)

    # For each frame of the autocorrelated onset strength envelope save the
    # weighted value (as seen in the Ellis paper)
    for tau in range(1, ose.size):
        res = autocorrelation_weighting(tau, Globals.TAU_0) * ac[tau]
        TPS.append(res)
    # This index stores the highest value -> this indicates the most likely tempo
    tau_index = np.argmax(TPS)
    # Express in terms of samples
    index_samples = tau_index * Globals.FFT_HOP
    # Express in terms of seconds
    tau_est = index_samples / Globals.OSE_SAMPLE_RATE
    tempo_est = 60 / tau_est

    # Helper functions for calculating the probabilities of duple and triple tempos
    def get_TPS2(tau):
        return TPS[tau] + 0.5 * TPS[2 * tau] + 0.25 * TPS[2 * tau - 1] + 0.25 * TPS[2 * tau + 1]
    def get_TPS3(tau):
        return TPS[tau] + 0.33 * TPS[3 * tau] + 0.33 * TPS[3 * tau - 1] + 0.33 * TPS[3 * tau + 1]

    TPS2 = []
    TPS3 = []
    search_range = 2000  # corresponds to the first 8 seconds of the song
    for tau in range(1, search_range):
        TPS2.append(get_TPS2(tau))
        TPS3.append(get_TPS3(tau))
    tau2 = np.argmax(TPS2)
    tau3 = np.argmax(TPS3)

    max_vals = [TPS[tau_index], TPS2[tau2], TPS3[tau3]]
    metre = np.argmax(max_vals)

    if metre == 0:
        # Duple tempo normal time
        tau_samples = tau_index * Globals.FFT_HOP
        tactus = tau_samples / Globals.OSE_SAMPLE_RATE
        return tactus, tau_index, True
    elif metre == 1:
        # Duple tempo double time
        tau_samples = tau2 * Globals.FFT_HOP
        tactus = (1 / 2) * tau_samples / Globals.OSE_SAMPLE_RATE
        return tactus, tau2, True
    elif metre == 2:
        # Triplet tempo
        tau_samples = tau3 * Globals.FFT_HOP
        tactus = (1 / 3) * tau_samples / Globals.OSE_SAMPLE_RATE
        return tactus, tau3, False


def apply_highpass_filter(sig, sr, cutoff, order):
    """
    Apply a butterworth filter of order "order" at given cutoff frequency
    :param sig: The input signal
    :param sr: The sampling rate of the system
    :param cutoff: The cutoff-frequency of the filter
    :param order: The filter order
    :return: The filtered signal
    """
    # Filter requirements
    T = 1 / sr  # Sampling period
    f_nyquist = sr / 2

    # Normalised cut-off frequency
    normal_cutoff = cutoff / f_nyquist
    # Get filter coefficients
    b, a = butter(N=order, Wn=normal_cutoff, btype='highpass', analog=False)
    return filtfilt(b, a, sig)


def extract_tempo_information_from_beats_file(file):
    """
    Reads a path to a *.beats file, counts the beats and extracts the tempo
    :param file: The name of the *.beats file
    :return: The BPM measure of the file
    """
    with open("BallroomAnnotations-master/" + file) as f:
        lines = f.read().splitlines()
        # Get the last beat
        last_beat = float(lines[-1][:-2])
        # Tempo will be the number of beats, i.e. number of lines divided by the number of the last beat
        tempo_bpm = 60 * len(lines) / last_beat
        return tempo_bpm


def find_tempo_period_bias():
    """
    Find tempo period bias, i.e. the mean period of the tempi over a set of data
    """
    # Check if tempo period bias has been calculated before
    if not os.path.exists("tempo_period_bias.txt") or os.stat("tempo_period_bias.txt").st_size == 0:
        all_files = Path('BallroomAnnotations-master').rglob('*.beats')
        tempos = []
        for file in all_files:
            tempos.append(extract_tempo_information_from_beats_file(file.name))
        mean_bpm = np.mean(tempos)
        mean_seconds = 60 / mean_bpm

        # Save to file
        f = open('tempo_period_bias.txt', 'w+')
        f.write(str(mean_bpm))
        return mean_bpm

    # If it has been calculated, save time and return the recorded value
    # NB: This means that the file has to be deleted or cleared in order to recalculate the tempo period bias
    else:
        f = open('tempo_period_bias.txt', 'r')
        return float(f.read())


def autocorrelation_weighting(tau, TAU_0):
    """
    Helper function for getting the window value for a given tau and the tempo period bias TAU_0
    :param tau: The current point in the autocorrelation function
    :param TAU_0: The precalculated tempo period bias
    :return: The weighted value of the autocorrelation function
    """
    # Value for στ (the width of the weighting curve for the autocorrelation window in octaves)
    weighting_curve = 0.9
    return np.exp((-1 / 2) * ((np.log2(tau / TAU_0) / weighting_curve) ** 2))

def F_squared_error(delta_t, tau):
    """
    Error function for the Ellis-07 method
    :param delta_t: Difference in frames
    :param tau: Current tempo estimate
    :return: The error
    """
    return -np.log10(delta_t / tau) ** 2

