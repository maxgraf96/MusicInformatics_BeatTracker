import numpy as np
import Constants
import Functions
from scipy.signal import find_peaks

# Weighting factor for the two terms in the objective function
ALPHA = 30

def ellis_07_search(ose, tau_index):
    Constants.TAU_0 = Functions.find_tempo_period_bias()

    # Naming conventions per paper: C is the objective function
    C = np.zeros(ose.size)
    # Indices of previous beats
    P_indices = np.zeros(ose.size, dtype=int)

    # Forward pass
    def calculate_beat(t):
        if t == 0:
            return ose[t]

        # For each point in time store index of previous best match for beat
        startwidth = t - int(tau_index * 2)
        endwidth = t - int(tau_index * 0.5)
        start_idx = startwidth if startwidth > 0 else 0
        end_idx = endwidth if endwidth > 0 else t
        window = end_idx - start_idx

        P_temp = np.zeros(window)
        for j in range(window):
            error = ALPHA * Functions.F_squared_error(t - j, tau_index)
            P_temp[j] = error + C[j]
        previous_best = start_idx + np.argmax(P_temp)

        P_indices[t] = previous_best
        return ose[t] + np.max(P_temp)

    for i in range(ose.size):
        C[i] = calculate_beat(i)

    # look for the largest value of C∗ (which will typically be within "tau_index" of the END of the time range)
    # Stores the index of the final beat
    final_beat = int(np.argmax(C[-tau_index:]) + (C.size - tau_index))

    # Search for the preceding beats
    # Backward pass
    beats = []
    # Stores the currently last beat
    last_beat = P_indices[final_beat]
    beats.append(last_beat)
    while last_beat != 0:
        last_beat = P_indices[last_beat]
        beats.append(last_beat)

    # Reverse so the order is correct
    beats.reverse()

    # Calculate total offset of beats from peaks
    peaks = find_peaks(ose)[0]
    # How far to look back for peak from beat
    look = 24
    for i in range(len(beats)):
        beat_idx = beats[i]
        if beat_idx < look:
            continue
        # Find closest peak
        start = beat_idx - look
        end = beat_idx
        for j in range(start, end):
            if j in peaks:
                beats[i] = j
                continue

    return beats, [], ose