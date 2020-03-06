from pathlib import Path

import numpy as np

import Functions
import Main
import Plot
from Constants import OSE_SAMPLE_RATE, FFT_HOP

# Allowed inaccuracy window in ms
from Ellis_07_Search import ellis_07_search

MARGIN = 70
N = 50

def get_beats_from_file(file):
    """
    Extract beat information from *.beats file
    :param file: The file
    :return: List of beats and list of downbeats
    """
    with open("BallroomAnnotations-master/" + file, 'r') as f:
        beats = []
        downbeats = []
        for line in f:
            time, beat_nr = tuple(line.rstrip().split())
            # Convert beat times to compare with results
            time = int(float(time) * OSE_SAMPLE_RATE / FFT_HOP)
            beats.append(time)
            # Append downbeat
            if int(beat_nr) == 1:
                downbeats.append(time)
            if not line:
                break  # End of file reached

        return beats, downbeats

def evaluate_file(file, ellis=False):
    """
    Analyse a given file and calculate its f-measure
    :param file: The path to the *.wav file
    :param ellis: If set to true, the algorithm specified by Ellis 2007 will be used for the beat calculation
    :return: The correct beats (read from *.beats file), the found beats, the correct downbeats, the downbeats,
    the onset strength envelope, accuracy for true positives of beats and downbeats, f-measure for beats and downbeats
    """
    # Get last part of file path for getting the original beat data
    # And replace ".wav" with ".beats"
    filename = file.split('\\')[2][:-4] + ".beats"
    c_beats, c_downbeats = get_beats_from_file(filename)
    beats, downbeats, ose, sig = Main.analyse(file)

    # Use the algorithm specified by Ellis
    if ellis:
        tau_est, tau_index, is_duple_tempo = Functions.estimate_tempo(ose)
        beats, downbeats, ose = ellis_07_search(ose, tau_index)

    # Calculate accuracy
    # Convert error margin to ose frame units and divide by 2 (half of specified allowed in each direction)
    # So 70 ms allow for 35ms before and 35ms after
    margin = np.ceil((MARGIN * 0.001) * OSE_SAMPLE_RATE / FFT_HOP / 2)
    # True positives, false positives, false negatives, for beats and downbeats respectively
    TP = []
    FP = []
    FN = []
    TP_D = []
    FP_D = []
    FN_D = []
    # Get TP
    for i in range(len(beats)):
        # Get allowed values for current beat
        space = np.arange(beats[i] - margin, beats[i] + margin, 1, dtype=int)
        for allowed in space:
            if allowed in c_beats:
                TP.append(beats[i])
                if beats[i] in downbeats and allowed in c_downbeats:
                    TP_D.append(beats[i])

    # Get FN: Beat should exist, but is missing
    for i in range(len(c_beats)):
        space = np.arange(c_beats[i] - margin, c_beats[i] + margin, 1, dtype=int)
        found = False
        for allowed in space:
            if allowed in beats:
                found = True
                break
        if not found:
            FN.append(c_beats[i])

    # Get FP: Beat should not exist, but something was found
    for i in range(len(beats)):
        space = np.arange(beats[i] - margin, beats[i] + margin, 1, dtype=int)
        found = False
        for allowed in space:
            if allowed in c_beats:
                found = True
                break
        if not found:
            # Increase false positives
            FP.append(beats[i])

    # Get FN Downbeats: Beat should be downbeat, but is not
    for c_downbeat in c_downbeats:
        start = c_downbeat - margin if c_downbeat - margin >= 0 else 0
        end = c_downbeat + margin if c_downbeat + margin < ose.size else ose.size
        space = np.arange(start, end, 1, dtype=int)
        found = False
        for candidate in space:
            if candidate in downbeats:
                found = True
                break
        if not found:
            FN_D.append(c_downbeat)

    # Get FP Downbeats: Beat should not be downbeat, but was classified as such
    for db_idx in downbeats:
        space = np.arange(db_idx - margin, db_idx + margin, 1, dtype=int)
        misclassified = True
        for allowed in space:
            if allowed in c_downbeats:
                misclassified = False
        if misclassified:
            FP_D.append(db_idx)

    # Calculate F-measure for beats
    if len(TP) == 0:
        f_measure = 0
    else:
        precision = len(TP) / (len(TP) + len(FP))
        recall = len(TP) / (len(TP) + len(FN))
        f_measure = 2 * ((precision * recall) / (precision + recall))

    # Calculate F-measure for downbeats
    if len(TP_D) == 0:
        f_measure_d = 0
    else:
        precision_d = len(TP_D) / (len(TP_D) + len(FP_D))
        recall_d = len(TP_D) / (len(TP_D) + len(FN_D))
        f_measure_d = 2 * ((precision_d * recall_d) / (precision_d + recall_d))

    # Calculate avg score for TP (beats and downbeats)
    acc_TP = len(TP) / len(c_beats)
    acc_TP_down = len(TP_D) / len(c_downbeats)

    # Print evaluation
    print("TP accuracy for " + filename[:-6] + ".wav: " + str(round(acc_TP, 2)))
    print("TP downbeat accuracy: " + str(round(acc_TP_down, 2)))
    print("F-measure: " + str(round(f_measure, 2)))
    print("F-measure for downbeats: " + str(round(f_measure_d, 2)))
    return c_beats, beats, c_downbeats, downbeats, ose, acc_TP, acc_TP_down, f_measure, f_measure_d

def evaluate_all():
    # Evaluate N files
    accuracies = []
    accuracies_d = []
    f_measures = []
    f_measures_d = []
    files = Path('BallroomData').rglob('*.wav')
    counter = 0
    for file in files:
        correct_beats, beats, correct_downbeats, downbeats, ose, acc_TP, acc_TP_down, f_measure, f_measure_d = evaluate_file(str(file))
        accuracies.append(acc_TP)
        accuracies_d.append(acc_TP_down)
        f_measures.append(f_measure)
        f_measures_d.append(f_measure_d)
        counter = counter + 1
        if counter > N:
            break

    # Info
    print("Mean TP accuracy over " + str(N) + " files: " + str(round(np.mean(accuracies), 2)))
    print("Mean TP accuracy for downbeats: " + str(round(np.mean(accuracies_d), 2)))
    print("Mean F-measure: " + str(round(np.mean(f_measures), 2)))
    print("Mean F-measure for downbeats: " + str(round(np.mean(f_measures_d), 2)))


# current_file = "BallroomData\\ChaChaCha\\Albums-Cafe_Paradiso-07.wav"
# current_file = "BallroomData\\ChaChaCha\\Albums-Cafe_Paradiso-06.wav"
# current_file = "BallroomData\\ChaChaCha\\Albums-Cafe_Paradiso-08.wav"
# current_file = "BallroomData\\Rumba-Misc\\Media-103511.wav"
# correct, found, correct_downbeats, found_downbeats, ose, accuracy, accuracy_down, f_measure, f_measure_d = evaluate_file(current_file)
# Plot.plot_evaluation(correct, found, correct_downbeats, found_downbeats, ose)
evaluate_all()