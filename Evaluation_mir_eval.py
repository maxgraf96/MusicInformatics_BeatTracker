import os
import numpy as np
import mir_eval
from pathlib import Path

import Evaluation
import Functions
import Main

ANNOTATION_FOLDER = "CreatedAnnotations"

def analyse_all(limit=None):
    """
    Analyse all files in the folder 'BallroomData'
    :param limit: Optionally limit the number of analysed files for quicker run
    :return: None
    """
    f_measures = []
    f_measures_downbeats = []
    files = Path('BallroomData').rglob('*.wav')
    counter = 0
    for file in files:
        f_measure, f_measure_downbeats = analyse(str(file))
        f_measures.append(f_measure)
        f_measures_downbeats.append(f_measure_downbeats)
        counter = counter + 1
        if limit is not None and counter > limit:
            break

    # Print overall results
    print("Mean F-measure: " + str(round(np.mean(f_measures), 2)))
    print("Mean F-measure for downbeats: " + str(round(np.mean(f_measures_downbeats), 2)))


def analyse(file):
    # Get last part of file path for getting the original beat data
    # And replace ".wav" with ".beats"
    filename = file.split('\\')[2][:-4] + ".beats"

    # Get correct beats and downbeat times in seconds
    c_beats, c_downbeats = Evaluation.get_beats_from_file(filename, in_seconds=True)

    # Get estimated beat and downbeat time in seconds
    beats, downbeats = Main.beatTracker(file)

    # Save to 4 *.txt files (2 for beats and 2 for downbeats)
    # Remove ".beats" extension
    filename = filename[:-6]

    # Save estimated beats
    path_est_beats = ANNOTATION_FOLDER + "\\" + filename + "_est.txt"
    path_correct_beats = ANNOTATION_FOLDER + "\\" + filename + "_correct.txt"
    path_est_downbeats = ANNOTATION_FOLDER + "\\" + filename + "_d_est.txt"
    path_correct_downbeats = ANNOTATION_FOLDER + "\\" + filename + "_d_correct.txt"
    save_to_txt(path_est_beats, beats)
    save_to_txt(path_correct_beats, c_beats)
    save_to_txt(path_est_downbeats, downbeats)
    save_to_txt(path_correct_downbeats, c_downbeats)

    # Load data
    reference_beats = mir_eval.io.load_events(path_correct_beats)
    estimated_beats = mir_eval.io.load_events(path_est_beats)
    reference_downbeats = mir_eval.io.load_events(path_correct_downbeats)
    estimated_downbeats = mir_eval.io.load_events(path_est_downbeats)

    # Compare and print score info
    scores = mir_eval.beat.evaluate(reference_beats, estimated_beats)
    scores_downbeats = mir_eval.beat.evaluate(reference_downbeats, estimated_downbeats)

    f_measure = scores['F-measure']
    f_measure_downbeats = scores_downbeats['F-measure']

    print("Evaluation for " + filename + " :")
    print("F-measure: " + str(f_measure))
    print("F-measure for downbeats: " + str(f_measure_downbeats))
    print()

    return f_measure, f_measure_downbeats

def save_to_txt(path, beats):
    # Save to file
    f = open(path, 'w+')  # w+ makes sure that a .txt file is created if it does not exist yet
    # Write beats
    f.writelines([str(beat) + "\n" for beat in beats])
    # Close file
    f.close()

# analyse("BallroomData\\ChaChaCha\\Albums-Cafe_Paradiso-07.wav")
analyse_all(limit=10)