import glob
import os
import numpy as np
import mir_eval
from pathlib import Path

import Evaluation
import Functions
import Main
import Plot

ANNOTATION_FOLDER = "CreatedAnnotations"

def analyse_all(limit=None):
    """
    Analyse all files in the folder 'BallroomData'
    :param limit: Optionally limit the number of analysed files for quicker run
    :return: None
    """
    f_measures = []
    f_measures_downbeats = []
    cemgils = []
    continuities = []
    files = Path('BallroomData').rglob('*.wav')
    number_of_files = len([_ for _ in files])  # This call "invalidates" the pathlib object
    # Reset pathlib object
    files = Path('BallroomData').rglob('*.wav')
    counter = 1
    for file in files:
        # Print progress
        print("Progress: Analysed " + str(counter) + "/" + str(number_of_files) + " files")
        f_measure, f_measure_downbeats, cemgil, continuity = analyse(str(file))
        f_measures.append(f_measure)
        f_measures_downbeats.append(f_measure_downbeats)
        cemgils.append(cemgil)
        continuities.append(continuity)
        counter = counter + 1
        if limit is not None and counter > limit:
            break

    # Print overall results
    print("Mean F-measure (70ms error margin): " + str(round(np.mean(f_measures), 2)))
    print("Mean F-measure for downbeats: " + str(round(np.mean(f_measures_downbeats), 2)))
    print("Mean Cemgil score: " + str(round(np.mean(cemgils), 2)))
    print("Mean continuity score: " + str(round(np.mean(continuities), 2)))



def analyse(file, plot=False):
    """
    Analyse a single "*.wav" audio file
    :param file: Path to the file
    :param plot: If true, a plot is produced and output showing the OSE, the found beats and the correct beats
    :return: Measures: F-measure for beats and downbeats, cemgil and continuity
    """
    # Get last part of file path for getting the original beat data
    # And replace ".wav" with ".beats"
    filename = file.split(os.path.sep)[2][:-4] + ".beats"

    # Get correct beats and downbeat times in seconds
    c_beats, c_downbeats = Evaluation.get_beats_from_file(filename, in_seconds=True)

    # Get estimated beat and downbeat time in seconds
    beats, downbeats, ose, sig = Main.analyse(file)

    # Plot results if specified
    if plot:
        Plot.plot_evaluation(c_beats, beats, c_downbeats, downbeats, ose)

    # Save to 4 *.txt files (2 for beats and 2 for downbeats)
    # Remove ".beats" extension
    filename = filename[:-6]

    # Create annotation folder if not exists
    if not os.path.exists(ANNOTATION_FOLDER):
        os.makedirs(ANNOTATION_FOLDER)

    # Save estimated beats
    path_est_beats = ANNOTATION_FOLDER + os.path.sep + filename + "_est.txt"
    path_correct_beats = ANNOTATION_FOLDER + os.path.sep + filename + "_correct.txt"
    path_est_downbeats = ANNOTATION_FOLDER + os.path.sep + filename + "_d_est.txt"
    path_correct_downbeats = ANNOTATION_FOLDER + os.path.sep + filename + "_d_correct.txt"
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
    # Gaussian error of each estimated beat
    cemgil = scores['Cemgil']
    # Continuity-based scores which compute the proportion of the beat sequence which is continuously correct
    continuity = scores['Any Metric Level Continuous']

    print("Evaluation for " + filename + " :")
    print("F-measure (70ms error margin): " + str(f_measure))
    print("F-measure for downbeats: " + str(f_measure_downbeats))
    # Print empty line
    print()

    return f_measure, f_measure_downbeats, cemgil, continuity

def save_to_txt(path, beats):
    """
    Saves a list of beats to a specified path in the format required by mir_eval.
    If the file does not exist it is created
    If the file exists, values are overwritten
    :param path: The path to save to
    :param beats: The list of beats
    :return:
    """
    # Save to file
    f = open(path, 'w+')  # w+ makes sure that a .txt file is created if it does not exist yet
    # Write beats
    f.writelines([str(beat) + "\n" for beat in beats])
    # Close file
    f.close()

# current_file = "BallroomData\\ChaChaCha\\Albums-Cafe_Paradiso-07.wav"
# current_file = "BallroomData\\ChaChaCha\\Albums-Cafe_Paradiso-06.wav"
# analyse(current_file, plot=True)
# analyse("BallroomData\\ChaChaCha\\Albums-Latin_Jam2-04.wav")
analyse_all(limit=None)