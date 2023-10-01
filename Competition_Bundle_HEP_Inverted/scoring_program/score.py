# ------------------------------------------
# Imports
# ------------------------------------------
import os
import numpy as np
import json
from datetime import datetime as dt

# ------------------------------------------
# Default Directories
# ------------------------------------------
# # root directory
# root_dir = "./"

# # Directory to output computed score into
# output_dir = root_dir + "scoring_output"

# # submitted/predicted lables
# prediction_dir = root_dir + "sample_result_submission"

# # score file to write score into
# score_file = os.path.join(output_dir, 'scores.json')

# ------------------------------------------
# Codabench Directories
# ------------------------------------------
# Directory read predictions and solutions from
input_dir = '/app/input'

# Directory to output computed score into
output_dir = '/app/output/'

# submitted/predicted labels
prediction_dir = os.path.join(input_dir, 'res')

# score file to write score into
score_file = os.path.join(output_dir, 'scores.json')


class Scoring():

    def __init__(self):

        # Initialize class variables
        self.start_time = None
        self.end_time = None
        self.ingestion_results = None

    def start_timer(self):
        self.start_time = dt.now()

    def stop_timer(self):
        self.end_time = dt.now()

    def get_duration(self):
        if self.start_time is None:
            print("[-] Timer was never started. Returning None")
            return None

        if self.end_time is None:
            print("[-] Timer was never stoped. Returning None")
            return None

        return self.end_time - self.start_time

    def show_duration(self):
        print("\n---------------------------------")
        print(f'[✔] Total duration: {self.get_duration()}')
        print("---------------------------------")

    def load_ingestion_results(self):
        print("[*] Reading Ingestion result")

        results_file = os.path.join(prediction_dir, "result.json")
        with open(results_file) as f:
            self.ingestion_results = json.load(f)

    def write_scores(self):
        print("[*] Writing scores")

        with open(score_file, 'w') as f_score:
            f_score.write(json.dumps(self.ingestion_results, indent=4))


if __name__ == '__main__':

    print("############################################")
    print("### Scoring Program")
    print("############################################\n")

    # Init scoring
    scoring = Scoring()

    # Start timer
    scoring.start_timer()

    # Load ingestions results
    scoring.load_ingestion_results()

    # Write scores
    scoring.write_scores()

    # Stop timer
    scoring.stop_timer()

    # Show duration
    scoring.show_duration()

    print("\n----------------------------------------------")
    print("[✔] Scroging Program executed successfully!")
    print("----------------------------------------------\n\n")
