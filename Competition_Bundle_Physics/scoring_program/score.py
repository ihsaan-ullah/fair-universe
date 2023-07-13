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

# root directory
root_dir = "./"

# Directory to output computed score into
output_dir = root_dir + "scoring_output"

# reference data (test labels)
reference_dir = os.path.join(root_dir, "reference_data")

# submitted/predicted lables
prediction_dir = root_dir + "sample_result_submission"

# score file to write score into
score_file = os.path.join(output_dir, 'scores.json')

# ------------------------------------------
# Codabench Directories
# ------------------------------------------
# # Directory read predictions and solutions from
# input_dir = '/app/input'

# # Directory to output computed score into
# output_dir = '/app/output/'

# # reference data (test labels)
# reference_dir = os.path.join(input_dir, 'ref')  # Ground truth data

# # submitted/predicted labels
# prediction_dir = os.path.join(input_dir, 'res')

# # score file to write score into
# score_file = os.path.join(output_dir, 'scores.json')


class Scoring():

    def __init__(self):

        # Initialize class variables
        self.start_time = None
        self.end_time = None
        self.test_labels = None
        self.test_settings = None
        self.ingestion_results = None

        self.scores_dict = {}

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

    def load_test_labels(self):
        print("[*] Reading test labels")

        labels_file = os.path.join(reference_dir, "data.labels")
        with open(labels_file, "r") as f:
            self.test_labels = np.array(f.read().splitlines(), dtype=float)

        print("[✔]")
        pass

    def load_test_settings(self):
        print("[*] Reading test settings")

        settings_file = os.path.join(reference_dir, "settings.json")
        with open(settings_file) as f:
            self.test_settings = json.load(f)

        print("[✔]")

    def load_ingestion_results(self):
        print("[*] Reading predictions")

        results_file = os.path.join(prediction_dir, "result.json")
        with open(results_file) as f:
            self.ingestion_results = json.load(f)

        print("[✔]")

    def compute_score(self):
        print("[*] Computing scores")

        # delta mu = mu - mu_hat_test
        delta_mu = np.abs(self.test_settings["ground_truth_mu"] - self.ingestion_results["mu_hat_test"])
        delta_delta_mu = np.abs(delta_mu - self.ingestion_results["delta_mu_hat"])
        score = delta_mu + delta_delta_mu
        self.scores_dict = {
            "delta_mu": delta_mu,
            "delta_delta_mu": delta_delta_mu,
            "score": score
        }
        print(f"[*] --- delta_mu: {delta_mu}")
        print(f"[*] --- delta_delta_mu: {delta_delta_mu}")
        print(f"[*] --- score: {score}")

        print("[✔]")

    def write_scores(self):
        print("[*] Writing scores")

        with open(score_file, 'w') as f_score:
            f_score.write(json.dumps(self.scores_dict, indent=4))

        print("[✔]")
        pass


if __name__ == '__main__':

    print("############################################")
    print("### Scoring Program")
    print("############################################\n")

    # Init scoring
    scoring = Scoring()

    # Start timer
    scoring.start_timer()

    # Load test labels
    scoring.load_test_labels()

    # Load test settings
    scoring.load_test_settings()

    # Load ingestions results
    scoring.load_ingestion_results()

    # Compute Scores
    scoring.compute_score()

    # Write scores
    scoring.write_scores()

    # Stop timer
    scoring.stop_timer()

    # Show duration
    scoring.show_duration()

    print("\n----------------------------------------------")
    print("[✔] Scroging Program executed successfully!")
    print("----------------------------------------------\n\n")
