# ------------------------------------------
# Imports
# ------------------------------------------
import os
import numpy as np
import json
from datetime import datetime as dt
from sklearn.metrics import (
    mean_absolute_error as mae,
    mean_squared_error as mse
)

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
# # # Directory read predictions and solutions from
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
        # self.test_labels = None
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

    def load_test_settings(self):
        print("[*] Reading test settings")
        self.test_settings = []
        for i in range(0, 10):
            settings_file = os.path.join(reference_dir, "settings", "data_"+str(i)+".json")
            with open(settings_file) as f:
                self.test_settings.append(json.load(f))

        print("[✔]")

    def load_ingestion_results(self):
        print("[*] Reading predictions")

        results_file = os.path.join(prediction_dir, "result.json")
        with open(results_file) as f:
            self.ingestion_results = json.load(f)

        print("[✔]")

    def compute_scores(self):
        print("[*] Computing scores")

        C = 0.02
        mus = [test_setting['ground_truth_mu'] for test_setting in self.test_settings]
        mu_hats = self.ingestion_results['mu_hats']
        delta_mus = [mu-mu_hat for mu, mu_hat in zip(mus, mu_hats)]
        delta_mu_hat = self.ingestion_results["delta_mu_hat"]
        delta_mu_hats = np.repeat(delta_mu_hat, len(delta_mus))

        # Compute MAE
        mae_mu = self.compute_MAE(mus, mu_hats)
        mae_delta_mu = self.compute_MAE(delta_mus, delta_mu_hats)

        # Compute MSE
        mse_mu = self.compute_MSE(mus, mu_hats)
        mse_delta_mu = self.compute_MSE(delta_mus, delta_mu_hats)

        # Compute Coverage
        coverage_mu = self.compute_coverage(mus, mu_hats, delta_mu_hat, None)
        coverage_C = self.compute_coverage(mus, mu_hats, None, C)

        # Compute Score
        score_mae = self.compute_score(mae_mu, mae_delta_mu)
        score_mse = self.compute_score(mse_mu, mse_delta_mu)

        self.scores_dict = {
            "mu_hat": np.mean(mu_hats),
            "delta_mu_hat": delta_mu_hat,
            "mae_mu": mae_mu,
            "mse_mu": mse_mu,
            "mae_delta_mu": mae_delta_mu,
            "mse_delta_mu": mse_delta_mu,
            "coverage_mu": coverage_mu,
            "coverage_C": coverage_C,
            "score1_mae": score_mae,
            "score1_mse": score_mse
        }
        print(f"[*] --- delta_mu_hat: {round(delta_mu_hat, 3)}")
        print(f"[*] --- MAE (mu): {round(mae_mu, 3)}")
        print(f"[*] --- MSE (mu): {round(mse_mu, 3)}")
        print(f"[*] --- MAE (delta mu): {round(mae_delta_mu, 3)}")
        print(f"[*] --- MSE (delta mu): {round(mse_delta_mu, 3)}")
        print(f"[*] --- coverage (mu): {coverage_mu}")
        print(f"[*] --- coverage (C): {coverage_C}")
        print(f"[*] --- score (MAE): {round(score_mae, 3)}")
        print(f"[*] --- score (MSE): {round(score_mse, 3)}")

        print("[✔]")

    def compute_MAE(self, actual, calculated):
        return mae(actual, calculated)

    def compute_MSE(self, actual, calculated):
        return mse(actual, calculated)

    def compute_coverage(self, mus, mu_hats, delta_mu_hat=None, C=None):

        coverage = 0
        n = len(mus)

        for mu, mu_hat in zip(mus, mu_hats):
            # calculate mu+ and mu-
            if C is None:
                mu_plus = mu_hat + delta_mu_hat
                mu_minu = mu_hat - delta_mu_hat
            else:
                mu_plus = mu_hat + (C/2)
                mu_minu = mu_hat - (C/2)

            # calculate how many times the groundtruth mu is between mu+ and mu-
            if mu >= mu_minu and mu <= mu_plus:
                coverage += 1

        return coverage/n

    def compute_score(self, mu_score, delta_mu_score):

        return mu_score + delta_mu_score

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

    # Load test settings
    scoring.load_test_settings()

    # Load ingestions results
    scoring.load_ingestion_results()

    # Compute Scores
    scoring.compute_scores()

    # Write scores
    scoring.write_scores()

    # Stop timer
    scoring.stop_timer()

    # Show duration
    scoring.show_duration()

    print("\n----------------------------------------------")
    print("[✔] Scroging Program executed successfully!")
    print("----------------------------------------------\n\n")
