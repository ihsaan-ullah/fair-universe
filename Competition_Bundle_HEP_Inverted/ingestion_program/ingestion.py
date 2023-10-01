# ------------------------------------------
# Imports
# ------------------------------------------
from sys import path
import numpy as np
import os
import pandas as pd
from datetime import datetime as dt
import json
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import (
    mean_absolute_error as mae,
    mean_squared_error as mse
)


# ------------------------------------------
# Default Directories
# ------------------------------------------
# # root directory
# root_dir = "./"
# # Input data directory to read training data from
# input_dir = root_dir + "input_data"
# # Output data directory to write predictions to
# output_dir = root_dir + "sample_result_submission"
# # Program directory
# program_dir = root_dir + "ingestion_program"
# # Directory to read submitted submissions from
# submission_dir = root_dir + "sample_code_submission"

# ------------------------------------------
# Codabench Directories
# ------------------------------------------
# Input data directory to read training data from
input_dir = '/app/input_data/'
# Output data directory to write predictions to
output_dir = '/app/output/'
# Program directory
program_dir = '/app/program'
# Directory to read submitted submissions from
submission_dir = '/app/ingested_program'

path.append(input_dir)
path.append(output_dir)
path.append(program_dir)
path.append(submission_dir)

# ------------------------------------------
# Import Model from input data
# ------------------------------------------
from model import Model

# ------------------------------------------
# Import Data from submitted submission
# ------------------------------------------
from data import Data


class Ingestion():

    def __init__(self):

        # Initialize class variables
        self.start_time = None
        self.end_time = None
        self.model = None
        self.train_set = None
        self.validation_sets = []
        self.test_sets = []

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

    def load_data(self):
        print("[*] Loading Data from submission")
        data = Data()
        self.train_set = data.get_train_set()
        self.validation_sets = data.get_validation_sets()
        self.test_sets = data.get_test_sets()

    def initialize_submission(self):
        print("[*] Initializing submitted model")
        self.model = Model(
            train_set=self.train_set,
            validation_sets=self.validation_sets,
            test_sets=self.test_sets
        )

    def fit_submission(self):
        print("[*] Calling fit method of submitted model")
        self.model.fit()

    def predict_submission(self):
        print("[*] Calling predict method of submitted model")
        predicted_dict = self.model.predict()

        self.mu_hats = predicted_dict["mu_hats"]
        self.delta_mu_hat = predicted_dict["delta_mu_hat"]

    def compute_scores(self):
        print("[*] Computing scores")

        C = 0.02
        mus = [test_set['settings']['ground_truth_mu'] for test_set in self.test_sets]
        mu_hats = self.mu_hats
        delta_mus = [mu-mu_hat for mu, mu_hat in zip(mus, mu_hats)]
        delta_mu_hat = self.delta_mu_hat
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
            "mu_hats": mu_hats,
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
        print(f"[*] --- mu (avg): {round(delta_mu_hat, 3)}")
        print(f"[*] --- delta_mu_hat: {round(np.mean(mu_hats), 3)}")
        print(f"[*] --- MAE (mu): {round(mae_mu, 3)}")
        print(f"[*] --- MSE (mu): {round(mse_mu, 3)}")
        print(f"[*] --- MAE (delta mu): {round(mae_delta_mu, 3)}")
        print(f"[*] --- MSE (delta mu): {round(mse_delta_mu, 3)}")
        print(f"[*] --- coverage (mu): {coverage_mu}")
        print(f"[*] --- coverage (C): {coverage_C}")
        print(f"[*] --- score (MAE): {round(score_mae, 3)}")
        print(f"[*] --- score (MSE): {round(score_mse, 3)}")

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

    def save_result(self):
        print("[*] Saving result")

        result_file = os.path.join(output_dir, "result.json")

        with open(result_file, 'w') as f:
            f.write(json.dumps(self.scores_dict, indent=4))


if __name__ == '__main__':

    print("############################################")
    print("### Ingestion Program")
    print("############################################\n")

    # Init Ingestion
    ingestion = Ingestion()

    # Start timer
    ingestion.start_timer()

    # load data
    ingestion.load_data()

    # Initialize submission
    ingestion.initialize_submission()

    # Call fit method of submission
    ingestion.fit_submission()

    # Call predict method of submission
    ingestion.predict_submission()

    # compute scores
    ingestion.compute_scores()

    # Save result
    ingestion.save_result()

    # Stop timer
    ingestion.stop_timer()

    # Show duration
    ingestion.show_duration()

    print("\n----------------------------------------------")
    print("[✔] Ingestions Program executed successfully!")
    print("----------------------------------------------\n\n")
