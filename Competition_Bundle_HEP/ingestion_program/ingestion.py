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


# ------------------------------------------
# Default Directories
# ------------------------------------------
# root directory
root_dir = "./"
# Input data directory to read training data from
input_dir = root_dir + "input_data"
# Output data directory to write predictions to
output_dir = root_dir + "sample_result_submission"
# Program directory
program_dir = root_dir + "ingestion_program"
# Directory to read submitted submissions from
submission_dir = root_dir + "sample_code_submission"

# ------------------------------------------
# Codabench Directories
# ------------------------------------------
# # Input data directory to read training data from
# input_dir = '/app/input_data/'
# # Output data directory to write predictions to
# output_dir = '/app/output/'
# # Program directory
# program_dir = '/app/program'
# # Directory to read submitted submissions from
# submission_dir = '/app/ingested_program'

path.append(input_dir)
path.append(output_dir)
path.append(program_dir)
path.append(submission_dir)


# ------------------------------------------
# Import Systamtics
# ------------------------------------------
from systematics import Systematics

# ------------------------------------------
# Import Model
# ------------------------------------------
from NN import Model


class Ingestion():

    def __init__(self):

        # Initialize class variables
        self.start_time = None
        self.end_time = None
        self.model = None
        self.train_set = None
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

    def load_train_set(self):
        print("[*] Loading Train data")

        train_data_file = os.path.join(input_dir, 'train', 'data', 'data.csv')
        train_labels_file = os.path.join(input_dir, 'train', 'labels', "data.labels")
        train_settings_file = os.path.join(input_dir, 'train', 'settings', "data.json")

        # read train data
        train_data = pd.read_csv(train_data_file)

        # read trian labels
        with open(train_labels_file, "r") as f:
            train_labels = np.array(f.read().splitlines(), dtype=float)

        # read train settings
        with open(train_settings_file) as f:
            train_settings = json.load(f)

        self.train_set = {
            "data": train_data,
            "labels": train_labels,
            "settings": train_settings
        }

    def load_test_sets(self):
        print("[*] Loading Test data")
        self.test_sets = []
        for i in range(0, 10):
            test_data_file = os.path.join(input_dir, 'test', 'data', 'data_'+str(i)+'.csv')
            self.test_sets.append(pd.read_csv(test_data_file))

    def initialize_submission(self):
        print("[*] Initializing submitted model")
        self.model = Model(
            train_set=self.train_set,
            test_sets=self.test_sets,
            systematics=Systematics
        )

    def fit_submission(self):
        print("[*] Calling fit method of submitted model")
        self.model.fit()

    def predict_submission(self):
        print("[*] Calling predict method of submitted model")
        predicted_dict = self.model.predict()

        self.mu_hats = predicted_dict["mu_hats"]
        self.delta_mu_hat = predicted_dict["delta_mu_hat"]
        
        self.q_1 = predicted_dict["q_1"]
        self.q_2 = predicted_dict["q_2"]

    def save_result(self):

        print("[*] Saving result")

        result_dict = {
            "delta_mu_hat": self.delta_mu_hat,
            "mu_hats": self.mu_hats,
            "q_1": self.q_1,
            "q_2": self.q_2
        }
        }
        print(f"[*] --- delta_mu_hat: {result_dict['delta_mu_hat']}")
        print(f"[*] --- mu_hats (avg): {np.mean(result_dict['mu_hats'])}")
        print(f"[*] --- q_1 (avg): {np.mean(result_dict['q_1'])}")
        print(f"[*] --- q_2 (avg): {np.mean(result_dict['q_2'])}")  

        result_file = os.path.join(output_dir, "result.json")

        with open(result_file, 'w') as f:
            f.write(json.dumps(result_dict, indent=4))


if __name__ == '__main__':

    print("############################################")
    print("### Ingestion Program")
    print("############################################\n")

    # Init Ingestion
    ingestion = Ingestion()

    # Start timer
    ingestion.start_timer()

    # load test set
    ingestion.load_train_set()

    # load test set
    ingestion.load_test_sets()

    # Initialize submission
    ingestion.initialize_submission()

    # Call fit method of submission
    ingestion.fit_submission()

    # Call predict method of submission
    ingestion.predict_submission()

    # Save result
    ingestion.save_result()

    # Stop timer
    ingestion.stop_timer()

    # Show duration
    ingestion.show_duration()

    print("\n----------------------------------------------")
    print("[✔] Ingestions Program executed successfully!")
    print("----------------------------------------------\n\n")
