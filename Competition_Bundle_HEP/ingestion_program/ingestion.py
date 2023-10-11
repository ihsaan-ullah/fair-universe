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
root_dir = "../"
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

from BDT import Model


class Ingestion():

    def __init__(self):

        # Initialize class variables
        self.start_time = None
        self.end_time = None
        self.model = None
        self.train_set = None
        self.test_sets = []
        self.test_sets_weights = []

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
        train_weights_file = os.path.join(input_dir, 'train', 'weights', "data.weights")

        # read train data
        train_data = pd.read_csv(train_data_file)

        # read trian labels
        with open(train_labels_file, "r") as f:
            train_labels = np.array(f.read().splitlines(), dtype=float)

        # read train settings
        with open(train_settings_file) as f:
            train_settings = json.load(f)
        
        # read train weights
        with open(train_weights_file) as f:
            train_weights = np.array(f.read().splitlines(), dtype=float)


        self.train_set = {
            "data": train_data,
            "labels": train_labels,
            "settings": train_settings,
            "weights": train_weights
        }

    
    def load_test_sets(self):
        print("[*] Loading Test data")
        self.test_sets = []
        for i in range(0, 10):
            test_data_file = os.path.join(input_dir, 'test', 'data', 'data_'+str(i)+'.csv')
            self.test_sets.append(pd.read_csv(test_data_file))

        self.test_sets_weights = []
        for i in range(0, 10):
            test_weights_file = os.path.join(input_dir, 'test', 'weights', 'data_'+str(i)+'.weights')
            test_labels_file = os.path.join(input_dir, 'test', 'labels', 'data_'+str(i)+'.labels')  
            with open(test_weights_file) as f:
                self.test_sets_weights.append(np.array(f.read().splitlines(), dtype=float))
            with open(test_labels_file) as f:
                self.test_labels = np.array(f.read().splitlines(), dtype=float)


    def initialize_submission(self):
        print("[*] Initializing submitted model")
        self.model = Model(
            train_set=self.train_set,
            test_sets=self.test_sets,
            test_sets_weights=self.test_sets_weights,
            test_labels=self.test_labels,
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

    def save_result(self):

        print("[*] Saving result")

        result_dict = {
            "delta_mu_hat": self.delta_mu_hat,
            "mu_hats": self.mu_hats
        }
        print(f"[*] --- delta_mu_hat: {result_dict['delta_mu_hat']}")
        print(f"[*] --- mu_hats (avg): {np.mean(result_dict['mu_hats'])}")

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
