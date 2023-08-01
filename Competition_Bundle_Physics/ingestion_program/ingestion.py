# ------------------------------------------
# Imports
# ------------------------------------------
from sys import path
import os
import numpy as np
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

data_generator_dir = "./Data_Generator"

path.append(input_dir)
path.append(output_dir)
path.append(program_dir)
path.append(submission_dir)
path.append(data_generator_dir)


# ------------------------------------------
# Import Data Gen classes
# ------------------------------------------
from params import Params
from Data_Generator.data_generator_physics import DataGenerator

# ------------------------------------------
# Import Model
# ------------------------------------------
from model import Model


class Ingestion():

    def __init__(self):

        # Initialize class variables
        self.start_time = None
        self.end_time = None
        self.model = None
        self.test_set = None

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

    def configure_data_generator(self):

        print("[*] Configuring Data generator")

        systematics = [{
            "name": "Translation",
            "z_range": [-10, 10],
            "z_angles": [45]
        }]

        # params
        self.data_gen_param = Params(
            pi=0.1,
            nu_1=100000,
            mu_range=[0.9, 1.1],
            systematics=systematics,
            verbose=False
        )

        self.data_gen = DataGenerator

    def load_test_set(self):
        print("[*] Loading Test data")
        test_data_file = os.path.join(input_dir, 'data.csv')
        self.test_set = pd.read_csv(test_data_file)

    def initialize_submission(self):
        print("[*] Initializing submitted model")
        self.model = Model(
            data_gen_param=self.data_gen_param,
            data_gen=self.data_gen,
            test_data=self.test_set
        )

    def fit_submission(self):
        print("[*] Calling fit method of submitted model")
        self.model.fit()

    def predict_submission(self):
        print("[*] Calling predict method of submitted model")
        predicted_dict = self.model.predict()

        self.mu_hat = predicted_dict["mu_hat"]
        self.delta_mu_hat = predicted_dict["delta_mu_hat"]

    def save_result(self):

        print("[*] Saving result")

        result_dict = {
            "delta_mu_hat": self.delta_mu_hat,
            "mu_hat_test": self.mu_hat
        }
        print(f"[*] --- delta_mu_hat: {result_dict['delta_mu_hat']}")
        print(f"[*] --- mu_hat_test: {result_dict['mu_hat_test']}")

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

    # Configure data generator
    ingestion.configure_data_generator()

    # load test set
    ingestion.load_test_set()

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
