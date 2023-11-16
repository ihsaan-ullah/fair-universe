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
# # Root directory
# module_dir = os.path.dirname(os.path.realpath(__file__))
# root_dir = os.path.dirname(module_dir)
# # Input data directory to read training and test data from
# input_dir = os.path.join(root_dir, "input_data")
# # Output data directory to write predictions to
# output_dir = os.path.join(root_dir, "sample_result_submission")
# # Program directory
# program_dir = os.path.join(root_dir, "ingestion_program")
# # Directory to read submitted submissions from
# submission_dir = os.path.join(root_dir, "sample_code_submission")

# ------------------------------------------
# Codabench Directories
# ------------------------------------------
# Root directory
root_dir = "/app"
# Input data directory to read training and test data from
input_dir = os.path.join(root_dir, "input_data")
# Output data directory to write predictions to
output_dir = os.path.join(root_dir, "output")
# Program directory
program_dir = os.path.join(root_dir, "program")
# Directory to read submitted submissions from
submission_dir = os.path.join(root_dir, "ingested_program")

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

from model import Model


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
        train_weights_file = os.path.join(input_dir, 'train', 'weights', "data.weights")

        # read train data
        train_data = pd.read_csv(train_data_file)

        # read train labels
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

    def load_test_set(self, set_index, test_set_index):

        test_data_file = os.path.join(input_dir, 'test', 'set_'+str(set_index), 'data', 'data_'+str(test_set_index)+'.csv')
        test_data = pd.read_csv(test_data_file)

        test_weights_file = os.path.join(input_dir, 'test', 'set_'+str(set_index), 'weights', 'data_'+str(test_set_index)+'.weights')
        with open(test_weights_file) as f:
            test_weights = np.array(f.read().splitlines(), dtype=float)

        test_set = {
            "data": test_data,
            "weights": test_weights
        }
        return test_set

    def init_submission(self):
        print("[*] Initializing Submmited Model")
        self.model = Model(
            train_set=self.train_set,
            systematics=Systematics
        )

    def fit_submission(self):
        print("[*] Calling fit method of submitted model")
        self.model.fit()

    def predict_submission(self):
        print("[*] Calling predict method of submitted model")

        num_sets = 10
        num_test_sets = 100
        set_shuffled_indices = np.random.permutation(num_sets)

        # loop over sets
        for set_shuffled_index in set_shuffled_indices:

            test_set_shuffled_indices = np.random.permutation(num_test_sets)
            test_set_index_mapping = {test_set_shuffled_index: test_set_original_index for test_set_original_index, test_set_shuffled_index in enumerate(test_set_shuffled_indices)}
            mu_hats, delta_mu_hats, p16s, p84s = [], [], [], []

            # loop over test sets
            for test_set_shuffled_index in test_set_shuffled_indices:
                test_set = self.load_test_set(set_index=set_shuffled_index, test_set_index=test_set_shuffled_index)
                predicted_dict = self.model.predict(test_set)
                mu_hats.append(predicted_dict["mu_hat"])
                delta_mu_hats.append(predicted_dict["delta_mu_hat"])
                p16s.append(predicted_dict["p16"])
                p84s.append(predicted_dict["p84"])

                del test_set
                del predicted_dict

            # Reorder the results using the original order of test sets
            mu_hats = [mu_hats[test_set_index_mapping[i]] for i in range(num_test_sets)]
            delta_mu_hats = [delta_mu_hats[test_set_index_mapping[i]] for i in range(num_test_sets)]
            p16s = [p16s[test_set_index_mapping[i]] for i in range(num_test_sets)]
            p84s = [p84s[test_set_index_mapping[i]] for i in range(num_test_sets)]

            print(f"\n[*] mu_hats (avg): {np.mean(mu_hats)}")
            print(f"[*] delta_mu_hats (avg): {np.mean(delta_mu_hats)}")
            print(f"[*] p16 (avg): {np.mean(p16s)}")
            print(f"[*] p84 (avg): {np.mean(p84s)}")

            result_dict = {
                "delta_mu_hats": delta_mu_hats,
                "mu_hats": mu_hats,
                "p16": p16s,
                "p84": p84s
            }

            self.save_result(set_index=set_shuffled_index, result_dict=result_dict)

    def save_result(self, set_index, result_dict):
        print("[*] Saving set result")
        result_file = os.path.join(output_dir, "result_"+str(set_index)+".json")
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

    # initialize submission
    ingestion.init_submission()

    # fit submission
    ingestion.fit_submission()

    # predict submission
    ingestion.predict_submission()

    # save result
    ingestion.save_result()

    # Stop timer
    ingestion.stop_timer()

    # Show duration
    ingestion.show_duration()

    print("\n----------------------------------------------")
    print("[✔] Ingestions Program executed successfully!")
    print("----------------------------------------------\n\n")
