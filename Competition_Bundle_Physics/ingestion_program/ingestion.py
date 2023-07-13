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

data_generator_dir = "../Data_Generator"

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
path.append(data_generator_dir)

# ------------------------------------------
# Import Data Gen classes
# ------------------------------------------
from params import Params
from setting import Setting
from data_generator_physics import DataGenerator

# ------------------------------------------
# Import Model
# ------------------------------------------
from model import Model


class Ingestion():

    def __init__(self):

        # Initialize class variables
        self.start_time = None
        self.end_time = None
        self.train_set = None
        self.validation_sets = None
        self.theta_candidates = np.arange(-10, 3)
        self.model = None
        self.best_theta = None
        self.best_validation_set = None
        self.test_set = {}

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

    def generate_train_set(self):
        print("[*] Generating Train set")

        # Generate train params
        data_gen_param = Params(
            pi=0.1,
            nu_1=100000,
            mu_range=[1.0, 1.0],
            systematics=[],
            verbose=False
        )

        # Generate train settings
        data_gen_setting = Setting(
            params=data_gen_param
        ).get_setting()

        # Generate train data
        data_gen = DataGenerator(settings_dict=data_gen_setting)
        data_gen.generate_data()
        self.train_set = data_gen.get_data()

        print("[✔]")

    def generate_validation_sets(self):
        print("[*] Generating Validation sets")

        systematics = [{
            "name": "Translation",
            "z_range": [-10, 10],
            "z_angles": [45]
        }]

        self.validation_sets = []

        # Loop 10 times to generate 10 validation sets
        for i in range(0, 10):

            # Generate validation params
            data_gen_param = Params(
                pi=0.1,
                nu_1=100000,
                mu_range=[1.0, 1.0],
                systematics=systematics,
                verbose=False
            )

            # Generate validation settings
            data_gen_setting = Setting(
                params=data_gen_param
            ).get_setting()

            # Generate validation data
            data_gen = DataGenerator(settings_dict=data_gen_setting)
            data_gen.generate_data()
            valid_set = data_gen.get_data()

            self.validation_sets.append(valid_set)

        print("[✔]")

    def train(self):

        if self.train_set is None:
            raise ValueError("[-] Train set is not generated! Call `generate_train_set` first")

        print("[*] Train a classifier")

        print("[*] --- Loading Model")
        self.model = Model()

        print("[*] --- Training Model")
        self.model.fit(self.train_set['data'], self.train_set['labels'])

        print("[*] --- Predicting Train set")
        self.train_set['predictions'] = self.model.predict(self.train_set['data'], 0)

        print("[✔]")

    def validate(self):

        if self.validation_sets is None:
            raise ValueError("[-] Validation sets are not generated! Call `generate_validation_sets` first.")

        print("[*] Validate trained classifier")

        validation_set_thetas = []
        validation_set_ROIs = []
        validation_set_predictions = []

        # loop over validation sets
        for valid_set in self.validation_sets:

            validation_ROIs = []
            validation_predictions = []

            # Loop over theta candidates
            # try each theta on each validation set
            # choose best theta for each validation set
            for theta in self.theta_candidates:

                # Get predictions from trained model
                predictions = self.model.predict(valid_set['data'], theta)

                # get N_ROI from predictions
                validation_ROIs.append(self._get_N_ROI(predictions))

                # save predictions
                validation_predictions.append(predictions)

            # save for this validation set
            # - best NROI
            # - theta
            # - predictions
            index_of_largest_N_ROI = np.argmax(validation_ROIs)
            validation_set_ROIs.append(validation_ROIs[index_of_largest_N_ROI])
            validation_set_thetas.append(self.theta_candidates[index_of_largest_N_ROI])
            validation_set_predictions.append(validation_predictions[index_of_largest_N_ROI])

        # get index of the best NROI
        best_index = np.argmax(validation_set_ROIs)
        # choose theta with best NROI
        self.best_theta = validation_set_thetas[best_index]
        # choose validation set with best NROI
        self.best_validation_set = self.validation_sets[best_index]
        self.best_validation_set["predictions"] = validation_set_predictions[best_index]

        print(f"[*] --- Best theta : {self.best_theta}")
        print("[✔]")

    def _get_N_ROI(self, predictions):
        return len(predictions[predictions == 1])

    def compute_validation_result(self):

        print("[*] Computing Validation result")

        Y_hat_train = self.train_set["predictions"]
        Y_train = self.train_set["labels"]
        Y_hat_valid = self.best_validation_set["predictions"]

        n_roi = len(Y_hat_valid[Y_hat_valid == 1])

        # get region of interest
        roi_indexes = np.argwhere(Y_hat_train == 1)
        roi_points = Y_train[roi_indexes]
        # compute nu_roi
        nu_roi = len(roi_points)

        # compute gamma_roi
        indexes = np.argwhere(roi_points == 1)

        # get signal class predictions
        signal_predictions = roi_points[indexes]
        gamma_roi = len(signal_predictions)

        # compute beta_roi
        beta_roi = nu_roi - gamma_roi

        # Compute mu_hat
        mu_hat = (n_roi - beta_roi)/gamma_roi

        print(f"[*] --- mu: {self.best_validation_set['settings']['ground_truth_mu']}")
        print(f"[*] --- mu_hat: {mu_hat}")

        # Compute delta_mu_hat
        self.best_validation_set['delta_mu_hat'] = self.best_validation_set["settings"]["ground_truth_mu"] - mu_hat

        print("[✔]")

    def load_test_set(self):
        print("[*] Loading Test set")
        test_data_file = os.path.join(input_dir, 'data.csv')
        self.test_set['data'] = pd.read_csv(test_data_file)
        print("[✔]")

    def test(self):
        print("[*] Testing")
        # Get predictions from trained model
        self.test_set['predictions'] = self.model.predict(self.test_set['data'], self.best_theta)
        print("[✔]")

    def compute_test_result(self):

        print("[*] Computing Test result")

        Y_hat_train = self.train_set["predictions"]
        Y_train = self.train_set["labels"]
        Y_hat_test = self.test_set["predictions"]

        n_roi = len(Y_hat_test[Y_hat_test == 1])

        # get region of interest
        roi_indexes = np.argwhere(Y_hat_train == 1)
        roi_points = Y_train[roi_indexes]
        # compute nu_roi
        nu_roi = len(roi_points)

        # compute gamma_roi
        indexes = np.argwhere(roi_points == 1)

        # get signal class predictions
        signal_predictions = roi_points[indexes]
        gamma_roi = len(signal_predictions)

        # compute beta_roi
        beta_roi = nu_roi - gamma_roi

        # Compute mu_hat
        mu_hat = (n_roi - beta_roi)/gamma_roi

        # Save mu_hat from test
        self.test_set['mu_hat'] = mu_hat

        print(f"[*] --- mu_hat: {mu_hat}")

        print("[✔]")

    def save_result(self):

        print("[*] Saving result")

        result_dict = {
            "delta_mu_hat": self.best_validation_set['delta_mu_hat'],
            "mu_hat_test": self.test_set['mu_hat']
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

    # Generate Train set
    ingestion.generate_train_set()

    # Generate Validation sets
    ingestion.generate_validation_sets()

    # Train classifier
    ingestion.train()

    # Validate using multiple validations sets
    # choose best theta from theta candidates
    ingestion.validate()

    # Compute mu and delta mu from validation set
    ingestion.compute_validation_result()

    # load test set
    ingestion.load_test_set()

    # Test using the best theta
    ingestion.test()

    # Compute mu and delta mu from test set
    ingestion.compute_test_result()

    # Save result
    ingestion.save_result()

    # Stop timer
    ingestion.stop_timer()

    # Show duration
    ingestion.show_duration()

    print("\n----------------------------------------------")
    print("[✔] Ingestions Program executed successfully!")
    print("----------------------------------------------\n\n")
