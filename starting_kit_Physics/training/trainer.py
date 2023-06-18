import os
import numpy as np
from model import Model


class Trainer:
    def __init__(self,
                 theta,
                 model_settings,
                 result_dir,
                 model_dir,
                 train_sets,
                 test_sets,
                 settings,
                 write
                 ):

        print("############################################")
        print("### Training Program")
        print("############################################")

        self.model_settings = model_settings
        self.result_dir = result_dir
        self.model_dir = model_dir
        self.train_sets = train_sets
        self.test_sets = test_sets
        self.settings = settings
        self.write = write
        self.theta = theta

    def train(self):

        # Train set
        X_Trains = [train_set["data"] for train_set in self.train_sets]
        Y_Trains = [train_set["labels"] for train_set in self.train_sets]

        # Test set
        X_Tests = [test_set["data"] for test_set in self.test_sets]
        Y_Tests = [test_set["labels"] for test_set in self.test_sets]

        self.results = []
        # ---------------------------------
        # Loop over model settings
        # ---------------------------------
        for model_setting in self.model_settings:
            print("\n--------------------------------------------")
            print("[*] Model : {} --- Preprocessing: {}".format(model_setting["model_name"], model_setting["preprocessing"]))
            print("--------------------------------------------")

            # ---------------------------------
            # Predictions Directory
            # ---------------------------------
            predictions_dir = os.path.join(self.result_dir, model_setting["model_name"])
            # create result directory if not created
            if not os.path.exists(predictions_dir):
                os.mkdir(predictions_dir)

            # ---------------------------------
            # Loop over datasets
            # ---------------------------------
            trained_models = []
            Y_hat_trains, Y_hat_tests = [], []
            mu_hats = []
            for index, _ in enumerate(X_Trains):

                print("\n\tDataset : {}".format(index+1))
                print("\t----------------")

                # model_name
                trained_model_name = self.model_dir + model_setting["model_name"]

                # ---------------------------------
                # Load Model
                # ---------------------------------
                print("\t[*] Loading Model")
                model = Model(
                    model_setting["model_name"],
                    X_Trains[index],
                    Y_Trains[index],
                    X_Tests[index],
                    model_setting["preprocessing"],
                    model_setting["preprocessing_method"],
                    self.theta
                )
                # Load Trained Model
                # model = model.load(trained_model_name)

                # ---------------------------------
                # Train Model
                # ---------------------------------
                # Train model if not trained
                print("\t[*] Training Model")
                if not (model.is_trained):
                    model.fit()

                # ---------------------------------
                # Get Predictions
                # ---------------------------------
                print("\t[*] Get Predictions")
                Y_hat_train = model.predict(X_Trains[index], preprocess=False)
                Y_hat_test = model.predict()
                Y_hat_trains.append(Y_hat_train)
                Y_hat_tests.append(Y_hat_test)

                # ---------------------------------
                # Compute N_ROI from Test
                # ---------------------------------
                print("\t[*] Compute Score")        
                # compute total number of test examples in ROI
                n_test_roi = len(Y_hat_test[Y_hat_test == 1])

                # ---------------------------------
                # compute nu_roi, gamma_roi and beta_roi from train
                # ---------------------------------

                # compute n_roi
                nu_roi = len(Y_hat_train[Y_hat_train == 1])

                # compute gamma_roi
                indexes = np.argwhere(Y_hat_train == 1)
                # get signal class predictions
                signal_predictions = Y_hat_train[indexes]
                gamma_roi = len(signal_predictions[signal_predictions == 1])

                # compute beta_roi
                beta_roi = nu_roi - gamma_roi

                # compute score
                mu_hat = (n_test_roi - beta_roi)/gamma_roi
                mu_hats.append(mu_hat)

                # ---------------------------------
                # Save Predictions
                # ---------------------------------
                print("\t[*] Saving Predictions and Scores")
                # prediction file name
                prediction_name_train = os.path.join(predictions_dir, "train_"+ str(index+1) + ".predictions")
                prediction_name_test = os.path.join(predictions_dir, "test_"+ str(index+1) + ".predictions")

                # save prediction
                self.write(prediction_name_train, Y_hat_train)
                self.write(prediction_name_test, Y_hat_test)

            self.results.append({
                "trained_models": trained_models,
                "Y_hat_trains": Y_hat_trains,
                "Y_hat_tests": Y_hat_tests,
                "mu_hat": mu_hats
            })

    def get_result(self):
        return self.results
