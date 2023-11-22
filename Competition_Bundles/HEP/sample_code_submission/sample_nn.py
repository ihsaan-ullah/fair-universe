import os
from sys import path
import numpy as np
import pandas as pd
from math import sqrt, log
from xgboost import XGBRegressor
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import keras
from keras.models import Sequential
from keras.layers import Dense
# ------------------------------
# Absolute path to submission dir
# ------------------------------
submissions_dir = os.path.dirname(os.path.abspath(__file__))
path.append(submissions_dir)

from bootstrap import *

# ------------------------------
# Constants
# ------------------------------
EPSILON = np.finfo(float).eps


# ------------------------------
# Baseline Model
# ------------------------------
class Model():
    """
    This is a model class to be submitted by the participants in their submission.

    This class should consists of the following functions
    1) init : initialize a classifier
    2) fit : can be used to train a classifier
    3) predict: predict mu_hats,  delta_mu_hat and q1,q2

    Note:   Add more methods if needed e.g. save model, load pre-trained model etc.
            It is the participant's responsibility to make sure that the submission 
            class is named "Model" and that its constructor arguments remains the same.
            The ingestion program initializes the Model class and calls fit and predict methods
    """

    def __init__(
            self,
            train_set=None,
            systematics=None
    ):
        """
        Model class constructor

        Params:
            train_set:
                labelled train set
                
            systematics:
                systematics class

        Returns:
            None
        """

        # Set class variables from parameters
        self.train_set = train_set
        self.systematics = systematics

        # Intialize class variables
        self.validation_sets = None
        self.best_theta = 0.8
        self.scaler = StandardScaler()

    def fit(self):
        """
        Params:
            None

        Functionality:
            this function can be used to train a model using the train set

        Returns:
            None
        """

        self._generate_validation_sets()
        self._init_model()
        self._train()
        self.mu_hat_calc()


    def predict(self, test_set):
        """
        Params:
            None

        Functionality:
           to predict using the test sets

        Returns:
            dict with keys
                - mu_hat
                - delta_mu_hat
                - p16
                - p84
        """

        print("[*] - Testing")
        test_df = test_set['data']
        test_df = self.scaler.transform(test_df)
        Y_hat_test = self._predict(test_df, self.best_theta)

        print("[*] - Computing Test result")
        weights_train = self.train_set["weights"].copy()
        weights_test = test_set["weights"].copy()

        print(f"[*] --- total weight test: {weights_test.sum()}") 
        print(f"[*] --- total weight train: {weights_train.sum()}")
        print(f"[*] --- total weight mu_cals_set: {self.mu_calc_set['weights'].sum()}")

        # get n_roi
        n_roi = self.N_calc_2(weights_test[Y_hat_test == 1])

        mu_hat = (n_roi - self.beta_roi)/self.gamma_roi

        sigma_mu_hat = np.std(mu_hat)

        delta_mu_hat = 2*sigma_mu_hat

        mu_p16 = np.percentile(mu_hat, 16)
        mu_p84 = np.percentile(mu_hat, 84)

        print(f"[*] --- mu_hat: {mu_hat.mean()}")
        print(f"[*] --- delta_mu_hat: {delta_mu_hat}")
        print(f"[*] --- p16: {mu_p16}")
        print(f"[*] --- p84: {mu_p84}")

        return {
            "mu_hat": mu_hat.mean(),
            "delta_mu_hat": delta_mu_hat,
            "p16": mu_p16,
            "p84": mu_p84
        }

    def _init_model(self):
        print("[*] - Intialize Baseline Model (XBM bases Uncertainty Estimator Model)")


        n_cols = self.train_set["data"].shape[1]

        self.model = Sequential()
        self.model.add(Dense(1000, input_dim=n_cols, activation='swish'))
        self.model.add(Dense(1000, activation='swish'))
        self.model.add(Dense(1000, activation='swish'))
        self.model.add(Dense(1000, activation='swish'))
        self.model.add(Dense(2, activation='linear'))
        self.model.compile(loss='mean_squared_error', optimizer='adam')

    def _generate_validation_sets(self):
        print("[*] - Generating Validation sets")

        # Calculate the sum of weights for signal and background in the original dataset
        signal_weights = self.train_set["weights"][self.train_set["labels"] == 1].sum()
        background_weights = self.train_set["weights"][self.train_set["labels"] == 0].sum()

        # Split the data into training and validation sets while preserving the proportion of samples with respect to the target variable
        train_df, mu_calc_set_df, train_labels, mu_calc_set_labels, train_weights, mu_calc_set_weights = train_test_split(
            self.train_set["data"],
            self.train_set["labels"],
            self.train_set["weights"],
            test_size=0.5,
            stratify=self.train_set["labels"]
        )


        # Calculate the sum of weights for signal and background in the training and validation sets
        train_signal_weights = train_weights[train_labels == 1].sum()
        train_background_weights = train_weights[train_labels == 0].sum()
        mu_calc_set_signal_weights = mu_calc_set_weights[mu_calc_set_labels == 1].sum()
        mu_calc_set_background_weights = mu_calc_set_weights[mu_calc_set_labels == 0].sum()

        # Balance the sum of weights for signal and background in the training and validation sets
        train_weights[train_labels == 1] *= signal_weights / train_signal_weights
        train_weights[train_labels == 0] *= background_weights / train_background_weights
        mu_calc_set_weights[mu_calc_set_labels == 1] *= signal_weights / mu_calc_set_signal_weights
        mu_calc_set_weights[mu_calc_set_labels == 0] *= background_weights / mu_calc_set_background_weights

        self.train_df = train_df

        self.train_set = {
            "data": train_df,
            "labels": train_labels,
            "weights": train_weights,
            "settings": self.train_set["settings"]
        }

        self.mu_calc_set = {
                "data": mu_calc_set_df,
                "labels": mu_calc_set_labels,
                "weights": mu_calc_set_weights
            }


        train_signal_weights = train_weights[train_labels == 1].sum()
        train_background_weights = train_weights[train_labels == 0].sum()
        mu_calc_set_signal_weights = mu_calc_set_weights[mu_calc_set_labels == 1].sum()
        mu_calc_set_background_weights = mu_calc_set_weights[mu_calc_set_labels == 0].sum()

        print(f"[*] --- original signal: {signal_weights} --- original background: {background_weights}")
        print(f"[*] --- train signal: {train_signal_weights} --- train background: {train_background_weights}")
        print(f"[*] --- mu_calc_set signal: {mu_calc_set_signal_weights} --- mu_calc_set background: {mu_calc_set_background_weights}")

    def _train(self):

        tes_sets = []

        for i in range(0, 5):

            tes_set = self.train_set['data'].copy()

            tes_set = pd.DataFrame(tes_set)

            tes_set["weights"] = self.train_set["weights"]
            tes_set["labels"] = self.train_set["labels"]

            tes_set = tes_set.sample(frac=0.1, replace=True, random_state=i).reset_index(drop=True)

            # adding systematics to the tes set
            # Extract the TES information from the JSON file
            tes = round(np.random.uniform(0.9, 1.10), 2)
            # tes = 1.0

            syst_set = tes_set.copy()
            data_syst = self.systematics(
                data=syst_set,
                verbose=0,
                tes=tes
            ).data

            data_syst = data_syst.round(3)
            tes_set = data_syst.copy()
            tes_set['tes'] = (tes*10)*2
            tes_sets.append(tes_set)
            del data_syst
            del tes_set

        tes_sets_df = pd.concat(tes_sets)

        train_tes_data = shuffle(tes_sets_df)

        tes_label_1 = train_tes_data.pop('tes')
        tes_label_2 = train_tes_data.pop('labels')
        tes_label = [tes_label_1, tes_label_2]
        tes_label = np.array(tes_label).T
        tes_weights = train_tes_data.pop('weights')

        weights_train = tes_weights.copy()

        class_weights_train = (weights_train[tes_label_2 == 0].sum(), weights_train[tes_label_2 == 1].sum())

        for i in range(len(class_weights_train)):  # loop on B then S target
            # training dataset: equalize number of background and signal
            weights_train[tes_label_2 == i] *= max(class_weights_train) / class_weights_train[i]
            # test dataset : increase test weight to compensate for sampling

        print("[*] --- Training Model")
        train_tes_data = self.scaler.fit_transform(train_tes_data)

        print("[*] --- shape of train tes data", train_tes_data.shape)

        self._fit(train_tes_data, tes_label, weights_train)

        print("[*] --- Predicting Train set")
        self.train_set['predictions'] = (self.train_set['data'], self.best_theta)

        self.train_set['score'] = self._return_score(self.train_set['data'])

        auc_train = roc_auc_score(
            y_true=self.train_set['labels'],
            y_score=self.train_set['score'],
            sample_weight=self.train_set['weights']
        )
        print(f"[*] --- AUC train : {auc_train}")

    def _fit(self, X, y, w):
        print("[*] --- Fitting Model")
        self.model.fit(X, y, sample_weight=w, epochs=100, batch_size=1000, verbose=0)

    def _return_score(self, X):
        y_predict = self.model.predict(X)[:, 1]
        return y_predict

    def _predict(self, X, theta):
        Y_predict = self._return_score(X)
        predictions = np.where(Y_predict > theta, 1, 0)
        return predictions

    def N_calc_2(self, weights, n=100):
        total_weights = []
        for i in range(n):
            bootstrap_weights = bootstrap(weights=weights, seed=42+i)
            total_weights.append(np.array(bootstrap_weights).sum())
        n_calc_array = np.array(total_weights)
        return n_calc_array

    def mu_hat_calc(self):

        self.mu_calc_set['data'] = self.scaler.transform(self.mu_calc_set['data'])
        Y_hat_mu_calc_set = self._predict(self.mu_calc_set['data'], self.best_theta)
        Y_mu_calc_set = self.mu_calc_set['labels']
        weights_mu_calc_set = self.mu_calc_set['weights']

        # compute gamma_roi
        weights_mu_calc_set_signal = weights_mu_calc_set[Y_mu_calc_set == 1]
        weights_mu_calc_set_bkg = weights_mu_calc_set[Y_mu_calc_set == 0]

        Y_hat_mu_calc_set_signal = Y_hat_mu_calc_set[Y_mu_calc_set == 1]
        Y_hat_mu_calc_set_bkg = Y_hat_mu_calc_set[Y_mu_calc_set == 0]

        self.gamma_roi = (weights_mu_calc_set_signal[Y_hat_mu_calc_set_signal == 1]).sum()

        # compute beta_roi
        self.beta_roi = (weights_mu_calc_set_bkg[Y_hat_mu_calc_set_bkg == 1]).sum()
        if self.gamma_roi == 0:
            self.gamma_roi = EPSILON

