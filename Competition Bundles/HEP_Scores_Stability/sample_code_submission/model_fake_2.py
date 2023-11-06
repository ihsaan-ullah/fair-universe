import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

import lightgbm as lgb
from math import sqrt
from math import log
from sys import path

module_dir= os.path.dirname(os.path.realpath(__file__))

root_dir = os.path.dirname(module_dir)

path.append(root_dir)

from bootstrap import bootstrap

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
    3) predict: predict mu_hat and delta_mu_hat

    Note:   Add more methods if needed e.g. save model, load pre-trained model etc.
            It is the participant's responsibility to make sure that the submission 
            class is named "Model" and that its constructor arguments remains the same.
            The ingestion program initializes the Model class and calls fit and predict methods
    """

    def __init__(
            self,
            train_set=None,
            test_sets=[],
            systematics=None,
            model_name="BDT",
            
    ):
        """
        Model class constructor

        Params:
            train_set:
                labelled train set

            test_sets:
                unlabelled test sets

            systematics:
                systematics class

            model_name:

                name of the model, default: BDT


        Returns:
            None
        """

        # Set class variables from parameters

        self.model_name = model_name
        self.train_set = train_set
        self.test_sets = test_sets
        # self.test_sets_weights = []
        # self.test_labels = []
        # for test_set in test_sets:
        #     self.test_sets.append({"data": test_set})

        # for test_set_weights in test_sets_weights:
        #     self.test_sets_weights.append(test_set_weights) 

        # for test_label in test_labels:
        #     self.test_labels.append(test_label)


        self.systematics = systematics

        # Intialize class variables
        self.validation_sets = None
        self.theta_candidates = np.arange(0.9, 0.96, 0.01)
        self.best_theta = 0.9
        self.scaler = StandardScaler()


        # # Hyper params
        # self.num_epochs = 10
        # self.batch_size = 32

    def fit(self):
        """
        Params:
            None

        Functionality:
            this function can be used to train a model using the train set

        Returns:
            None
        """

        self._init_model()

    def predict(self):
        """
        Params:
            None

        Functionality:
           to predict using the test sets

        Returns:
            dict with keys
                - mu_hats
                - delta_mu_hats
                - p16
                - p84
        """

        self._test()
        self._compute_test_result()

        return {
            "mu_hats": self.mu_hats,
            "delta_mu_hats": self.delta_mu_hats,
            "p16": self.p16,
            "p84": self.p84
        }
    
    def _init_model(self):
        print("[*] - Intialize NN model")

        
    def _fit(self):
        print("[*] --- Fitting Model")
    
    def _predict(self):
        print("[*] --- Predicting")
    
    def N_calc_2(self, weights, n=10000):
        total_weights = []
        for i in range(n):
            bootstrap_weights = bootstrap(weights=weights, seed=42+i)
            total_weights.append(np.array(bootstrap_weights).sum())

        n_calc_array = np.array(total_weights)

        return n_calc_array


    def N_calc(self, weights, n=10000):
        total_weights = []
        for i in range(n):
            bootstrap_weights = bootstrap(weights=weights, seed=42+i)
            total_weights.append(np.array(bootstrap_weights).sum())

        n_calc_array = np.array(total_weights)
        guss_mean = np.mean(n_calc_array)
        sigma = np.sqrt(sum((n_calc_array - guss_mean)**2)/n)

        p16 = np.percentile(n_calc_array, 0.16)
        p84 = np.percentile(n_calc_array, 0.84)

        print(f'[*] --- mean N: {guss_mean} --- sigma N: {sigma}')
        print(f'[*] --- p16: {p16} --- p84: {p84}')
        return np.array([guss_mean, sigma, p16, p84])



    def _test(self):
        print("[*] - Testing")



    def _compute_test_result(self):

        print("[*] - Computing Test result")

        mu_hats = []
        self.test_set_delta_mu_hats = []
        p16s, p84s = [], []

        for test_set in self.test_sets:

            weights_train = self.train_set["weights"].copy()
            weights_test = test_set["weights"].copy()

            s_train = weights_train[self.train_set['labels'] == 1].sum()

            # Compute mu_hat
            N_ = self.N_calc_2(weights_test)


            N = weights_train.sum()

            mu_hat = ((N_ - N)/s_train) + 1 
            

            sigma_mu_hat = np.std(mu_hat)


            delta_mu_hat = 2*sigma_mu_hat



            mu_p16 = np.percentile(mu_hat, 16)
            mu_p84 = np.percentile(mu_hat, 84)

            p16s.append(mu_p16)
            p84s.append(mu_p84)
            self.test_set_delta_mu_hats.append(delta_mu_hat)

            mu_hats.append(np.mean(mu_hat))

            print(f"[*] --- delta_mu_hat: {delta_mu_hat}")
            print(f"[*] --- p16: {mu_p16} --- p84: {mu_p84}")


        print("\n")

        # Save mu_hat from test
        self.mu_hats = mu_hats
        self.p16 = p16s
        self.p84 = p84s

        # right now we are using test set delta mu hat, 
        self.delta_mu_hats = self.test_set_delta_mu_hats

        # uncomment the next line if you want to use validation delta mu hat
        # self.delta_mu_hats = self.validation_delta_mu_hats

        print(f"[*] --- mu_hats (avg): {np.mean(mu_hats)}")
        print(f"[*] --- mu_hats (std): {np.std(mu_hats)}")
        print(f"[*] --- delta_mu_hat (avg): {np.round(np.mean(self.delta_mu_hats), 4)}")
        print(f"[*] --- delta_mu_hat (std): {np.round(np.std(self.delta_mu_hats), 4)}")
        print(f"[*] --- p16 (avg): {np.round(np.mean(self.p16), 4)}")
