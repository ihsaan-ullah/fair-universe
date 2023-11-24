# ------------------------------
# Imports
# ------------------------------
import os
from sys import path
import numpy as np
from sklearn.preprocessing import StandardScaler


# ------------------------------
# Import local modules
# ------------------------------
module_dir= os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.dirname(module_dir)
path.append(root_dir)
from bootstrap import bootstrap


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
    3) predict: predict mu_hat and delta_mu_hat

    Note:   Add more methods if needed e.g. save model, load pre-trained model etc.
            It is the participant's responsibility to make sure that the submission 
            class is named "Model" and that its constructor arguments remains the same.
            The ingestion program initializes the Model class and calls fit and predict methods
    """

    def __init__(
            self,
            train_set=None,
            systematics=None,
            model_name="one-bin",

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
        self.systematics = systematics

        # Intialize class variables
        self.validation_sets = None
        self.theta_candidates = np.arange(0.9, 0.96, 0.01)
        self.best_theta = 0.9
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

        self._init_model()
        self._predict()
    def predict(self, test_set):
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

        self._test(test_set)

        return {
            "mu_hat": self.mu_hat,
            "delta_mu_hat": self.delta_mu_hat,
            "p16": self.p16,
            "p84": self.p84
        }

    def _init_model(self):
        print("[*] - Intialize model")

    def _fit(self):
        print("[*] --- Fitting Model")
        

    def _predict(self):
        print("[*] --- Predicting")
        train_weights = self.train_set["weights"].copy()
        train_labels = self.train_set["labels"].copy()

        self.gamma_roi = (train_weights*(train_labels)).sum()
        self.beta_roi = (train_weights*(1-train_labels)).sum()   

    def _sigma_asimov_SR(self,mu):
        return mu*self.gamma_roi + self.beta_roi


    def calculate_NLL(self,mu_scan, weight_data):
        sum_data_total_SR = weight_data.sum()
        comb_llr = []
        for i, mu in enumerate(mu_scan):
            hist_llr = (
                -2
                * sum_data_total_SR
                * np.log((self._sigma_asimov_SR(mu) / self._sigma_asimov_SR(1.0)))
            ) + (2 * (self._sigma_asimov_SR(mu) - self._sigma_asimov_SR(1.0)))

            comb_llr.append(hist_llr)

        comb_llr = np.array(comb_llr)
        comb_llr = comb_llr - np.amin(comb_llr)

        return comb_llr
    
    def _compute_result(self,weights):
        mu_scan = np.linspace(0, 5, 100)
        nll = self.calculate_NLL(mu_scan, weights)
        hist_llr = np.array(nll)

        if (mu_scan[np.where((hist_llr <= 1.0) & (hist_llr >= 0.0))].size == 0):
            p16 = 0
            p84 = 0
            mu = 0
        else:
            p16 = min(mu_scan[np.where((hist_llr <= 1.0) & (hist_llr >= 0.0))])
            p84 = max(mu_scan[np.where((hist_llr <= 1.0) & (hist_llr >= 0.0))]) 
            mu = mu_scan[np.argmin(hist_llr)]
        return mu, p16, p84
    
    def _test(self, test_set=None):
        print("[*] - Testing")
        weights_test = test_set["weights"].copy()


        mu_hat, mu_p16, mu_p84 = self._compute_result(weights_test)
        delta_mu_hat = mu_p84 - mu_p16
        
        print(f"[*] --- mu_hat: {mu_hat.mean()}")
        print(f"[*] --- delta_mu_hat: {delta_mu_hat}")
        print(f"[*] --- p16: {mu_p16}")
        print(f"[*] --- p84: {mu_p84}")

        self.mu_hat = mu_hat.mean()
        self.delta_mu_hat = delta_mu_hat
        self.p16 = mu_p16
        self.p84 = mu_p84