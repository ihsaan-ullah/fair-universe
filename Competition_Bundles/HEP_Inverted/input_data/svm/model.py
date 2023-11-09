import numpy as np
import pandas as pd
from sklearn.svm import SVC

EPSILON = np.finfo(float).eps


# ------------------------------
# Baseline Model
# ------------------------------
class Model:
    def __init__(
            self,
            train_set=None,
            validation_sets=[],
            test_sets=[]
    ):

        # Set class variables from parameters
        self.train_set = train_set
        self.validation_sets = validation_sets
        self.test_sets = test_sets

        # Intialize class variables
        self.model_name = "SVM"
        self.theta_candidates = np.arange(-10, 10)
        self.best_theta = None

        print(f"[*] - Model name: {self.model_name}")

    def fit(self):

        # Train classifier
        self.train()

        # Choose best theta
        self.choose_theta()

        # Validate using multiple validations sets
        self.validate()

        # Compute mu and delta mu from validation set
        self.compute_validation_result()

    def predict(self):

        # Test using the best theta
        self.test()

        # Compute mu and delta mu from test set
        self.compute_test_result()

        return {
            "mu_hats": self.mu_hats,
            "delta_mu_hat": self.delta_mu_hat,
            "predictions": [test_set["predictions"] for test_set in self.test_sets],
            "decisions": [test_set["decisions"] for test_set in self.test_sets],
            "theta": self.best_theta
        }

    def _init_model(self):
        self.clf = SVC(kernel='linear')

    def _fit(self, X, y):
        weights = None
        if "New_Weight" in X.columns:
            weights = X["New_Weight"]
            X = X.drop("New_Weight", axis=1)
        if "Weight" in X.columns:
            X = X.drop("Weight", axis=1)

        self.clf.fit(X, y, sample_weight=weights)

    def _predict(self, X, theta):
        if "New_Weight" in X.columns:
            X = X.drop("New_Weight", axis=1)
        if "Weight" in X.columns:
            X = X.drop("Weight", axis=1)
        predictions = np.zeros(X.shape[0])
        decisions = self._decision_function(X, theta)

        # class 1 -> if decision function  > theta
        # class 0 -> otherwise
        predictions = (decisions > theta).astype(int)
        return predictions

    def _decision_function(self, X, theta):

        if "New_Weight" in X.columns:
            X = X.drop("New_Weight", axis=1)
        if "Weight" in X.columns:
            X = X.drop("Weight", axis=1)

        # decision funciton: output between -inf and +inf
        # 0 is the neutral point between the 2 classes which is on the decision boundary

        decisions = None
        if self.model_name in ["SVM", "XGBoost"]:
            decisions = self.clf.decision_function(X)
        else:
            # to make the output of both decision funciton and predict proba the same
            # we transform the predict proba output
            predicted_score = self.clf.predict_proba(X)
            # Transform with log
            predicted_score = -np.log((1/(predicted_score+EPSILON))-1)
            decisions = predicted_score[:, 1]

        # subtract the threshold from decisions because we want the decision boundary to be at 0
        return decisions - theta

    def get_meta_validation_set(self):

        meta_validation_data = []
        meta_validation_labels = []

        for valid_set in self.validation_sets:
            meta_validation_data.append(valid_set['data'])
            meta_validation_labels = np.concatenate((meta_validation_labels, valid_set['labels']))

        return {
            'data': pd.concat(meta_validation_data),
            'labels': meta_validation_labels
        }

    def train(self):

        print("[*] - Training classifier")

        print("[*] --- Loading Model")
        self._init_model()

        print("[*] --- Training Model")
        self._fit(self.train_set['data'], self.train_set['labels'])

        print("[*] --- Predicting Train set")
        self.train_set['predictions'] = self._predict(self.train_set['data'], 0)

    def choose_theta(self):

        print("[*] Choose best theta")

        meta_validation_set = self.get_meta_validation_set()
        theta_sigma_squared = []

        # Loop over theta candidates
        # try each theta on meta-validation set
        # choose best theta
        for theta in self.theta_candidates:

            # Get predictions from trained model
            Y_hat_valid = self._predict(meta_validation_set['data'], theta)
            Y_valid = meta_validation_set["labels"]

            # get region of interest
            roi_indexes = np.argwhere(Y_hat_valid == 1).flatten()
            roi_points = Y_valid[roi_indexes]
            # compute nu_roi
            # nu_roi = len(roi_points)
            nu_roi = meta_validation_set['data'].iloc[roi_indexes]["Weight"].sum()

            # compute gamma_roi
            # indexes = np.argwhere(roi_points == 1)
            indexes = np.argwhere(roi_points == 1).flatten()

            # get signal class predictions
            # signal_predictions = roi_points[indexes]
            # gamma_roi = len(signal_predictions)
            gamma_roi = meta_validation_set['data'].iloc[indexes]["Weight"].sum()

            # compute beta_roi
            # beta_roi = nu_roi - gamma_roi

            # Compute sigma squared mu hat
            sigma_squared_mu_hat = nu_roi/np.square(gamma_roi)

            # get N_ROI from predictions
            theta_sigma_squared.append(sigma_squared_mu_hat)

        # Choose theta with min sigma squared
        try:
            index_of_least_sigma_squared = np.nanargmin(theta_sigma_squared)
        except:
            print("[!] - WARNING! All sigma squared are nan")
            index_of_least_sigma_squared = np.argmin(theta_sigma_squared)

        self.best_theta = self.theta_candidates[index_of_least_sigma_squared]

        print(f"[*] --- Best theta : {self.best_theta}")

    def validate(self):
        for valid_set in self.validation_sets:
            valid_set['predictions'] = self._predict(valid_set['data'], self.best_theta)

    def compute_validation_result(self):

        print("[*] - Computing Validation result")

        delta_mu_hats = []
        for valid_set in self.validation_sets:

            mu_hat = self.compute_mu_hat(
                train_set=self.train_set,
                test_set=valid_set,
                Y_hat_train=self.train_set["predictions"],
                Y_train=self.train_set["labels"],
                Y_hat_test=valid_set["predictions"]
                )

            # Compute delta mu hat (absolute value)
            delta_mu_hat = np.abs(valid_set["settings"]["ground_truth_mu"] - mu_hat)

            delta_mu_hats.append(delta_mu_hat)

            # print(f"[*] --- n_roi: {n_roi} --- nu_roi: {nu_roi} --- beta_roi: {beta_roi} --- gamma_roi: {gamma_roi}")
            print(f"[*] --- mu: {np.round(valid_set['settings']['ground_truth_mu'], 3)} --- mu_hat: {np.round(mu_hat, 3)} --- delta_mu_hat: {np.round(delta_mu_hat, 3)}")

        # Average delta mu_hat
        self.delta_mu_hat = np.mean(delta_mu_hats)
        print(f"[*] --- delta_mu_hat (avg): {np.round(self.delta_mu_hat, 2)}")

    def test(self):
        print("[*] - Testing")
        # Get predictions from trained model
        for test_set in self.test_sets:
            test_set["predictions"] = self._predict(test_set['data'], self.best_theta)
            test_set["decisions"] = self._decision_function(test_set['data'], self.best_theta)

    def compute_mu_hat(self, train_set, test_set, Y_hat_train, Y_train, Y_hat_test):

        # n_roi = len(Y_hat_test[Y_hat_test == 1])
        # n_roi is sum of weights of the predicted signals in train
        roi_indexes_test = np.argwhere(Y_hat_test == 1).flatten()
        n_roi = test_set['data'].iloc[roi_indexes_test]["Weight"].sum()

        # get region of interest from train

        # get indexes of points in train predicted as signal
        roi_indexes = np.argwhere(Y_hat_train == 1).flatten()
        # get all points from train with these indexes
        roi_points = Y_train[roi_indexes]
        # compute nu_roi
        # nu_roi = len(roi_points)
        # nu_roi is sum of weights of the groundtruth signals in train
        nu_roi = train_set['data'].iloc[roi_indexes]["Weight"].sum()

        # compute gamma_roi
        # indexes = np.argwhere(roi_points == 1)
        # get indexes of roi points where label = 1
        indexes = np.argwhere(roi_points == 1).flatten()

        # get signal class predictions
        # signal_predictions = roi_points[indexes]
        # gamma_roi = len(signal_predictions)
        gamma_roi = train_set['data'].iloc[indexes]["Weight"].sum()

        # compute beta_roi
        beta_roi = nu_roi - gamma_roi

        if gamma_roi == 0:
            gamma_roi = EPSILON

        # Compute mu_hat
        mu_hat = (n_roi - beta_roi)/gamma_roi

        print(f"n_roi: {n_roi} --- nu_roi: {nu_roi} --- gamma_roi: {gamma_roi} --- beta_roi: {beta_roi}")

        return mu_hat

    def compute_test_result(self):

        print("[*] - Computing Test result")

        mu_hats = []
        for test_set in self.test_sets:

            mu_hat = self.compute_mu_hat(
                train_set=self.train_set,
                test_set=test_set,
                Y_hat_train=self.train_set["predictions"],
                Y_train=self.train_set["labels"],
                Y_hat_test=test_set["predictions"]
                )
            mu_hats.append(mu_hat)
            print(f"[*] --- mu_hat: {np.round(mu_hat, 3)}")

        # Save mu_hat from test
        self.mu_hats = mu_hats
