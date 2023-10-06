import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

EPSILON = np.finfo(float).eps

MODELS = {
    "NB": GaussianNB,
    "Ridge R": RidgeClassifier,
    "LDA": LinearDiscriminantAnalysis,
    "DTC": DecisionTreeClassifier,
    "SVM": SVC,
    "XG": GradientBoostingClassifier
}


# ------------------------------
# Baseline Model
# ------------------------------
class Model:
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
            model_name="XG",
            use_systematics=False
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
                name of the model, default: NB

        Returns:
            None
        """

        # Set class variables from parameters
        self.model_name = model_name
        self.train_set = train_set
        self.test_sets = []
        for test_set in test_sets:
            self.test_sets.append({"data": test_set})
        self.systematics = systematics
        self.use_systematics = use_systematics

        # Intialize class variables
        self.validation_sets = None
        self.theta_candidates = np.arange(-10, 10)
        self.best_theta = None

    def fit(self):
        """
        Params:
            None

        Functionality:
            this function can be used to train a model using the train set

        Returns:
            None
        """

        # Generate Validation sets
        self.generate_validation_sets()

        # Train classifier
        self.train()

        # Choose best theta
        self.choose_theta()

        # Validate using multiple validations sets
        self.validate()

        # Compute mu and delta mu from validation set
        self.compute_validation_result()

    def predict(self):
        """
        Params:
            None

        Functionality:
           to predict using the test sets

        Returns:
            dict with keys
                - mu_hat
                - delta_mu_hat
        """

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
        if self.model_name == "SVM":
            self.clf = SVC(kernel='linear')
        elif self.model_name == "XG":
            self.clf = GradientBoostingClassifier()
        else:
            model = MODELS[self.model_name]
            self.clf = model()

    def _fit(self, X, y):
        if self.model_name == "XG":
            weights = X["New_Weight"]
            X = X.drop("Weight", axis=1)
            X = X.drop("New_Weight", axis=1)
            self.clf.fit(X, y, sample_weight=weights)
        else:
            self.clf.fit(X, y)

    def _predict(self, X, theta):
        if self.model_name == "XG":
            X = X.drop("Weight", axis=1)
            if "New_Weight" in X.columns:
                X = X.drop("New_Weight", axis=1)
        predictions = np.zeros(X.shape[0])
        decisions = self._decision_function(X, theta)

        # class 1 -> if decision function  > theta
        # class 0 -> otherwise
        predictions = (decisions > theta).astype(int)
        return predictions

    def _decision_function(self, X, theta):

        if self.model_name == "XG":
            if "Weight" in X.columns:
                X = X.drop("Weight", axis=1)

        # decision funciton: output between -inf and +inf
        # 0 is the neutral point between the 2 classes which is on the decision boundary

        decisions = None
        if self.model_name in ["Ridge R", "LDA", "SVM", "XG"]:
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

    def generate_validation_sets(self):
        print("[*] - Generating Validation sets")

        # Keep 70% of train set for training
        # Use the remaining 30% as validation set
        # Add systematics to validation set and create multiple validation sets

        # create a df for train test split
        train_df = self.train_set["data"]
        train_df["Label"] = self.train_set["labels"]

        # train: 70%
        # valid: 30%
        train, valid = train_test_split(train_df, test_size=0.3)

        self.train_set = {
            "data": train.drop('Label', axis=1),
            "labels": train["Label"].values,
            "settings": self.train_set["settings"]
        }

        self.validation_sets = []
        # Loop 10 times to generate 10 validation sets
        for i in range(0, 10):
            if self.use_systematics:
                tes = round(np.random.uniform(0.9, 1.10), 2)
                # apply systematics
                valid_with_systematics = self.systematics(
                    data=valid,
                    tes=tes
                ).data
                self.validation_sets.append({
                    "data": valid_with_systematics.drop('Label', axis=1),
                    "labels": valid["Label"].values,
                    "settings": self.train_set["settings"]
                })
            else:
                self.validation_sets.append({
                    "data": valid.drop('Label', axis=1),
                    "labels": valid["Label"].values,
                    "settings": self.train_set["settings"]
                })


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

        print("[*] - Train a classifier")

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

        # print(f"n_roi: {n_roi} --- nu_roi: {nu_roi} --- gamma_roi: {gamma_roi}")

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
