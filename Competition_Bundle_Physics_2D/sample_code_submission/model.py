import numpy as np
from copy import deepcopy
import pandas as pd
from sklearn.utils import shuffle

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from GDA import GaussianDiscriminativeAnalysisClassifier


MODELS = {
    "NB": GaussianNB,
    "Ridge R": RidgeClassifier,
    "LDA": LinearDiscriminantAnalysis,
    "GDA": GaussianDiscriminativeAnalysisClassifier
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
            data_gen=None,
            test_sets=[],
            SEED=None,
            model_name="NB",
            preprocessing=False,
            augmentation=False,
            use_systematics=True
    ):
        """
        Model class constructor

        Params:
            data_gen:
                data generator itself
                takes params and generates data

            test_sets:
                unlabelled test sets

        Returns:
            None
        """

        # Set class variables from parameters
        self.model_name = model_name
        self.preprocessing = preprocessing
        self.augmentation = augmentation
        self.data_gen = data_gen
        self.test_sets = []
        for test_set in test_sets:
            self.test_sets.append({"data": test_set})
        self.SEED = SEED
        self.use_systematics = use_systematics

        # Intialize class variables
        self.train_set = None
        self.validation_sets = None
        self.theta_candidates = np.arange(-10, 3)
        self.best_theta = None
        self.best_validation_set = None

    def fit(self):
        """
        Params:
            None

        Functionality:
            this function can be used to generate multiple datasets and train a model
            or anything else to do with the data generator

        Returns:
            None
        """

        # Generate Train set
        self.generate_train_set()

        # Generate Validation sets
        self.generate_validation_sets()

        # Set Means of Train and Test
        self.set_train_test_mean()

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
            this function can be used to generate multiple datasets and train a model
            or anything else to do with the data generator

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
            "delta_mu_hat": self.delta_mu_hat
        }

    def _init_model(self):
        model = MODELS[self.model_name]
        self.clf = model()

    def _fit(self, X, y):
        if self.augmentation:
            X, y = self._augment_data_translation()
        self.clf.fit(X, y)

    def _predict(self, X, theta):

        if self.preprocessing:
            # Preprocess
            X = self._preprocess_translation(X)

        predictions = np.zeros(X.shape[0])
        decisions = self._decision_function(X, theta)

        # class 1 -> if decision function  > theta
        # class 0 -> otherwise
        predictions = (decisions > theta).astype(int)
        return predictions

    def _decision_function(self, X, theta):

        decisions = None

        if self.model_name in ["Ridge R", "LDA"]:
            decisions = self.clf.decision_function(X)
        else:
            predicted_score = self.clf.predict_proba(X)
            # Transform with log
            epsilon = np.finfo(float).eps
            predicted_score = -np.log((1/(predicted_score+epsilon))-1)
            decisions = predicted_score[:, 1]

        return decisions - theta

    def _preprocess_translation(self, X):

        translation = self.test_mean - self.train_mean
        return (X - translation)

    def _augment_data_translation(self):

        random_state = 42
        size = 100000

        # Esitmate z0
        translation = self.test_mean - self.train_mean

        train_data_augmented, train_labels_augmented = [], []
        for i in range(0, 5):
            # randomly choose an alpha

            alphas = np.repeat(np.random.uniform(-3.0, 3.0, size=size).reshape(-1, 1), 2, axis=1)

            # transform z0 by alpha
            translation_ = translation * alphas

            np.random.RandomState(random_state)
            train_df = deepcopy(self.train_set['data'])
            train_df["labels"] = self.train_set['labels']

            df_sampled = train_df.sample(n=size, random_state=random_state, replace=True)
            data_sampled = df_sampled.drop("labels", axis=1)
            labels_sampled = df_sampled["labels"].values

            train_data_augmented.append(data_sampled + translation_)
            train_labels_augmented.append(labels_sampled)

        augmented_data = pd.concat(train_data_augmented)
        augmented_labels = np.concatenate(train_labels_augmented)

        augmented_data = shuffle(augmented_data, random_state=random_state)
        augmented_labels = shuffle(augmented_labels, random_state=random_state)

        return augmented_data, augmented_labels

    def generate_train_set(self):
        print("[*] - Generating Train set")

        # Data Gen params
        data_gen_param = {
            "pi": 0.1,
            "nu_1": 100000,
            "mu_range": [1.0, 1.0],
            "systematics": None
        }

        # Generate train data
        data_gen = self.data_gen(params=data_gen_param, SEED=self.SEED)
        data_gen.generate_data()
        self.train_set = data_gen.get_data()

    def generate_validation_sets(self):
        print("[*] - Generating Validation sets")

        self.validation_sets = []

        # Loop 10 times to generate 10 validation sets
        for i in range(0, 10):

            # Data Gen params
            systematics = None
            if self.use_systematics:
                systematics = [{
                    "name": "Translation",
                    "z_range": [-10, 10],
                    "z_angles": [45]
                }]
            data_gen_param = {
                "pi": 0.1,
                "nu_1": 100000,
                "mu_range": [0.9, 1.1],
                "systematics": systematics
            }

            # Generate validation data
            data_gen = self.data_gen(params=data_gen_param, SEED=None)
            data_gen.generate_data()
            valid_set = data_gen.get_data()

            self.validation_sets.append(valid_set)

    def set_train_test_mean(self):

        self.train_mean = np.mean(self.train_set['data']).values
        test_means = []
        for test_set in self.test_sets:
            test_means.append(np.mean(test_set['data']).values)
        self.test_mean = np.mean(test_means)

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

        if self.train_set is None:
            raise ValueError("[-] Train set is not generated! Call `generate_train_set` first")

        print("[*] - Train a classifier")

        print("[*] --- Loading Model")
        self._init_model()

        print("[*] --- Training Model")
        self._fit(self.train_set['data'], self.train_set['labels'])

        print("[*] --- Predicting Train set")
        self.train_set['predictions'] = self._predict(self.train_set['data'], 0)

    def choose_theta(self):

        if self.validation_sets is None:
            raise ValueError("[-] Validation sets are not generated! Call `generate_validation_sets` first.")

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
            roi_indexes = np.argwhere(Y_hat_valid == 1)
            roi_points = Y_valid[roi_indexes]
            # compute nu_roi
            nu_roi = len(roi_points)

            # compute gamma_roi
            indexes = np.argwhere(roi_points == 1)

            # get signal class predictions
            signal_predictions = roi_points[indexes]
            gamma_roi = len(signal_predictions)

            # compute beta_roi
            beta_roi = nu_roi - gamma_roi

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

            Y_hat_train = self.train_set["predictions"]
            Y_train = self.train_set["labels"]
            Y_hat_valid = valid_set["predictions"]

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

            if gamma_roi == 0:
                gamma_roi = 1

            # Compute mu_hat
            mu_hat = (n_roi - beta_roi)/gamma_roi

            # Compute delta mu hat (absolute value)
            delta_mu_hat = np.abs(valid_set["settings"]["ground_truth_mu"] - mu_hat)

            delta_mu_hats.append(delta_mu_hat)

            # print(f"[*] --- n_roi: {n_roi} --- nu_roi: {nu_roi} --- beta_roi: {beta_roi} --- gamma_roi: {gamma_roi}")
            print(f"[*] --- mu: {np.round(valid_set['settings']['ground_truth_mu'], 2)} --- mu_hat: {np.round(mu_hat, 2)} --- delta_mu_hat: {np.round(delta_mu_hat, 2)}")

        # Average delta mu_hat
        self.delta_mu_hat = np.mean(delta_mu_hats)
        print(f"[*] --- delta_mu_hat (avg): {np.round(self.delta_mu_hat, 2)}")

    def test(self):
        print("[*] - Testing")
        # Get predictions from trained model
        for test_set in self.test_sets:
            test_set['predictions'] = self._predict(test_set['data'], self.best_theta)

    def compute_test_result(self):

        print("[*] - Computing Test result")

        mu_hats = []
        for test_set in self.test_sets:

            Y_hat_train = self.train_set["predictions"]
            Y_train = self.train_set["labels"]
            Y_hat_test = test_set["predictions"]

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

            if gamma_roi == 0:
                gamma_roi = 1

            # Compute mu_hat
            mu_hat = (n_roi - beta_roi)/gamma_roi

            mu_hats.append(mu_hat)
            print(f"[*] --- mu_hat: {np.round(mu_hat, 2)}")

        # Save mu_hat from test
        self.mu_hats = mu_hats
