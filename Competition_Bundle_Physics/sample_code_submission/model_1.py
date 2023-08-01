import numpy as np
from sklearn.naive_bayes import GaussianNB


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
            data_gen_param: None,
            data_gen: None,
            test_data: None
    ):
        """
        Model class constructor

        Params:
            data_gen_params:
                data generator parameters class
                used to generate settings for data generator
                data_gen_params is preconfigured with some parameters
                to reconfigure again call `data_gen_params.reset_params()`

            data_gen:
                data generator itself
                takes generated settings from data_gen_params and generates data

            test_data:
                unlabelled test data

        Returns:
            None
        """

        # Set class variables from parameters
        self.data_gen_param = data_gen_param
        self.data_gen = data_gen
        self.test_set = {
            "data": test_data
        }

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

        # Train classifier
        self.train()

        # Validate using multiple validations sets
        # choose best theta from theta candidates
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
            "mu_hat": self.test_set['mu_hat'], 
            "delta_mu_hat": self.best_validation_set["delta_mu_hat"]
        }

    def _init_model(self):
        self.clf = GaussianNB()

    def _fit(self, X, y):
        self.clf.fit(X, y)

    def _predict(self, X, theta):
        predictions = np.zeros(X.shape[0])
        decisions = self._decision_function(X, theta)

        # class 1 -> if decision function  > theta
        # class 0 -> otherwise
        predictions = (decisions > theta).astype(int)
        return predictions

    def _decision_function(self, X, theta):

        predicted_score = self.clf.predict_proba(X)
        # Transform with log
        epsilon = np.finfo(float).eps
        predicted_score = -np.log((1/(predicted_score+epsilon))-1)
        decisions = predicted_score[:, 1]

        return decisions - theta

    def generate_train_set(self):
        print("[*] - Generating Train set")

        # Generate train settings without systematics
        data_gen_setting = self.data_gen_param.get_settings(use_systematics=False)

        # Generate train data
        data_gen = self.data_gen(settings_dict=data_gen_setting)
        data_gen.generate_data()
        self.train_set = data_gen.get_data()

    def generate_validation_sets(self):
        print("[*] - Generating Validation sets")

        self.validation_sets = []

        # Loop 10 times to generate 10 validation sets
        for i in range(0, 10):

            self.data_gen_param.reset_params()

            # Generate validation settings with systamatics
            data_gen_setting = self.data_gen_param.get_settings()

            # Generate validation data
            data_gen = self.data_gen(settings_dict=data_gen_setting)
            data_gen.generate_data()
            valid_set = data_gen.get_data()

            self.validation_sets.append(valid_set)

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

    def validate(self):

        if self.validation_sets is None:
            raise ValueError("[-] Validation sets are not generated! Call `generate_validation_sets` first.")

        print("[*] - Validate trained classifier")

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
                predictions = self._predict(valid_set['data'], theta)

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

    def _get_N_ROI(self, predictions):
        return len(predictions[predictions == 1])

    def compute_validation_result(self):

        print("[*] - Computing Validation result")

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

    def test(self):
        print("[*] - Testing")
        # Get predictions from trained model
        self.test_set['predictions'] = self._predict(self.test_set['data'], self.best_theta)

    def compute_test_result(self):

        print("[*] - Computing Test result")

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
