import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


EPSILON = np.finfo(float).eps


# ------------------------------
# Two layerd Neural Network
# ------------------------------
# class TwoLayerNN(nn.Module):

#     def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
#         super(TwoLayerNN, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size1)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_size1, hidden_size2)
#         self.relu2 = nn.ReLU()
#         self.fc3 = nn.Linear(hidden_size2, output_size)

#     def forward(self, x):
#         out = self.fc1(x)
#         out = self.relu1(out)
#         out = self.fc2(out)
#         out = self.relu2(out)
#         out = self.fc3(out)
#         out = torch.sigmoid(out)
#         return out


# ------------------------------
# Custom Dataset
# ------------------------------
# class CustomDataset(Dataset):
#     def __init__(self, data, labels=None):
#         self.data = data
#         self.labels = labels

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         sample = {'data': self.data[idx]}
#         if self.labels is not None:
#             sample['labels'] = self.labels[idx]
#         return sample


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
                name of the model, default: NN

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

        # Intialize class variables
        self.validation_sets = None
        self.theta_candidates = np.arange(0, 1, 0.1)
        self.best_theta = None

        # Hyper params
        self.num_epochs = 10
        self.batch_size = 32

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
        self._choose_theta()
        self._validate()
        self._compute_validation_result()

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

        self._test()
        self._compute_test_result()

        return {
            "mu_hats": self.mu_hats,
            "delta_mu_hat": self.delta_mu_hat
        }

    def _init_model(self):
        print("[*] - Intialize BDT")

        self.model = XGBClassifier(tree_method="hist",use_label_encoder=False,eval_metric='logloss')

    def _generate_validation_sets(self):
        print("[*] - Generating Validation sets")

        # Keep 70% of train set for training
        # Use the remaining 30% as validation set
        # Add systematics to validation set and create multiple validation sets

        # create a df for train test split
        train_df = self.train_set["data"]
        train_df["Label"] = self.train_set["labels"]
        train_df["Weight"] = self.train_set["weights"]  

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

    def _train(self):
        print("[*] - Train Neural Network")

        self._init_model()

        print("[*] --- Training Model")
        self._fit(self.train_set['data'], self.train_set['labels'], sample_weight = self.train_set['weights'])

        print("[*] --- Predicting Train set")
        self.train_set['predictions'] = self._predict(self.train_set['data'], 0.95)

    def _fit(self, X, y):
        self.model.fit(X, y.values) 

    def _calculate_Events(self, y, weights):
        events = weights[y == 1].sum()
        return events

    def _predict(self, X, theta):
        y_predict = self.models.predict_proba(X)[:,1]
        predictions = np.where(y_predict > theta, 1, 0) 
        return predictions

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

    def _choose_theta(self):

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

            print(nu_roi, gamma_roi, nu_roi/np.square(gamma_roi))

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

    def _validate(self):
        for valid_set in self.validation_sets:
            valid_set['predictions'] = self._predict(valid_set['data'], self.best_theta)

    def _compute_validation_result(self):

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
                gamma_roi = EPSILON

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

    def _test(self):
        print("[*] - Testing")
        # Get predictions from trained model
        for test_set in self.test_sets:
            test_set['predictions'] = self._predict(test_set['data'], self.best_theta)

    def _compute_test_result(self):

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
                gamma_roi = EPSILON

            # Compute mu_hat
            mu_hat = (n_roi - beta_roi)/gamma_roi

            mu_hats.append(mu_hat)
            print(f"[*] --- mu_hat: {np.round(mu_hat, 2)}")

        # Save mu_hat from test
        self.mu_hats = mu_hats
