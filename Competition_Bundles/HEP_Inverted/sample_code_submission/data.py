# ------------------------------------------
# Imports
# ------------------------------------------
import os
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from copy import deepcopy


# ------------------------------------------
# Data Class
# ------------------------------------------
class Data:
    def __init__(self):
        self.train_set = None
        self.test_sets = []
        self.validation_sets = []
        self.data_dir = os.path.dirname(os.path.abspath(__file__))

        self._load_train_set()
        self._split_train_valid()
        self._load_test_sets()

    def _load_train_set(self):
        print("[*] - Loading Train data")

        train_data_file = os.path.join(self.data_dir, 'train', 'data', 'data.csv')
        train_labels_file = os.path.join(self.data_dir, 'train', 'labels', "data.labels")
        train_settings_file = os.path.join(self.data_dir, 'train', 'settings', "data.json")

        # read train data

        train_data = pd.read_csv(train_data_file)

        # read trian labels
        with open(train_labels_file, "r") as f:
            train_labels = np.array(f.read().splitlines(), dtype=float)

        # read train settings
        with open(train_settings_file) as f:
            train_settings = json.load(f)

        # reweight train
        weights_train = deepcopy(train_data["Weight"])
        class_weights_train = (weights_train[train_labels == 0].sum(), weights_train[train_labels == 1].sum())

        for i in range(len(class_weights_train)):  # loop on B then S target
            # training dataset: equalize number of background and signal
            weights_train[train_labels == i] *= max(class_weights_train) / class_weights_train[i]

        # reassign weight to data
        train_data["New_Weight"] = weights_train

        self.train_set = {
            "data": train_data,
            "labels": train_labels,
            "settings": train_settings
        }

    def _split_train_valid(self):
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
            self.validation_sets.append({
                "data": valid.drop('Label', axis=1),
                "labels": valid["Label"].values,
                "settings": self.train_set["settings"]
            })

    def _load_test_sets(self):
        print("[*] - Loading Test data")
        self.test_sets = []
        for i in range(0, 10):

            test_data_file = os.path.join(self.data_dir, 'test', 'data', 'data_'+str(i)+'.csv')
            test_labels_file = os.path.join(self.data_dir, 'test', 'labels', 'data_'+str(i)+'.labels')
            test_settings_file = os.path.join(self.data_dir, 'test', 'settings', 'data_'+str(i)+'.json')

            # read test data
            test_data = pd.read_csv(test_data_file)

            # read test labels
            with open(test_labels_file, "r") as f:
                test_labels = np.array(f.read().splitlines(), dtype=float)

            # read test settings
            with open(test_settings_file) as f:
                test_settings = json.load(f)

            self.test_sets.append({
                "data": test_data,
                "labels": test_labels,
                "settings": test_settings
            })

    def get_train_set(self):
        if self.train_set is None:
            raise ValueError("[-] Train set is not loaded!")
        return self.train_set

    def get_validation_sets(self):
        if not self.validation_sets:
            raise ValueError("[-] Validation sets are not generated!")
        return self.validation_sets

    def get_test_sets(self):
        if not self.test_sets:
            raise ValueError("[-] Test sets are not loaded!")
        return self.test_sets
