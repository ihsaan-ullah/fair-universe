import pickle
from os.path import isfile
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from copy import deepcopy
from sklearn.naive_bayes import GaussianNB


MODEL_CONSTANT = "Constant"
MODEL_NB = "NB"


PREPROCESS_TRANSLATION = "translation"


# ------------------------------
# Baseline Model
# ------------------------------
class Model:

    def __init__(self,
                 model_name=MODEL_NB,
                 X_train=None,
                 Y_train=None,
                 X_test=None,
                 preprocessing=False,
                 preprocessing_method=PREPROCESS_TRANSLATION,
                 theta=None,  # threshold for decision function
                 case=None
                 ):

        self.model_name = model_name
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test

        self.preprocessing = preprocessing
        self.preprocessing_method = preprocessing_method

        if case is None:
            self.case = case
        else:
            self.case = case - 1

        self.theta = theta

        self._set_model()

    def _set_model(self):

        if self.model_name == MODEL_CONSTANT:
            self.clf = None
        if self.model_name == MODEL_NB:
            self.clf = GaussianNB()

        self.is_trained = False

    def _preprocess_translation(self, X):

        train_mean = np.mean(self.X_train).values
        test_mean = np.mean(self.X_test).values

        translation = test_mean - train_mean
        return (X - translation)

    def fit(self, X=None, y=None):

        if self.model_name != MODEL_CONSTANT:

            if X is None:
                X = self.X_train
            if y is None:
                y = self.Y_train

            self.clf.fit(X, y)
            self.is_trained = True

    def predict(self, X=None, preprocess=True):

        if X is None:
            X = self.X_test

        if self.model_name == MODEL_CONSTANT:
            return np.zeros(X.shape[0])

        if self.preprocessing & preprocess:
            if self.preprocessing_method == PREPROCESS_TRANSLATION:
                X = self._preprocess_translation(X)

        # if decision function  > theta --> class 1
        # else --> class 0
        predictions = np.zeros(X.shape[0])
        decisions = self.decision_function(X)

        predictions = (decisions > self.theta).astype(int)
        return predictions

    def decision_function(self, X=None, preprocess=True):

        if X is None:
            X = self.X_test

        if self.model_name == MODEL_CONSTANT:
            return np.zeros(X.shape[0])

        if self.preprocessing and preprocess:
            if self.preprocessing_method == PREPROCESS_TRANSLATION:
                X = self._preprocess_translation(X)

        if self.model_name in [MODEL_NB]:
            predicted_score = self.clf.predict_proba(X)
            # Transform with log
            epsilon = np.finfo(float).eps
            predicted_score = -np.log((1/(predicted_score+epsilon))-1)
            decisions = predicted_score[:, 1]
        else:
            decisions = self.clf.decision_function(X)

        # decision function = decision function - theta
        # return decisions - self.theta
        return decisions

    def save(self, name):
        pickle.dump(self.clf, open(name + '.pickle', "wb"))

    def load(self, name):
        modelfile = name + '.pickle'
        if isfile(modelfile):
            with open(modelfile, 'rb') as f:
                self = pickle.load(f)
            print("Model reloaded from: " + modelfile)
        return self
