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
    2) fit : fit a classifer with train data
    3) predict: predict the output using decision function and threshold theta
    4) decision_function: outputs decisions/predict_proba of the classifier
    """

    def __init__(self):
        # Do not remove this attribute
        self.model_name = "Gaussian Naive Bayes"

        # Initalize classifier
        self.clf = GaussianNB()

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X, theta):

        predictions = np.zeros(X.shape[0])
        decisions = self.decision_function(X, theta)

        # class 1 -> if decision function  > theta
        # class 0 -> otherwise
        predictions = (decisions > theta).astype(int)
        return predictions

    def decision_function(self, X, theta):

        predicted_score = self.clf.predict_proba(X)
        # Transform with log
        epsilon = np.finfo(float).eps
        predicted_score = -np.log((1/(predicted_score+epsilon))-1)
        decisions = predicted_score[:, 1]

        return decisions - theta
