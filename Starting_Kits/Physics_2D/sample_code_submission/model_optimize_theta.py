import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB


# ------------------------------
# Model for Optimizing Theta
# ------------------------------
class Model:

    def __init__(self):

        self.clf = GaussianNB()
        self.is_trained = False

    def fit(self, X=None, y=None):

        self.clf.fit(X, y)
        self.is_trained = True

    def predict(self, X=None, theta=None):

        # if decision function  > theta --> class 1
        # else --> class 0
        predictions = np.zeros(X.shape[0])
        decisions = self.decision_function(X)

        predictions = (decisions > theta).astype(int)
        return predictions

    def decision_function(self, X=None):

        predicted_score = self.clf.predict_proba(X)
        # Transform with log
        epsilon = np.finfo(float).eps
        predicted_score = -np.log((1/(predicted_score+epsilon))-1)
        decisions = predicted_score[:, 1]

        # decision function = decision function - theta
        # return decisions - self.theta
        return decisions
