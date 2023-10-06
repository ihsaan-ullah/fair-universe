import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score


class GaussianDiscriminativeAnalysisClassifier:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        self.n_features = X.shape[1]
        self.priors = np.zeros(self.n_classes)
        self.means = np.zeros((self.n_classes, self.n_features))
        self.stds = np.zeros((self.n_classes, self.n_features))
        self.covs = np.zeros(
            (self.n_classes, self.n_features, self.n_features))

        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.priors[i] = len(X_c) / len(X)
            self.means[i, :] = X_c.mean(axis=0)
            self.stds[i, :] = X_c.std(axis=0)
            self.covs[i, :, :] = np.cov(X_c.T)

    def _pdf(self, class_idx, X):
        mean = self.means[class_idx]
        cov = self.covs[class_idx]
        inv = np.linalg.inv(cov)
        det = np.linalg.det(cov)
        coefficient = 1 / np.sqrt((2 * np.pi) ** self.n_features * det)

        if (type(X) == pd.DataFrame):
            X = X.to_numpy()

        diff = X - mean.reshape(1, -1)
        exponent = -0.5 * np.sum(diff @ inv * diff, axis=1)
        pdfs = coefficient * np.exp(exponent)

        return pdfs

    def _softmax(self, X):
        exps = np.exp(X)
        return exps / np.sum(exps, axis=1).reshape(-1, 1)

    def predict_joint_log_proba(self, X):
        posteriors = []

        for i in range(self.n_classes):
            prior = np.log(self.priors[i])
            class_conditional = np.log(self._pdf(i, X))
            posterior = prior + class_conditional
            posteriors.append(posterior)

        posteriors = np.array(posteriors)
        return posteriors.T

    def predict_proba(self, X):
        return self._softmax(self.predict_joint_log_proba(X))

    def predict(self, X):
        y_pred = np.argmax(self.predict_joint_log_proba(X), axis=1)
        return y_pred

    def balanced_accuracy_score(self, X, y):
        y_pred = self.predict(X)
        return balanced_accuracy_score(y, y_pred)
