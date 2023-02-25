import pickle
import random
import numpy as np  
import pandas as pd
from os.path import isfile
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import shuffle



#------------------------------
# Naive Bayes Baseline Model with Data Augmentation
#------------------------------
class Model:

    def __init__(self):
        self.model_name = "Augmented"
        self.clf = GaussianNB()
        self.is_trained=False
        self.alpha = random.choice([0.1,0.6,0.9,1.6,2.1,3.4,4.6,5.9])

    def _get_augmented_data(self, X_train, X_test):


        train_mean = np.mean(X_train).values
        test_mean = np.mean(X_test).values

        # Esitmate z0
        z0 = train_mean - test_mean

        # transform z0 by alpha
        z0 = np.array(z0) * self.alpha

        X_train_augmented = X_train + z0

        return X_train_augmented

    def fit(self, X_train, X_test, Y_train):

        augmented_data = self._get_augmented_data(X_train, X_test)


        # combine real and augemnted data
        X = pd.concat([X_train , augmented_data])

        Y = np.append(Y_train , Y_train)

        # shuffle data
        X, Y = shuffle(X, Y, random_state=0)

        # fit data
        self.clf.fit(X, Y)

        self.is_trained=True

    def predict(self, X):
        return self.clf.predict(X)

    def predict_score(self, X):
        return self.clf.predict_proba(X)[:, 1]

    def save(self, path="./"):
        pickle.dump(self.clf, open(path + '_model.pickle', "wb"))

    def load(self, path="./"):
        modelfile = path + '_model.pickle'
        if isfile(modelfile):
            with open(modelfile, 'rb') as f:
                self = pickle.load(f)
            print("Model reloaded from: " + modelfile)
        return self
