import pickle
from os.path import isfile
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from copy import deepcopy
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import RidgeClassifier



MODEL_CONSTANT = "Constant"
MODEL_NB = "NB"
MODEL_LDA = "LDA"
MODEL_RR = "RR"


#------------------------------
# Baseline Model
#------------------------------
class Model:

    def __init__(self, 
                 model_name=MODEL_NB, 
                 X_train=None, 
                 Y_train=None, 
                 X_test=None, 
                 preprocessing=False, 
                 data_aumentation=False
        ):

        self.model_name = model_name
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test

        self.preprocessing = preprocessing
        self.data_aumentation = data_aumentation

        self._set_model()

    def _set_model(self):

        if self.model_name == MODEL_CONSTANT:
            self.clf = None 
        if self.model_name == MODEL_NB:
            self.clf = GaussianNB()
        if self.model_name == MODEL_LDA:
            self.clf = LinearDiscriminantAnalysis()
        if self.model_name == MODEL_RR:
            self.clf = RidgeClassifier()

        self.is_trained=False

    def _preprocess(self):

        train_mean = np.mean(self.X_train).values
        test_mean = np.mean(self.X_test).values

        X_test_preprocessed = self.X_test + train_mean - test_mean

        return X_test_preprocessed

    def _augment_data(self):

        random_state = 42
        size = 1000

        # Mean of Train and Test
        train_mean = np.mean(self.X_train).values
        test_mean = np.mean(self.X_test).values

        # Esitmate z0
        z0 = train_mean - test_mean


        train_data_augmented, train_labels_augmented = [], []
        for i in range(0, 5):
            # randomly choose an alpha

            alphas = np.repeat(np.random.uniform(-3.0, 3.0, size=size).reshape(-1,1), 2, axis=1 )

            # transform z0 by alpha
            z0 = z0 * alphas

            np.random.RandomState(random_state)
            train_df = deepcopy(self.X_train)
            train_df["labels"] = self.Y_train

            df_sampled = train_df.sample(n=size, random_state=random_state, replace=True)
            data_sampled = df_sampled.drop("labels", axis=1)
            labels_sampled = df_sampled["labels"].values
    

            train_data_augmented.append(data_sampled + z0)
            train_labels_augmented.append(labels_sampled)

 

        augmented_data = pd.concat(train_data_augmented)
        augmented_labels = np.concatenate(train_labels_augmented)

        augmented_data = shuffle(augmented_data, random_state=random_state)
        augmented_labels =shuffle(augmented_labels, random_state=random_state)


        return augmented_data, augmented_labels
        
    def fit(self, X=None, y=None):

        if X is None:
            X = self.X_train
        if y is None:
            y = self.Y_train

        if self.data_aumentation:
            X, y = self._augment_data()
  
        self.clf.fit(X, y)
        self.is_trained=True

    def predict(self, X=None):
        if X is None:
            X = self.X_test

        if self.preprocessing:
            X = self._preprocess()

        return self.clf.predict(X)

    
    def decision_function(self, X=None):
        if X is None:
            X = self.X_test
        
        if self.preprocessing:
            X = self._preprocess()

        if self.model_name == MODEL_NB:
            return self.clf.predict_proba(X)[:, 1]
        else:
            return self.clf.decision_function(X)
        
    def save(self, name):
        pickle.dump(self.clf, open(name + '.pickle', "wb"))

    
    def load(self, name):
        modelfile = name + '.pickle'
        if isfile(modelfile):
            with open(modelfile, 'rb') as f:
                self = pickle.load(f)
            print("Model reloaded from: " + modelfile)
        return self
