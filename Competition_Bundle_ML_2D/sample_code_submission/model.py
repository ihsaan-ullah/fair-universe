import pickle
from os.path import isfile
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from copy import deepcopy
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
import tensorflow as tf



MODEL_CONSTANT = "Constant"
MODEL_NB = "NB"
MODEL_LDA = "LDA"
MODEL_RR = "RR"
MODEL_SVM = "SVM"
MODEL_NN = "NN"


PREPROCESS_TRANSLATION = "translation"
PREPROCESS_SCALING = "scaling"

AUGMENTATION_TRANSLATION = "translation"
AUGMENTATION_TRANSLATION_SCALING = "translation-scaling"

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
                 preprocessing_method = PREPROCESS_TRANSLATION,
                 data_augmentation=False,
                 data_augmentation_type=AUGMENTATION_TRANSLATION
        ):

        self.model_name = model_name
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test

        self.preprocessing = preprocessing
        self.preprocessing_method = preprocessing_method
        self.data_augmentation = data_augmentation
        self.data_augmentation_type = data_augmentation_type

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
        if self.model_name == MODEL_SVM:
            self.clf = SVC()
        if self.model_name == MODEL_NN:
            self.clf = tf.keras.models.Sequential([
                tf.keras.layers.Dense(100, activation='relu'),
                tf.keras.layers.Dense(100, activation='relu'),
                tf.keras.layers.Dense(100, activation='relu'),
                tf.keras.layers.Dense(1, activation='linear'),
            ])
            self.clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        self.is_trained=False

    def _preprocess_translation(self):

        train_mean = np.mean(self.X_train).values
        test_mean = np.mean(self.X_test).values

        translation = test_mean- train_mean

        X_test_preprocessed = self.X_test - translation

        return X_test_preprocessed
    
    def _preprocess_scaling(self):

        train_mean = np.mean(self.X_train).values
        test_mean = np.mean(self.X_test).values

        train_std = np.std(self.X_train).values
        test_std = np.std(self.X_test).values


        translation = test_mean- train_mean
        scaling = test_std/train_std


        X_test_preprocessed = (self.X_test - translation)/scaling

        return X_test_preprocessed

    def _augment_data_translation(self):

        random_state = 42
        size = 1000


        # Mean of Train and Test
        train_mean = np.mean(self.X_train, axis=0).values
        test_mean = np.mean(self.X_test, axis=0).values

        # Esitmate z0
        translation = test_mean - train_mean



        train_data_augmented, train_labels_augmented = [], []
        for i in range(0, 5):
            # randomly choose an alpha

            alphas = np.repeat(np.random.uniform(-3.0, 3.0, size=size).reshape(-1,1), 2, axis=1 )

            # transform z0 by alpha
            translation_ = translation * alphas

            np.random.RandomState(random_state)
            train_df = deepcopy(self.X_train)
            train_df["labels"] = self.Y_train

            df_sampled = train_df.sample(n=size, random_state=random_state, replace=True)
            data_sampled = df_sampled.drop("labels", axis=1)
            labels_sampled = df_sampled["labels"].values
    

            train_data_augmented.append(data_sampled + translation_)
            train_labels_augmented.append(labels_sampled)

 

        augmented_data = pd.concat(train_data_augmented)
        augmented_labels = np.concatenate(train_labels_augmented)

        augmented_data = shuffle(augmented_data, random_state=random_state)
        augmented_labels =shuffle(augmented_labels, random_state=random_state)


        return augmented_data, augmented_labels
    
    def _augment_data_scaling(self):

        random_state = 42
        size = 1000

        # Mean of Train and Test
        train_mean = np.mean(self.X_train, axis=0).values
        test_mean = np.mean(self.X_test, axis=0).values

        train_std = np.std(self.X_train, axis=0).values
        test_std = np.std(self.X_test, axis=0).values

        # Esitmate z0
        translation = test_mean- train_mean
        scaling = test_std/train_std


        train_data_augmented, train_labels_augmented = [], []
        for i in range(0, 5):
            
            # uniformly choose alpha between -3 and 3
            alphas = np.repeat(np.random.uniform(-3.0, 3.0, size=size).reshape(-1,1), 2, axis=1 )

            # uniformly choose beta between 1 and 1.5
            betas = np.repeat(np.random.uniform(1.0, 1.5, size=size).reshape(-1,1), 2, axis=1 )

            # translation
            translation_ = translation * alphas
            # sclaing
            scaling_ = scaling * betas

            np.random.RandomState(random_state)
            train_df = deepcopy(self.X_train)
            train_df["labels"] = self.Y_train

            df_sampled = train_df.sample(n=size, random_state=random_state, replace=True)
            data_sampled = df_sampled.drop("labels", axis=1)
            labels_sampled = df_sampled["labels"].values

            transformed_train_data = (data_sampled + translation_)*scaling_
    

            train_data_augmented.append(transformed_train_data)
            train_labels_augmented.append(labels_sampled)

 

        augmented_data = pd.concat(train_data_augmented)
        augmented_labels = np.concatenate(train_labels_augmented)

        augmented_data = shuffle(augmented_data, random_state=random_state)
        augmented_labels =shuffle(augmented_labels, random_state=random_state)


        return augmented_data, augmented_labels
        
    def fit(self, X=None, y=None):

        if self.model_name != MODEL_CONSTANT:
            
            if X is None:
                X = self.X_train
            if y is None:
                y = self.Y_train

            if self.data_augmentation:
                if self.data_augmentation_type == AUGMENTATION_TRANSLATION:
                    X, y = self._augment_data_translation()
                else:
                    X, y = self._augment_data_scaling()
    
            self.clf.fit(X, y)
            self.is_trained=True

    def predict(self, X=None):
        if X is None:
            X = self.X_test

        if self.model_name == MODEL_CONSTANT:
            return np.zeros(X.shape[0])

        if self.preprocessing:
            if self.preprocessing_method == PREPROCESS_TRANSLATION:
                X = self._preprocess_translation()
            else:
                X = self._preprocess_scaling()

        return self.clf.predict(X)

    def decision_function(self, X=None):
        
        if X is None:
            X = self.X_test

        if self.model_name == MODEL_CONSTANT:
            return np.zeros(X.shape[0])
        
        if self.preprocessing:
            if self.preprocessing_method == PREPROCESS_TRANSLATION:
                X = self._preprocess_translation()
            else:
                X = self._preprocess_scaling()

        if self.model_name == MODEL_NB:
            return self.clf.predict_proba(X)[:, 1]
        elif self.model_name == MODEL_NN:
            return self.clf.predict(X)
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
