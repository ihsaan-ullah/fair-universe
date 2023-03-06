import pickle
import numpy as np  
from os.path import isfile



#------------------------------
# Dubm Baseline Model
#------------------------------
class Model:

    def __init__(self):
        self.model_name = "Dumb"

    def fit(self, X, y):
        pass

    def predict(self, X):
        return np.zeros(X.shape[0])

    def predict_score(self, X):
        return np.zeros(X.shape[0])
    
    def save(self, path="./"):
        pickle.dump(self.clf, open(path + '_model.pickle', "wb"))

    
    def load(self, path="./"):
        modelfile = path + '_model.pickle'
        if isfile(modelfile):
            with open(modelfile, 'rb') as f:
                self = pickle.load(f)
            print("Model reloaded from: " + modelfile)
        return self