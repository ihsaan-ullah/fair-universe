#---------------------------
# Imports
#---------------------------
import random
import numpy as np
from sklearn.utils import shuffle


#---------------------------
# Get augmented data
#---------------------------
def get_augmented_data(X_train, X_test):
        
        alpha = random.choice([0.1,0.6,0.9,1.6,2.1,3.4,4.6,5.9])

        train_mean = np.mean(X_train).values
        test_mean = np.mean(X_test).values

        # Esitmate z0
        z0 = train_mean - test_mean

        # transform z0 by alpha
        z0 = np.array(z0) * alpha

        X_train_augmented = X_train + z0

        return X_train_augmented