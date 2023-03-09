#---------------------------
# Imports
#---------------------------
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from copy import deepcopy


#---------------------------
# Get augmented data
#---------------------------
def get_augmented_data(train_set, test_set):

        
        random_state = 42
        train_mean = np.mean(train_set["data"]).values
        test_mean = np.mean(test_set["data"]).values

        size = 1000

        # Esitmate z0
        z0 = train_mean - test_mean

        train_data_augmented, train_labels_augmented = [], []
        for i in range(0, 5):
                # randomly choose an alpha

                alphas = np.repeat(np.random.uniform(-3.0, 3.0, size=size).reshape(-1,1), 2, axis=1 )

                # transform z0 by alpha
                z0 = z0 * alphas

                np.random.RandomState(random_state)
                train_df = deepcopy(train_set["data"])
                train_df["labels"] = train_set["labels"]

                df_sampled = train_df.sample(n=size, random_state=random_state, replace=True)
                data_sampled = df_sampled.drop("labels", axis=1)
                labels_sampled = df_sampled["labels"].values
                # data_sampled = train_set["data"].sample(n=size, random_state=random_state, replace=True)
                # labels_sampled = np.random.choice(train_set["labels"], size)

                train_data_augmented.append(data_sampled + z0)
                train_labels_augmented.append(labels_sampled)

 

        augmented_data = pd.concat(train_data_augmented)
        augmented_labels = np.concatenate(train_labels_augmented)

        augmented_data = shuffle(augmented_data, random_state=random_state)
        augmented_labels =shuffle(augmented_labels, random_state=random_state)

        return {
                "data" : augmented_data,
                "labels" : augmented_labels
        }