import numpy as np
import pandas as pd
from numpy.random import RandomState
import os
import json

root_dir = "./"
input_dir = root_dir + "input_data"

def bootstrap(weights, seed=42):
    
    prng = RandomState(seed)
    n_obs = len(weights)
    new_weights = prng.poisson(lam=weights)
    return new_weights

def bootstrap_data(data,weights,label, n = 1000,seed=42):

    total_signal_weight = np.sum(weights[label==1])
    total_background_weight = np.sum(weights[label==0])

    data_BS = data.copy()
    data_BS['weights'] = weights
    data_BS['label'] = label

    del data

    data = data_BS.sample(n=n, replace=False, weights=weights, random_state=seed)

    data_bootstrap = data

    subset_signal_weight = np.sum(data_bootstrap['weights'][data_bootstrap['label']==1])
    subset_background_weight = np.sum(data_bootstrap['weights'][data_bootstrap['label']==0])

    data_bootstrap['weights'][data_bootstrap['label']==1] *= total_signal_weight / subset_signal_weight
    data_bootstrap['weights'][data_bootstrap['label']==0] *= total_background_weight / subset_background_weight

    prng = RandomState(seed)

    data_bootstrap['weights'] = prng.poisson(lam=data_bootstrap['weights'])

    weights_bootstrap = data_bootstrap.pop('weights')
    label_bootstrap = data_bootstrap.pop('label')

    return {'data':data_bootstrap, 'weights':weights_bootstrap, 'label':label_bootstrap}


if __name__ == '__main__':
    print("[*] Loading Train data")

    train_data_file = os.path.join(input_dir, 'train', 'data', 'data.csv')
    train_labels_file = os.path.join(input_dir, 'train', 'labels', "data.labels")
    train_settings_file = os.path.join(input_dir, 'train', 'settings', "data.json")
    train_weights_file = os.path.join(input_dir, 'train', 'weights', "data.weights")

    # read train data
    train_data = pd.read_csv(train_data_file)

    # read trian labels
    with open(train_labels_file, "r") as f:
        train_labels = np.array(f.read().splitlines(), dtype=float)

    # read train settings
    with open(train_settings_file) as f:
        train_settings = json.load(f)
    
    # read train weights
    with open(train_weights_file) as f:
        train_weights = np.array(f.read().splitlines(), dtype=float)


    train_set = {
        "data": train_data,
        "labels": train_labels,
        "settings": train_settings,
        "weights": train_weights
    }

    bootstrap_datasets = bootstrap(train_data, train_weights, n=1000, seed=42)
    total_weights_distribution = []
    for data in bootstrap_datasets:
        total_weights = data['Weight'].sum()
        total_weights_distribution.append(total_weights)

    total_weights_distribution_array = np.array(total_weights_distribution)
    np.histogram(total_weights_distribution_array, bins=100)






