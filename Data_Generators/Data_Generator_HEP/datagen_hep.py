from systematics import Systematics
from bootstrap import bootstrap_data

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import warnings
warnings.filterwarnings("ignore")


root_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(root_dir)
write_dir = os.path.join(parent_dir, 'Datagenerator')




# Load the CSV file
def dataGenerator(verbose=0):

    print('root - dir', root_dir)
    # Put reference_data.csv in the same dir as the datagen_hep.py
    csv_file_path = os.path.join(root_dir, 'FU_challenge_2023_FullData.csv')
    df = pd.read_csv(csv_file_path)

    # Remove the "label" and "weights" columns from the data


    flag = df.pop('Process_flag')
    label = df.pop('Label')
    weights = df.pop('Weight')
    eventid = df.pop('EventId')
    df.pop('PRI_lep_charge')
    df.pop('PRI_had_charge')    
    df.pop('PRI_jet_leading_charge')    
    df.pop('PRI_jet_subleading_charge')

    # Print the features of the data
    features = df.columns.tolist()
    if verbose > 1:
        print ("[*] --- Features of the data")
        for feature in features:
            print(feature)

    df = df.round(3)

    if verbose > 0:
        print (f"[*] --- sum of weights : {np.sum(weights)}")
        print (f"[*] --- sum of signal : {np.sum(weights[label==1])}")
        print (f"[*] --- sum of background : {np.sum(weights[label==0])}")
    
    # Calculate the sum of weights of signal and background for the whole data
    total_signal_weight = np.sum(weights[label==1])
    total_background_weight = np.sum(weights[label==0])

    # Split the data into training and testing sets
    train_df, test_df, train_label, test_label, train_weights, test_weights = train_test_split(df, label, weights, test_size=0.2)

    # Create directories to store the label and weight files
    train_label_path =  os.path.join(write_dir, 'input_data','train','labels')
    if not os.path.exists(train_label_path):
        os.makedirs(train_label_path)

    train_weights_path =  os.path.join(write_dir, 'input_data','train','weights')
    if not os.path.exists(train_weights_path):
        os.makedirs(train_weights_path)

    train_data_path =  os.path.join(write_dir, 'input_data','train','data')
    if not os.path.exists(train_data_path):
        os.makedirs(train_data_path)

    train_settings_path =  os.path.join(write_dir, 'input_data','train','settings')
    if not os.path.exists(train_settings_path):
        os.makedirs(train_settings_path)


    subset_signal_weight = np.sum(train_weights[train_label==1])
    subset_background_weight = np.sum(train_weights[train_label==0])


    # Adjust the weights of the training set to match the weights of the whole data

    train_weights[train_label==1] *= total_signal_weight / subset_signal_weight
    train_weights[train_label==0] *= total_background_weight / subset_background_weight

    if verbose > 1:
        print (f"[*] --- Signal in Training set " , np.sum(train_weights[train_label==1]))
        print (f"[*] --- Background in Training set" , np.sum(train_weights[train_label==0]))

    import json

    train_settings = {"tes": 1.0, "ground_truth_mu": 1.0}

    # Specify the file path
    Settings_file_path = os.path.join(train_settings_path, 'data.json')

    # Save the settings to a JSON file
    with open(Settings_file_path, 'w') as json_file:
        json.dump(train_settings, json_file, indent=4)

    # Save the training set as a CSV file
    train_df = train_df.round(3)
    print (f"[*] --- Signal in Training set " , np.sum(train_weights[train_label==1]))
    print (f"[*] --- Background in Training set" , np.sum(train_weights[train_label==0]))
    train_data_path = os.path.join(train_data_path, 'data.csv')
    train_df.to_csv(train_data_path, index=False)

    # Save the label and weight files for the training set
    train_label_path =  os.path.join(train_label_path, 'data.labels')
    train_label.to_csv(train_label_path, index=False, header=False)

    train_weights_path =  os.path.join(train_weights_path, 'data.weights')
    train_weights.to_csv(train_weights_path, index=False, header=False)

    subset_signal_weight_test = np.sum(test_weights[test_label==1])
    subset_background_weight_test = np.sum(test_weights[test_label==0])

    test_weights[test_label==1] *= total_signal_weight / subset_signal_weight_test
    test_weights[test_label==0] *= total_background_weight / subset_background_weight_test
    
    # Create directories to store the label and weight files
    reference_settings_path =  os.path.join(write_dir, 'reference_data','settings')
    if not os.path.exists(reference_settings_path):
        os.makedirs(reference_settings_path)

    test_weights_path =  os.path.join(write_dir, 'input_data','test','weights')
    if not os.path.exists(test_weights_path):
        os.makedirs(test_weights_path)

    test_data_path =  os.path.join(write_dir, 'input_data','test','data')
    if not os.path.exists(test_data_path):
        os.makedirs(test_data_path)

    test_settings_path =  os.path.join(write_dir, 'input_data','test','settings')
    if not os.path.exists(test_settings_path):
        os.makedirs(test_settings_path)
        


    print (f"[*] --- Signal in  test set " , np.sum(test_weights[test_label==1]))
    print (f"[*] --- Background in  test set " , np.sum(test_weights[test_label==0]))
    print (f"[*] --- Total Events in  test set " , np.sum(test_weights)) 

    print (f"[*] --- avegare weight of signal in test set : ", np.mean(test_weights[test_label==1]))
    print (f"[*] --- avegare weight of background in test set : ", np.mean(test_weights[test_label==0]))



    # Save the current subset as a CSV file
    data_file_path = os.path.join(test_data_path, f'data.csv')
    weights_file_path = os.path.join(test_weights_path, f'data.weights')

    # Writing data to files
    test_df.to_csv(data_file_path, index=False)
    test_weights.to_csv(weights_file_path, index=False, header=False)


    mu = np.random.uniform(0, 3, 10)
    mu = np.round(mu, 3)
    mu_list = mu.tolist()
    print (f"[*] --- mu in test set : ", mu_list)

    test_settings = {"ground_truth_mu": mu_list}
    Settings_file_path = os.path.join(reference_settings_path, 'data.json')
    with open(Settings_file_path, 'w') as json_file:
        json.dump(test_settings, json_file, indent=4)

    Settings_file_path = os.path.join(test_settings_path, 'data.json')
    with open(Settings_file_path, 'w') as json_file:
        json.dump(test_settings, json_file, indent=4)

def dataSimulator(n=1000, verbose=0, tes=1.0, mu=1.0):
    '''
    Simulates data for the HEP competition using reference data and systematics.

    Args:
        n (int): Number of samples to simulate. Default is 1000.
        verbose (int): Verbosity level. Default is 0.
        tes (float): TES (Trigger Efficiency Scale Factor) value. Default is 1.0.
        mu (float): Mu (Signal Strength) value. Default is 1.0.

    Returns:
        pandas.DataFrame: Simulated data with systematics and weights.
    '''

    # Get the directory of the current script (datagen_temp.py)
    module_dir = os.path.dirname(os.path.realpath(__file__))
    
    # Construct the absolute path to reference_data.csv
    csv_file_path = os.path.join(module_dir, 'reference_data.csv')
    df = pd.read_csv(csv_file_path)

    # Sample n rows from the reference data
    data = df.sample(n=n, replace=True)

    # Apply systematics to the sampled data
    data_syst = Systematics(
        data=data,
        verbose=verbose,
        tes=tes
    ).data

    # Apply weight scaling factor mu to the data
    data_syst['Weight'] *= mu

    return data_syst



    


if __name__ == "__main__":
    dataGenerator(verbose=2) 

