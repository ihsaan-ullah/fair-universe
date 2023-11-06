import pandas as pd
from sys import path
import numpy as np
from sklearn.model_selection import train_test_split
import os
import warnings
warnings.filterwarnings("ignore")

module_dir = os.path.dirname(os.path.realpath(__file__))

root_dir = os.path.join(module_dir,'fair-universe','Competition_Bundle_HEP_Scores_Stability')
ingestion_dir = os.path.join(root_dir,'ingestion_program')
bootstrapper_dir = os.path.join(root_dir,'sample_code_submission')

path.append(ingestion_dir)
path.append(bootstrapper_dir)

from systematics import Systematics
from bootstrap import bootstrap_data

# Load the CSV file
def dataGenerator(verbose=0):
    
    # Get the directory of the current script (my_module.py)
    module_dir = os.path.dirname(os.path.realpath(__file__))
    
    # Construct the absolute path to something.csv
    csv_file_path = os.path.join(module_dir, 'reference_data.csv')
    df = pd.read_csv(csv_file_path)

    # Remove the "label" and "weights" columns from the data


    flag = df.pop('Process_flag')
    label = df.pop('Label')
    weights = df.pop('Weight')
    entry = df.pop('entry')
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
    train_label_path =  os.path.join(root_dir, 'input_data','train','labels')
    if not os.path.exists(train_label_path):
        os.makedirs(train_label_path)

    train_weights_path =  os.path.join(root_dir, 'input_data','train','weights')
    if not os.path.exists(train_weights_path):
        os.makedirs(train_weights_path)

    train_data_path =  os.path.join(root_dir, 'input_data','train','data')
    if not os.path.exists(train_data_path):
        os.makedirs(train_data_path)

    train_settings_path =  os.path.join(root_dir, 'input_data','train','settings')
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

    # Save the label and weight files for the training set
    train_label_path =  os.path.join(train_label_path, 'data.labels')
    train_label.to_csv(train_label_path, index=False, header=False)

    train_weights_path =  os.path.join(train_weights_path, 'data.weights')
    train_weights.to_csv(train_weights_path, index=False, header=False)

    subset_signal_weight_test = np.sum(test_weights[test_label==1])
    subset_background_weight_test = np.sum(test_weights[test_label==0])

    test_weights[test_label==1] *= total_signal_weight / subset_signal_weight_test
    test_weights[test_label==0] *= total_background_weight / subset_background_weight_test
    

    for i in range(0, 10):
        test_set = []
        if i == 10:
            break
        # Calculate the sum of weights of signal and background for the current subset

        mu = round(np.random.uniform(0.01, 3.0), 2)
        # mu = 1.0
        if verbose > 1:
            print(f'[*] --- mu = {mu}')


        reference_label_path =  os.path.join(root_dir, 'reference_data',f'set_{i}','labels')
        if not os.path.exists(reference_label_path):
            os.makedirs(reference_label_path)

        reference_settings_path =  os.path.join(root_dir, 'reference_data',f'set_{i}','settings')
        if not os.path.exists(reference_settings_path):
            os.makedirs(reference_settings_path)

        test_weights_path =  os.path.join(root_dir, 'input_data','test',f'set_{i}','weights')
        if not os.path.exists(test_weights_path):
            os.makedirs(test_weights_path)

        test_data_path =  os.path.join(root_dir, 'input_data','test',f'set_{i}','data')
        if not os.path.exists(test_data_path):
            os.makedirs(test_data_path)

        tes_s = []
        # Adjust the weights of the current subset to match the weights of the whole data

        test_weights_set = test_weights.copy()

        test_weights_set[test_label==1] *= mu

        print (f"[*] --- mu in test set {i} : ", mu)
        print (f"[*] --- Signal in  test set {i}" , np.sum(test_weights_set[test_label==1]))
        print (f"[*] --- Background in  test set {i}" , np.sum(test_weights_set[test_label==0]))
        print (f"[*] --- Total Events in  test set {i}" , np.sum(test_weights_set)) 

        print (f"[*] --- avegare weight of signal in test set : ", np.mean(test_weights_set[test_label==1]))
        print (f"[*] --- avegare weight of background in test set : ", np.mean(test_weights_set[test_label==0]))




        for j in range(0, 100):



            bootstrap_test_data = bootstrap_data(data = test_df, weights = test_weights_set, label =  test_label, n = 20000, seed=42 + j) 

            test_df_bs = bootstrap_test_data['data']
            test_weights_bs = bootstrap_test_data['weights']
            test_label_bs = bootstrap_test_data['label']


            #adding systematics to the test set

            # Extract the TES information from the JSON file
            tes = round(np.random.uniform(0.9, 1.10), 2)
            # tes = 1.0

            tes_s.append(tes)

            test_syst = test_df_bs

            data_syst = Systematics(
            data=test_syst,
            verbose=verbose,
            tes=tes
            ).data


            # Save the current subset as a CSV file
            data_file_path = os.path.join(test_data_path, f'data_{j}.csv')
            weights_file_path = os.path.join(test_weights_path, f'data_{j}.weights')
            labels_file_path = os.path.join(reference_label_path, f'data_{j}.labels')

            # Writing data to files
            data_syst.to_csv(data_file_path, index=False)
            test_weights_bs.to_csv(weights_file_path, index=False, header=False)
            test_label_bs.to_csv(labels_file_path, index=False, header=False)

            test_background_weight = np.sum(test_weights_bs[test_label_bs==0])
            test_signal_weight = np.sum(test_weights_bs[test_label_bs==1])


            test_set_info = {
                "signal_weight": test_signal_weight,
                "background_weight": test_background_weight,
                "tes": tes
            }

            test_set.append(test_set_info)

            del data_syst
            del test_syst

        test_set = pd.DataFrame(test_set)
        if verbose > 1:


            print (f"[*] --- Shape of test set : ",np.shape(test_df))
            print (f"[*] --- Signal in  test set {i}" , np.mean(test_set['signal_weight']))
            print (f"[*] --- Background in  test set {i}" , np.mean(test_set['background_weight']))
            print (f"[*] --- Total Events in  test set {i}" ,   np.mean(test_set['signal_weight']) + np.mean(test_set[:]['background_weight']))
        
        # Save the mu and TES information to a JSON file
        test_settings = {"tes": tes_s, "ground_truth_mu": mu}
        Settings_file_path = os.path.join(reference_settings_path, 'data.json')
        with open(Settings_file_path, 'w') as json_file:
            json.dump(test_settings, json_file, indent=4)


    # Save the training set as a CSV file
    train_data_path = os.path.join(train_data_path, 'data.csv')
    train_df.to_csv(train_data_path, index=False)

    if verbose > 0:
        print ("Shape of train set : ",np.shape(train_df)) 



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

