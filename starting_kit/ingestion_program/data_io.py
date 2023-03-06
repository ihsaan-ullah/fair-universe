# This file consists of functions for 
# > Data Loading
# > Data Checking
# > Writing Predictions
# > Zip files


#-------------------------------------
# Imports
#-------------------------------------
import os
import numpy as np
import pandas as pd
from zipfile import ZipFile, ZIP_DEFLATED
from contextlib import closing
import json


#-------------------------------------
# Load Data
#-------------------------------------
def load_data (data_dir, load_settings=False) :


    print("\n\n###-------------------------------------###")
    print("### Data Loading")
    print("###-------------------------------------###\n")

    # set train and test directories
    train_data_dir = os.path.join(data_dir,"train", "data")
    train_labels_dir = os.path.join(data_dir,"train", "labels")
    test_data_dir = os.path.join(data_dir,"test", "data")
    test_labels_dir = os.path.join(data_dir,"test", "labels")
    settings_dir = os.path.join(data_dir,"settings")
    
    # print directories
    print("[*] data dir : ", data_dir)
    print("[*] train data dir : ", train_data_dir)
    print("[*] train labels dir : ", train_labels_dir)
    print("[*] test data dir : ", test_data_dir)
    print("[*] test labels dir : ", test_labels_dir)
    if load_settings:
        print("[*] settings dir : ", settings_dir)


    # check if directories exist
    if not os.path.exists(train_data_dir):
        print("[-] train data dir : ", train_data_dir, " not found")
        return
    else:
        print("[+] train data dir found")

    if not os.path.exists(train_labels_dir):
        print("[-] train labels dir : ", train_labels_dir, " not found")
        return
    else:
        print("[+] train labels dir found")

    if not os.path.exists(test_data_dir):
        print("[-] test data dir : ", test_data_dir, " not found")
        return
    else:
        print("[+] test data dir found")

    if not os.path.exists(test_labels_dir):
        print("[!] test labels dir : ", test_labels_dir, " not found")
    else:
        print("[+] test labels dir found")


    if load_settings:
        if not os.path.exists(settings_dir):
            print("[!] settings dir : ", settings_dir, " not found")
        else:
            print("[+] settings dir found")
    

    # train and test files
    train_data_files = [f for f in os.listdir(train_data_dir) if f.endswith('.csv')]
    train_labels_files = [f for f in os.listdir(train_labels_dir) if f.endswith('.labels')]
    test_data_files = [f for f in os.listdir(test_data_dir) if f.endswith('.csv')]


    # check if files exist
    if len(train_data_files) != len(train_labels_files) != len(test_data_files):
        print("[-] Number of train data, train labels, test data files do not match! ")
        return
    

    total_files = len(train_data_files)

    print("[+] {} train and test sets found".format(total_files))
    

    train_sets, test_sets, settings = [], [], []


    for i in range(0,total_files):


        train_data_file = "train_"+str(i+1)+".csv"
        test_data_file = "test_"+str(i+1)+".csv"
        train_labels_file = "train_"+str(i+1)+".labels"
        test_labels_file = "test_"+str(i+1)+".labels"
        settings_file = "settings_"+str(i+1)+".json"

        train_data_file_path = os.path.join(train_data_dir, train_data_file)
        test_data_file_path = os.path.join(test_data_dir, test_data_file)
        train_labels_file_path = os.path.join(train_labels_dir, train_labels_file)
        test_labels_file_path = os.path.join(test_labels_dir, test_labels_file)
        settings_file_path = os.path.join(settings_dir, settings_file)
        
        
        train_sets.append({
            "data" : read_data_file(train_data_file_path),
            "labels" : read_labels_file(train_labels_file_path)
        })

        if os.path.exists(test_labels_file_path):
            test_sets.append({
                "data" : read_data_file(test_data_file_path),
                "labels" : read_labels_file(test_labels_file_path),
            })
        else:
            test_sets.append({
                "data" : read_data_file(test_data_file_path),
            })
        
        if load_settings:
            settings.append(read_json_file(settings_file_path))

    
    print("---------------------------------")
    print("[+] Train and Test data loaded!")
    print("---------------------------------\n\n")
    if load_settings:
        return train_sets, test_sets, settings
    else:
        return train_sets, test_sets

#-------------------------------------
# Read Data File
#-------------------------------------
def read_data_file(data_file):

    # check data file
    if not os.path.isfile(data_file):
        print("[-] data file {} does not exist".format(data_file))
        return

    # Load data file
    df = pd.read_csv(data_file)

    return df

#-------------------------------------
# Read Labels File
#-------------------------------------
def read_labels_file(labels_file):

    # check labels file
    if not os.path.isfile(labels_file):
        print("[-] labels file {} does not exist".format(labels_file))
        return

    # Read labels file
    f = open(labels_file, "r")
    labels = f.read().splitlines()
    labels = np.array(labels,dtype=float)
    f.close()
    return labels

#-------------------------------------
# Read Json File
#-------------------------------------
def read_json_file(json_file):

    # check json file
    if not os.path.isfile(json_file):
        print("[-] json file {} does not exist".format(json_file))
        return
    return json.load(open(json_file))

#-------------------------------------
# Data Statistics 
#-------------------------------------
def show_data_statistics(data_sets, name="Train"):

    print("###-------------------------------------###")
    print("### Data Statistics " + name)
    print("###-------------------------------------###")

    for index, data_set in enumerate(data_sets):
        print("-------------------")
        print("Set " + str(index+1))
        print("-------------------")

        print("[*] Total points: ", data_set["data"].shape[0])
        if "labels" in data_set:
            print("[*] Background points: ", len(data_set["labels"]) - np.count_nonzero(data_set["labels"] == 1))
            print("[*] Signal points: ", np.count_nonzero(data_set["labels"] == 1))

#-------------------------------------
# Write Predictions 
#-------------------------------------
def write(filename, predictions):

    with open(filename, 'w') as f:
        for ind, lbl in enumerate(predictions):
            str_label = str(float(lbl))
            if ind < len(predictions)-1:
                f.write(str_label + "\n")
            else:
                f.write(str_label)

#-------------------------------------
# Zip files 
#-------------------------------------
def zipdir(archivename, basedir):
    '''Zip directory, from J.F. Sebastian http://stackoverflow.com/'''
    assert os.path.isdir(basedir)
    with closing(ZipFile(archivename, "w", ZIP_DEFLATED)) as z:
        for root, dirs, files in os.walk(basedir):
            #NOTE: ignore empty directories
            for fn in files:
                if fn[-4:]!='.zip' and fn!='.DS_Store' :
                    absfn = os.path.join(root, fn)
                    zfn = absfn[len(basedir):] #XXX: relative path
                    z.write(absfn, zfn)
                    
