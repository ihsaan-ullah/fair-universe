# This file consists of functions for 
# > reading solutions
# > reading predictions
# > writing score

#------------------------------------------
# Imports
#------------------------------------------
import os
import numpy as np
import json

#------------------------------------------
# Read Predictions and solutions
#------------------------------------------
def read_pred_sol(prediction_dir, solution_dir):

    
    # predicion + score, solution files
    prediction_files = [f for f in os.listdir(prediction_dir) if f.endswith('.predictions')]
    score_files = [f for f in os.listdir(prediction_dir) if f.endswith('.scores')]
    solution_files = [f for f in os.listdir(solution_dir) if f.endswith('.labels')]





    print("[*] Predictions files : {}".format(prediction_files))
    print("[*] Score files : {}".format(score_files))
    print("[*] Solutions files : {}".format(solution_files))

    # check if files exist
    if len(prediction_files) != len(score_files) != len(solution_files):
        print("[-] Number of test labels,  prediction, and socre files do not match!")
        return

    
    total_files = len(solution_files)

    print("[+] {} prediction, score, solution files found".format(total_files))
    

    predicted_targets, predicted_scores, test_labels  = [], [], []

    for i in range(0,total_files):

        prediction_file = "test_"+str(i+1)+".predictions"
        score_file = "test_"+str(i+1)+".scores"
        solution_file = "test_"+str(i+1)+".labels"

        prediction_file_path = os.path.join(prediction_dir, prediction_file)
        score_file_path = os.path.join(prediction_dir, score_file)
        solution_file_path = os.path.join(solution_dir, solution_file)

        predicted_targets.append(read_file(prediction_file_path))
        predicted_scores.append(read_file(score_file_path))
        test_labels.append(read_file(solution_file_path))

    print("---------------------------------")
    print("[+] Predictions and Solution files loaded!")
    print("---------------------------------\n\n")
   


    return predicted_targets, test_labels, predicted_scores



#------------------------------------------
# Read Predictions
#------------------------------------------
def read_file(file_name):

    # check labels file
    if not os.path.isfile(file_name):
        print("[-]  File {} does not exist".format(file_name))
        return

    # Read labels file
    f = open(file_name, "r")
    file_data = f.read().splitlines()
    file_data = np.array(file_data,dtype=float)
    f.close()
    return file_data
    



#------------------------------------------
# Write Score
#------------------------------------------
def write_score(score_file, scores):

    with open(score_file, 'w') as f_score:
        f_score.write(json.dumps(scores, indent=4))
        f_score.close()

