#------------------------------------------
# Imports
#------------------------------------------
from sys import argv, path
import os
import numpy as np
import pandas as pd 
import time




#------------------------------------------
# Default Directories
#------------------------------------------
if len(argv) == 1:
    # root directory
    root_dir = "./"
    # Input data directory to read training data from
    input_dir = root_dir + "sample_data"
    # Output data directory to write predictions to
    output_dir = root_dir + "sample_result_submission"
    # Program directory
    program_dir = root_dir + "ingestion_program"
    # Directory to read submitted submissions from
    submission_dir = root_dir + "sample_code_submission"
#------------------------------------------
# Codabench Directories
#------------------------------------------
else:
    # Input data directory to read training data from
    input_dir = '/app/input_data/' 
    # Output data directory to write predictions to
    output_dir = '/app/output/' 
    # Program directory 
    program_dir = '/app/program'
    # Directory to read submitted submissions from
    submission_dir = '/app/ingested_program'
    
       
    
path.append(output_dir)
path.append(program_dir)
path.append(submission_dir)


if __name__ == '__main__':

    print("############################################")
    print("### Ingestion Program")
    print("############################################")

    #------------------------------------------
    # Start Timer
    #------------------------------------------
    start = time.time()

    #------------------------------------------
    # Read Data
    #------------------------------------------
    from data_io import load_data, write, show_data_statistics
    train_sets, test_sets = load_data(input_dir)


    #------------------------------------------
    # Import Model
    #------------------------------------------
    from model import Model
   
    for index, _ in enumerate(train_sets):
        
        
        print("\n[*] Dataset :", index+1)
        print("-----------")


        #------------------------------------------
        # Load Model
        #------------------------------------------
        print("[*] Loading Model")
        m = Model(
            X_train=train_sets[index]["data"], 
            Y_train=train_sets[index]["labels"], 
            X_test=test_sets[index]["data"],
        )
        

        #------------------------------------------
        # Train Model
        #------------------------------------------
        print("[*] Training Model")
        m.fit()

        #------------------------------------------
        # Make Predictions
        #------------------------------------------
        print("[*] Making Predictions")
        predictions = m.predict()
        scores = m.decision_function()

        #------------------------------------------
        # Save Predictions
        #------------------------------------------
        print("[*] Saving Predictions")
        prediction_file_name = "test_"+str(index+1)+".predictions"
        prediction_file_path = os.path.join(output_dir, prediction_file_name)
        write(prediction_file_path, predictions)

        #------------------------------------------
        # Save Scores
        #------------------------------------------
        print("[*] Saving Scores")
        score_file_name = "test_"+str(index+1)+".scores"
        score_file_path = os.path.join(output_dir, score_file_name)
        write(score_file_path, scores)



    #------------------------------------------
    # Show Ingestion Time
    #------------------------------------------
    duration = time.time() - start
    print("\n---------------------------------")
    print(f'[*] Total duration: {duration}')
    print("---------------------------------")


    print("\n----------------------------------------------")
    print("[+] Ingestions Program executed successfully!")
    print("----------------------------------------------\n\n")


    


