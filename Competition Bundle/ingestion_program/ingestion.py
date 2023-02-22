#------------------------------------------
# Imports
#------------------------------------------
import sys
import os
import numpy as np
import pandas as pd 
import time


#------------------------------------------
# Directories
#------------------------------------------
# Input data directory to read training data from
input_dir = '/app/input_data/' 

# Output data directory to write predictions to
output_dir = '/app/output/'    

# Program directory
program_dir = '/app/program'

# Directory to read submitted submissions from
submission_dir = '/app/ingested_program'

sys.path.append(output_dir)
sys.path.append(program_dir)
sys.path.append(submission_dir)



#------------------------------------------
# Read Train Data
#------------------------------------------
def get_training_data():

    train_dir = os.path.join(input_dir, 'train')
    
    # train data
    train_data_files = [
        os.path.join(train_dir, "train_1.csv"),
        os.path.join(train_dir, "train_2.csv"),
        os.path.join(train_dir, "train_3.csv")
    ]

    # train labels
    train_solution_files = [
        os.path.join(train_dir, "train_1.labels"),
        os.path.join(train_dir, "train_2.labels"),
        os.path.join(train_dir, "train_3.labels")
    ]

    X_trains , y_trains = [], []
    for index, file in enumerate(train_data_files):
    
        train_data_file = train_data_files[index]
        train_solution_file = train_solution_files[index]
    
        # Read Train data
        X_train = pd.read_csv(train_data_file)

        # Read Train solution
        f = open(train_solution_file, "r")
        y_train = f.read().splitlines()
        y_train = np.array(y_train,dtype=float)

        X_trains.append(X_train)
        y_trains.append(y_train)

    return X_trains, y_trains

#------------------------------------------
# Read Test Data
#------------------------------------------
def get_prediction_data():

    test_dir = os.path.join(input_dir, 'test')

    # test data
    test_data_files = [
        os.path.join(test_dir, "test_1.csv"),
        os.path.join(test_dir, "test_2.csv"),
        os.path.join(test_dir, "test_3.csv")
    ]

    X_tests = []
    for test_data_file in test_data_files:
        # Read Test data
        X_test = pd.read_csv(test_data_file)

        X_tests.append(X_test)



    return X_tests

#------------------------------------------
# Save Predictions
#------------------------------------------
def save_prediction(file_name, prediction_prob):

    prediction_file = os.path.join(output_dir, file_name)

    predictions = prediction_prob[:,1]

    with open(prediction_file, 'w') as f:
        for ind, lbl in enumerate(predictions):
            str_label = str(int(lbl))
            if ind < len(predictions)-1:
                f.write(str_label + "\n")
            else:
                f.write(str_label)

    
def print_pretty(text):
    print("-------------------")
    print("#---",text)
    print("-------------------")

#------------------------------------------
# Run the pipeline 
# > Load 
# > Trein 
# > Predict 
# > Save
#------------------------------------------
def main():

    #------------------------------------------
    # Start Timer
    #------------------------------------------
    start = time.time()


    #------------------------------------------
    # Import Model
    #------------------------------------------
    from model import Model
   
    #------------------------------------------
    # Read Data
    #------------------------------------------
    print_pretty('Reading Data')
    X_trains, y_trains = get_training_data()
    X_tests = get_prediction_data()

    for index, _ in enumerate(X_trains):

        print("Dataset :", index+1)

        #------------------------------------------
        # Load Model
        #------------------------------------------
        print_pretty('Starting Learning')
        m = Model()

        #------------------------------------------
        # Train Model
        #------------------------------------------
        print_pretty('Training Model')
        m.fit(X_trains[index], y_trains[index])

        #------------------------------------------
        # Make Predictions
        #------------------------------------------
        print_pretty('Making Prediction')
        prediction_prob = m.predict_score(X_tests[index])

        #------------------------------------------
        # Save  Predictions
        #------------------------------------------
        print_pretty('Saving Prediction')
        prediction_file_name = "test_"+str(index+1)+".predictions"
        save_prediction(prediction_file_name, prediction_prob)


    #------------------------------------------
    # Show Ingestion Time
    #------------------------------------------
    duration = time.time() - start
    print_pretty(f'Total duration: {duration}')

if __name__ == '__main__':
    main()