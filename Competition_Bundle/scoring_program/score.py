#------------------------------------------
# Imports
#------------------------------------------
import sys
import os
import json
import numpy as np



#------------------------------------------
# Import Metric
#------------------------------------------
from metric import auc_metric


#------------------------------------------
# constants
#------------------------------------------
data_name = "fair_universe"

#------------------------------------------
# directories
#------------------------------------------
# Directory read predictions and solutions from
input_dir = '/app/input' 

# Directory to output computed score into
output_dir = '/app/output/'

# reference data (test labels)
reference_dir = os.path.join(input_dir, 'ref')  # Ground truth data

# submitted/predicted lables
prediction_dir = os.path.join(input_dir, 'res')

# score file to write score into
score_file = os.path.join(output_dir, 'scores.json') 



#------------------------------------------
# Read Predictions
#------------------------------------------
def read_prediction():
    prediction_files = [
        os.path.join(prediction_dir,'test_1.predictions'),
        os.path.join(prediction_dir,'test_2.predictions'),
        os.path.join(prediction_dir,'test_3.predictions')
    ]
    
    predicted_scores = []
    for prediction_file in prediction_files:

        # Check if file exists
        if not os.path.isfile(prediction_file):
            print("[-] " + prediction_file +" file not found!")
            return

        f = open(prediction_file, "r")
    
        predicted_score = f.read().splitlines()
        predicted_score = np.array(predicted_score,dtype=float)

        predicted_scores.append(predicted_score)
    return predicted_scores

#------------------------------------------
# Read Solutions
#------------------------------------------
def read_solution():

    solution_files = [
        os.path.join(reference_dir, 'test_1.labels'),
        os.path.join(reference_dir, 'test_2.labels'),
        os.path.join(reference_dir, 'test_3.labels')
    ]

    test_labels = []
    for solution_file in solution_files:

        # Check if file exists
        if not os.path.isfile(solution_file):
            print('[-] Test solution file not found!')
            return

        f = open(solution_file, "r")
        
        test_label = f.read().splitlines()
        test_label = np.array(test_label,dtype=float)

        test_labels.append(test_label)
    return test_labels

def save_score(scores):

    scores = {
        "auc_1": scores[0],
        "auc_2": scores[1],
        "auc_3": scores[2],
        "auc":  np.mean(scores)
    }
    with open(score_file, 'w') as f_score:
        f_score.write(json.dumps(scores))
        f_score.close()

def print_pretty(text):
    print("-------------------")
    print("#---",text)
    print("-------------------")


    
def main():


    #------------------------------------------
    # Read predictions and solutions
    #------------------------------------------
    print_pretty('Reading predictions')
    predictions = read_prediction()

    print_pretty('Reading solutions')
    solutions = read_solution()



    #------------------------------------------
    # Compute Scores
    #------------------------------------------
    print_pretty('Computing scores')
    auc_scores = []
    for index, _ in enumerate(predictions):
        auc_score = auc_metric(solutions[index], predictions[index])
        print("AUC Score for Prediction " + str(index+1)+" :"  , auc_score)
        auc_scores.append(auc_score)

    #------------------------------------------
    # Write Score
    #------------------------------------------
    print_pretty('Saving Score')
    save_score(auc_scores)
        
    
    




if __name__ == '__main__':
    main()