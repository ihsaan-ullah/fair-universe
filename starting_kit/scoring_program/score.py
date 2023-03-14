#------------------------------------------
# Imports
#------------------------------------------
import os
from sys import argv
import numpy as np



#------------------------------------------
# Import Metric
#------------------------------------------
from metric import auc_metric, bac_metric



#------------------------------------------
# Default Directories
#------------------------------------------
if len(argv) == 1:
    # root directory
    root_dir = "./"
    
    # Directory read predictions and solutions from
    input_dir = root_dir + "sample_data"

    # Directory to output computed score into
    output_dir = root_dir + "scoring_output"

    # reference data (test labels)
    reference_dir = os.path.join(input_dir, "test", "labels")

    # submitted/predicted lables
    prediction_dir = root_dir + "sample_result_submission"

    # score file to write score into
    score_file = os.path.join(output_dir, 'scores.json')

else:
#------------------------------------------
# Codabench Directories
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

if __name__ == '__main__':

    print("############################################")
    print("### Scoring Program")
    print("############################################")




    from solution import read_pred_sol, write_score


    #------------------------------------------
    # Read predictions and solutions
    #------------------------------------------
    print("[*] Reading predictions and solutions")
    predictions, solutions, scores = read_pred_sol(prediction_dir, reference_dir)


    #------------------------------------------
    # Compute Scores
    #------------------------------------------
    print("[*] Computing AUC and BAC scores")

    scores_dict = {}
    auc_scores, bac_scores = [], []

    for index, _ in enumerate(predictions):
        print("\nTest set {}".format(index+1))
        print("-----------")
        auc_score = round(auc_metric(solutions[index], scores[index]), 2)
        bac_score = round(bac_metric(solutions[index], predictions[index]), 2)
        print("AUC: {}\nBAC: {}".format(auc_score, bac_score))

        # keys for scores dict
        auc_key = "auc_" + str(index+1)
        bac_key = "bac_" + str(index+1)

        # adding scores to dict
        scores_dict[auc_key] = auc_score
        scores_dict[bac_key] = bac_score

        # adding scores to list
        auc_scores.append(auc_score)
        bac_scores.append(bac_score)

    scores_dict["auc"] = round(np.mean(auc_scores), 2)
    scores_dict["bac"] = round(np.mean(bac_scores), 2)

    print("\n-----------")
    print("Average AUC : {}".format(scores_dict["auc"]))
    print("Average BAC : {}".format(scores_dict["bac"]))
    print("-----------\n")

    #------------------------------------------
    # Write Score
    #------------------------------------------
    print("[*] Saving Score")
    write_score(score_file, scores_dict)



    print("\n----------------------------------------------")
    print("[+] Scoring Program executed successfully!")
    print("----------------------------------------------\n\n")