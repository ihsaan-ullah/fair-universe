import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os


# Load the CSV file
df = pd.read_csv('./reference_data.csv')

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
for feature in features:
    print(feature)



print ("sum of weights : ", np.sum(weights))
print ("sum of signal" , np.sum(weights[label==1]))
print ("sum of background" , np.sum(weights[label==0]))

# Calculate the sum of weights of signal and background for the whole data
total_weight = np.sum(weights)
total_signal_weight = np.sum(weights[label==1])
total_background_weight = np.sum(weights[label==0])

# Split the data into training and testing sets
train_df, test_df, train_label, test_label, train_weights, test_weights = train_test_split(df, label, weights, test_size=0.3)

# Create directories to store the label and weight files
if not os.path.exists('./input_data/train/labels'):
    os.makedirs('./input_data/train/labels')

if not os.path.exists('./input_data/train/weights'):
    os.makedirs('./input_data/train/weights')

if not os.path.exists('./input_data/test/weights'):
    os.makedirs('./input_data/test/weights')
if not os.path.exists('./input_data/test/labels'):
    os.makedirs('./input_data/test/labels')
    
subset_weight = np.sum(train_weights)
subset_signal_weight = np.sum(train_weights[train_label==1])
subset_background_weight = np.sum(train_weights[train_label==0])


# Adjust the weights of the training set to match the weights of the whole data

train_weights[train_label==1] *= total_signal_weight / subset_signal_weight
train_weights[train_label==0] *= total_background_weight / subset_background_weight

print ("sum of signal" , np.sum(train_weights[train_label==1]))
print ("sum of background" , np.sum(train_weights[train_label==0]))




# Save the label and weight files for the training set
train_label.to_csv('./input_data/train/labels/data.labels', index=False, header=False)
train_weights.to_csv('./input_data/train/weights/data.weights', index=False, header=False)

# Divide the test set into 10 equal parts and save each part as a separate CSV file
test_dfs = [test_df[i:i+len(test_df)//10] for i in range(0, len(test_df), len(test_df)//10)]
test_weights_ = [test_weights[i:i+len(test_weights)//10] for i in range(0, len(test_weights), len(test_weights)//10)]
test_label_ = [test_label[i:i+len(test_label)//10] for i in range(0, len(test_label), len(test_label)//10)]

for i, (test_df, test_weights) in enumerate(zip(test_dfs, test_weights_)):
    # Calculate the sum of weights of signal and background for the current subset
    subset_weight = np.sum(test_weights)
    subset_signal_weight = np.sum(test_weights[test_label==1])
    subset_background_weight = np.sum(test_weights[test_label==0])
    
    # Adjust the weights of the current subset to match the weights of the whole data
    
    test_weights[test_label==1] *= total_signal_weight / subset_signal_weight
    test_weights[test_label==0] *= total_background_weight / subset_background_weight
    
    # Save the current subset as a CSV file
    test_df.to_csv(f'./input_data/test/data/data_{i}.csv', index=False)
    test_weights.to_csv(f'./input_data/test/weights/data_{i}.weights', index=False, header=False)
    test_label.to_csv(f'./input_data/test/labels/data_{i}.labels', index=False, header=False)


    # print ("sum of signal" , np.sum(test_weights[test_label==1]))
    # print ("sum of background" , np.sum(test_weights[test_label==0]))

# Save the training set as a CSV file
train_df.to_csv('./input_data/train/data/data.csv', index=False)

print ("Shape of test set : ",np.shape(test_df))
print ("Shape of train set : ",np.shape(train_df)) 
