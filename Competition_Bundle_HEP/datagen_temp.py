import pandas as pd
from sklearn.model_selection import train_test_split
import os


# Load the CSV file
df = pd.read_csv('./reference_data.csv')

# Remove the "label" and "weights" columns from the data
label = df.pop('Label')
weights = df.pop('Weight')

# Split the data into training and testing sets
train_df, test_df, train_label, test_label, train_weights, test_weights = train_test_split(df, label, weights, test_size=0.3)

# Create directories to store the label and weight files
if not os.path.exists('./input_data/train/labels'):
    os.makedirs('./input_data/train/labels')

if not os.path.exists('./input_data/train/weights'):
    os.makedirs('./input_data/train/weights')

# Save the label and weight files for the training set
train_label.to_csv('./input_data/train/label/data.label', index=False)
train_weights.to_csv('./input_data/train/weights/data.weights', index=False)


# Divide the test set into 10 equal parts and save each part as a separate CSV file
test_dfs = [test_df[i:i+len(test_df)//10] for i in range(0, len(test_df), len(test_df)//10)]
for i, test_df in enumerate(test_dfs):
    test_df.to_csv(f'./input_data/test/data/data_{i}.csv', index=False)


# Save the training set as a CSV file
train_df.to_csv('./input_data/train/data/data.csv', index=False)
