import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Load the CSV file
df = pd.read_csv('reference_data.csv')

# Split the data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.3)

# Create a directory to store the test files
if not os.path.exists('./input_data/test/data'):
    os.makedirs('./input_data/test/data')

if not os.path.exists('./input_data/train/data'):
    os.makedirs('./input_data/train/data')

# Divide the test set into 10 equal parts and save each part as a separate CSV file
test_dfs = [test_df[i:i+len(test_df)//10] for i in range(0, len(test_df), len(test_df)//10)]
for i, test_df in enumerate(test_dfs):
    test_df.to_csv(f'input_data/test/data/data_{i}.csv', index=False)

# Save the training and testing sets as CSV files
train_df.to_csv('input_data/train/data/data.csv', index=False)

