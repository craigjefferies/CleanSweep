# CleanSweep
Python based dataset cleaning functions 

# Clone the repository from GitHub
!git clone https://github.com/craigjefferies/CleanSweep.git

# Add the repository directory to the Python path so you can import CleanSweep.py
import sys
sys.path.append('./CleanSweep')

import pandas as pd
from CleanSweep import automated_cleaning_pipeline

# Load the dataset from data.csv
df = pd.read_csv('data.csv')

# Run the cleaning pipeline with your desired imputation strategies
cleaned_df, cleaning_report = automated_cleaning_pipeline(
    df,
    numeric_imputation_strategy='median',
    categorical_imputation_strategy='most_frequent'
)

# Print the cleaning report to see the details of the cleaning process
print("Cleaning Report:")
print(cleaning_report)

# Optionally, save the cleaned DataFrame to a new CSV file
cleaned_df.to_csv('data_cleaned.csv', index=False)
print("Cleaned data saved to 'data_cleaned.csv'.")

