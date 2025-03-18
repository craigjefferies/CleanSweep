# CleanSweep
Python based dataset cleaning functions 


## Example Usage

Copy and paste the code below into your Jupyter Notebook or Python script:

```python
# Clone the repository from GitHub (if you haven't already)
!git clone https://github.com/craigjefferies/CleanSweep.git

# Add the repository directory to your Python path
import sys
sys.path.append('./CleanSweep')

import pandas as pd
from CleanSweep import automated_cleaning_pipeline, print_cleaning_report

# Load your dataset from a CSV file (ensure data.csv is in the working directory)
df = pd.read_csv('data.csv')

# Run the cleaning pipeline with your desired imputation strategies.
# The human-readable cleaning report will be generated and printed.
cleaned_df, cleaning_report = automated_cleaning_pipeline(
    df,
    numeric_imputation_strategy='median',          # Options: 'median', 'mean', 'constant'
    categorical_imputation_strategy='most_frequent'  # Options: 'most_frequent', 'constant'
)

# Print the human-readable cleaning report
print_cleaning_report(cleaning_report)

# Optionally, save the cleaned DataFrame to a new CSV file
cleaned_df.to_csv('data_cleaned.csv', index=False)
print("Cleaned data saved to 'data_cleaned.csv'.")

# Print the cleaning report to see the details of the cleaning process
print("Cleaning Report:")
print(cleaning_report)

# Optionally, save the cleaned dataset to a new CSV file
cleaned_df.to_csv('data_cleaned.csv', index=False)
print("Cleaned data saved to 'data_cleaned.csv'.")
