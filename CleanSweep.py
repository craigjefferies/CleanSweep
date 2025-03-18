import pandas as pd
import numpy as np
import logging
from sklearn.impute import SimpleImputer

# Configure logging to display time, level, and message.
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def check_data_quality(df):
    """
    Compute initial data quality metrics for the DataFrame.
    
    Returns a dictionary with:
      - missing_values: Count of missing values per column.
      - duplicates: Total duplicate rows.
      - total_rows: Total number of rows.
      - memory_usage: DataFrame memory usage in MB.
    """
    quality_report = {
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum(),
        'total_rows': len(df),
        'memory_usage': df.memory_usage().sum() / 1024**2  # in MB
    }
    logging.info("Initial data quality metrics computed.")
    return quality_report

def standardize_datatypes(df):
    """
    Standardize datatypes in the DataFrame:
      - Convert object columns to datetime (using errors='coerce').
      - If datetime conversion fails, attempt numeric conversion after cleaning common symbols.
    
    Logs the conversion status for each column.
    """
    for column in df.columns:
        if df[column].dtype == 'object':
            try:
                # Attempt to convert to datetime
                df[column] = pd.to_datetime(df[column], errors='coerce')
                if df[column].notnull().all():
                    logging.info(f"Column '{column}' converted entirely to datetime.")
                else:
                    logging.info(f"Column '{column}' partially converted to datetime; non-date values set to NaT.")
            except Exception as e:
                logging.error(f"Error converting column '{column}' to datetime: {e}")
                try:
                    # Clean symbols and attempt numeric conversion
                    df[column] = pd.to_numeric(df[column].str.replace('$', '', regex=False).str.replace(',', '', regex=False), errors='coerce')
                    logging.info(f"Column '{column}' converted to numeric.")
                except Exception as e2:
                    logging.error(f"Error converting column '{column}' to numeric: {e2}")
    return df

def handle_missing_values(df, numeric_strategy='median', categorical_strategy='most_frequent', fill_value=None, log_changes=True):
    """
    Impute missing values in the DataFrame using customizable strategies.
    
    Parameters:
      - numeric_strategy: Strategy for numeric imputation ('median', 'mean', 'constant', etc.).
      - categorical_strategy: Strategy for categorical imputation ('most_frequent', 'constant', etc.).
      - fill_value: Value to use when strategy is 'constant'.
      - log_changes: If True, returns a report of imputation changes.
      
    Returns:
      - If log_changes is True: tuple (df, imputation_report)
      - Otherwise: the modified DataFrame.
    """
    imputation_report = {}
    
    # Impute numeric columns
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_columns) > 0:
        if numeric_strategy == 'constant' and fill_value is not None:
            num_imputer = SimpleImputer(strategy='constant', fill_value=fill_value)
        else:
            num_imputer = SimpleImputer(strategy=numeric_strategy)
        before_numeric = df[numeric_columns].isnull().sum()
        df[numeric_columns] = num_imputer.fit_transform(df[numeric_columns])
        after_numeric = df[numeric_columns].isnull().sum()
        if log_changes:
            imputation_report['numeric'] = (before_numeric - after_numeric).to_dict()
            logging.info(f"Numeric imputation changes: {imputation_report['numeric']}")
    
    # Impute categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    if len(categorical_columns) > 0:
        if categorical_strategy == 'constant' and fill_value is not None:
            cat_imputer = SimpleImputer(strategy='constant', fill_value=fill_value)
        else:
            cat_imputer = SimpleImputer(strategy=categorical_strategy)
        before_categorical = df[categorical_columns].isnull().sum()
        df[categorical_columns] = cat_imputer.fit_transform(df[categorical_columns])
        after_categorical = df[categorical_columns].isnull().sum()
        if log_changes:
            imputation_report['categorical'] = (before_categorical - after_categorical).to_dict()
            logging.info(f"Categorical imputation changes: {imputation_report['categorical']}")
    
    if log_changes:
        return df, imputation_report
    else:
        return df

def remove_outliers(df):
    """
    Identify and cap outliers in numeric columns using the IQR method.
    Instead of removing rows, values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR] are clipped.
    
    Returns:
      - The modified DataFrame.
      - A dictionary with counts of outliers capped per numeric column.
    """
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    outliers_removed = {}
    
    for column in numeric_columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Count outliers before clipping
        outlier_count = df[(df[column] < lower_bound) | (df[column] > upper_bound)].shape[0]
        df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
        
        if outlier_count > 0:
            outliers_removed[column] = outlier_count
            logging.info(f"Clipped {outlier_count} outliers in column '{column}'.")
            
    return df, outliers_removed

def validate_cleaning(df, original_shape, cleaning_report):
    """
    Validate the cleaning process by comparing the cleaned DataFrame to the original data.
    
    Reports:
      - Rows remaining.
      - Total missing values remaining.
      - Duplicates remaining.
      - Data loss percentage (if rows were dropped).
      
    Updates and returns the cleaning report with these metrics.
    """
    validation_results = {
        'rows_remaining': len(df),
        'missing_values_remaining': df.isnull().sum().sum(),
        'duplicates_remaining': df.duplicated().sum(),
        'data_loss_percentage': (1 - len(df) / original_shape[0]) * 100
    }
    
    cleaning_report['validation'] = validation_results
    logging.info("Validation of cleaning process completed.")
    return cleaning_report

def automated_cleaning_pipeline(df, 
                                numeric_imputation_strategy='median', 
                                categorical_imputation_strategy='most_frequent', 
                                fill_value=None, 
                                log_imputation=True):
    """
    Execute the complete automated cleaning pipeline:
      1. Compute initial quality metrics.
      2. Standardize data types.
      3. Impute missing values (using specified strategies).
      4. Identify and cap outliers.
      5. Validate the cleaning process.
    
    Returns:
      - The cleaned DataFrame.
      - A comprehensive cleaning report (as a dictionary).
    """
    logging.info("Starting automated cleaning pipeline.")
    original_shape = df.shape
    cleaning_report = {}

    # Step 1: Initial quality check
    cleaning_report['initial_quality'] = check_data_quality(df)
    
    # Step 2: Standardize data types
    df = standardize_datatypes(df)
    
    # Step 3: Handle missing values
    df, imputation_report = handle_missing_values(df, 
                                                  numeric_strategy=numeric_imputation_strategy, 
                                                  categorical_strategy=categorical_imputation_strategy, 
                                                  fill_value=fill_value, 
                                                  log_changes=log_imputation)
    cleaning_report['imputation_report'] = imputation_report
    
    # Step 4: Remove (clip) outliers using IQR method
    df, outliers = remove_outliers(df)
    cleaning_report['outliers_removed'] = outliers
    
    # Step 5: Validate cleaning process
    cleaning_report = validate_cleaning(df, original_shape, cleaning_report)
    
    logging.info("Automated cleaning pipeline completed.")
    return df, cleaning_report

def print_cleaning_report(report):
    """
    Print a human-readable cleaning report.
    """
    print("\nCLEANING REPORT")
    print("=" * 40)
    
    # Initial Quality Metrics
    if 'initial_quality' in report:
        print("\nInitial Quality:")
        for key, value in report['initial_quality'].items():
            if isinstance(value, dict):
                print(f"  {key.replace('_', ' ').title()}:")
                for subkey, subval in value.items():
                    print(f"    {subkey}: {subval}")
            else:
                print(f"  {key.replace('_', ' ').title()}: {value}")
    
    # Imputation Report
    if 'imputation_report' in report:
        print("\nImputation Report:")
        for key, value in report['imputation_report'].items():
            if isinstance(value, dict):
                print(f"  {key.replace('_', ' ').title()}:")
                for subkey, subval in value.items():
                    print(f"    {subkey}: {subval}")
            else:
                print(f"  {key.replace('_', ' ').title()}: {value}")
    
    # Outliers Removed
    if 'outliers_removed' in report:
        print("\nOutliers Removed:")
        for key, value in report['outliers_removed'].items():
            print(f"  {key}: {value}")
    
    # Validation Metrics
    if 'validation' in report:
        print("\nValidation:")
        for key, value in report['validation'].items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
    
    print("=" * 40)

# If the module is executed directly, run an example cleaning process
if __name__ == '__main__':
    # For demonstration, attempt to load 'data.csv'. Adjust the filename as needed.
    try:
        df = pd.read_csv('data.csv')
        logging.info("Loaded 'data.csv' successfully.")
    except Exception as e:
        logging.error("Error loading 'data.csv'. Please ensure the file exists in the current directory.")
        raise e

    # Run the cleaning pipeline with default imputation strategies
    cleaned_df, cleaning_report = automated_cleaning_pipeline(
        df,
        numeric_imputation_strategy='median',
        categorical_imputation_strategy='most_frequent'
    )

    # Print a human-readable cleaning report
    print_cleaning_report(cleaning_report)

    # Optionally, save the cleaned data to a CSV file
    cleaned_df.to_csv('data_cleaned.csv', index=False)
    logging.info("Cleaned data saved to 'data_cleaned.csv'.")
