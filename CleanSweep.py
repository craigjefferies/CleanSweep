import pandas as pd
import numpy as np
import logging
from sklearn.impute import SimpleImputer

# Configure logging to display the time, log level, and message.
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def check_data_quality(df):
    """
    Compute initial data quality metrics for the DataFrame.
    Returns a dictionary containing:
        - missing_values: Missing values count per column.
        - duplicates: Total number of duplicate rows.
        - total_rows: Total number of rows.
        - memory_usage: Memory usage in MB.
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
    Standardize datatypes in the DataFrame by:
        - Converting object columns to datetime where possible.
        - If datetime conversion is not fully successful, attempt numeric conversion.
    Logs the conversion status for each column.
    """
    for column in df.columns:
        if df[column].dtype == 'object':
            try:
                # Convert to datetime; use errors='coerce' to convert non-date strings to NaT.
                df[column] = pd.to_datetime(df[column], errors='coerce')
                # Log based on conversion success:
                if df[column].notnull().all():
                    logging.info(f"Column '{column}' converted entirely to datetime.")
                else:
                    logging.info(f"Column '{column}' partially converted to datetime; non-date values set to NaT.")
            except Exception as e:
                logging.error(f"Error converting column '{column}' to datetime: {e}")
                try:
                    # Attempt numeric conversion after cleaning common symbols
                    df[column] = pd.to_numeric(df[column].str.replace('$', '', regex=False).str.replace(',', '', regex=False), errors='coerce')
                    logging.info(f"Column '{column}' converted to numeric.")
                except Exception as e2:
                    logging.error(f"Error converting column '{column}' to numeric: {e2}")
    return df

def handle_missing_values(df, numeric_strategy='median', categorical_strategy='most_frequent', fill_value=None, log_changes=True):
    """
    Impute missing values in the DataFrame using customizable strategies.
    
    Parameters:
        df: pandas DataFrame.
        numeric_strategy: Imputation strategy for numeric columns (e.g., 'mean', 'median', 'constant').
        categorical_strategy: Imputation strategy for categorical columns (e.g., 'most_frequent', 'constant').
        fill_value: Value to use when strategy is 'constant'.
        log_changes: If True, logs the number of missing values filled per column.
    
    Returns:
        If log_changes is True: tuple (df, imputation_report)
        Otherwise: the modified DataFrame.
    """
    imputation_report = {}
    
    # Handle numeric columns
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
    
    # Handle categorical columns
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
    Outliers are not removed but clipped to lower and upper bounds.
    
    Returns:
        - The modified DataFrame.
        - A dictionary with the count of outliers capped for each numeric column.
    """
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    outliers_removed = {}
    
    for column in numeric_columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Count the number of outliers before clipping
        outlier_count = df[(df[column] < lower_bound) | (df[column] > upper_bound)].shape[0]
        
        # Clip values to be within the computed bounds
        df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
        
        if outlier_count > 0:
            outliers_removed[column] = outlier_count
            logging.info(f"Clipped {outlier_count} outliers in column '{column}'.")
            
    return df, outliers_removed

def validate_cleaning(df, original_shape, cleaning_report):
    """
    Validate the cleaning process by comparing the cleaned DataFrame with the original data.
    Reports:
        - Rows remaining.
        - Total missing values remaining.
        - Duplicates remaining.
        - Data loss percentage (if rows have been dropped).
    
    Updates and returns the cleaning report.
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
    Execute the full automated data cleaning pipeline:
        1. Compute initial data quality metrics.
        2. Standardize data types.
        3. Impute missing values with user-specified strategies.
        4. Identify and cap outliers.
        5. Validate the cleaning process.
    
    Parameters:
        df: Input DataFrame.
        numeric_imputation_strategy: Strategy for numeric imputation.
        categorical_imputation_strategy: Strategy for categorical imputation.
        fill_value: Value for constant imputation if required.
        log_imputation: Whether to log imputation details.
    
    Returns:
        A tuple containing the cleaned DataFrame and a comprehensive cleaning report.
    """
    logging.info("Starting automated cleaning pipeline.")
    original_shape = df.shape
    cleaning_report = {}

    # Step 1: Initial data quality check
    cleaning_report['initial_quality'] = check_data_quality(df)
    
    # Step 2: Standardize data types
    df = standardize_datatypes(df)
    
    # Step 3: Handle missing values with flexible imputation strategies
    df, imputation_report = handle_missing_values(df, 
                                                  numeric_strategy=numeric_imputation_strategy, 
                                                  categorical_strategy=categorical_imputation_strategy, 
                                                  fill_value=fill_value, 
                                                  log_changes=log_imputation)
    cleaning_report['imputation_report'] = imputation_report
    
    # Step 4: Remove (clip) outliers using the IQR method
    df, outliers = remove_outliers(df)
    cleaning_report['outliers_removed'] = outliers
    
    # Step 5: Validate the cleaning process
    cleaning_report = validate_cleaning(df, original_shape, cleaning_report)
    
    logging.info("Automated cleaning pipeline completed.")
    return df, cleaning_report

if __name__ == "__main__":
    # Example usage: Create a sample DataFrame to demonstrate the cleaning pipeline
    data = {
        'A': [1, 2, np.nan, 4, 100],  # Numeric column with a potential outlier (100)
        'B': ['2020-01-01', 'not a date', '2020-03-01', None, '2020-05-01'],  # Mixed date strings
        'C': ['a', 'b', None, 'b', 'a']  # Categorical column with missing value
    }
    df_sample = pd.DataFrame(data)
    
    # Run the cleaning pipeline on the sample DataFrame
    cleaned_df, report = automated_cleaning_pipeline(
        df_sample,
        numeric_imputation_strategy='median',
        categorical_imputation_strategy='most_frequent'
    )
    
    # Log the final cleaning report and display the cleaned DataFrame
    logging.info(f"Cleaning Report: {report}")
    print(cleaned_df)
