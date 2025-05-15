import pandas as pd
import numpy as np
from logger.logger_config import get_logger
from sklearn.preprocessing import LabelEncoder
import os

logger = get_logger('data_preprocessing')


def clean_data(st):
    if isinstance(st, str):
        return st.strip()
    return st

def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the DataFrame by combining columns, cleaning text,
    handling missing values, and label encoding categorical variables.
    """
    try:
        logger.debug('Starting preprocessing for DataFrame')

        # Strip whitespace from column names
        df.columns = df.columns.str.strip()

        # Combine asset columns
        df['Assets'] = (
            df['residential_assets_value'] +
            df['commercial_assets_value'] +
            df['luxury_assets_value'] +
            df['bank_asset_value']
        )

        # Drop individual asset columns
        df.drop(columns=[
            'residential_assets_value',
            'commercial_assets_value',
            'luxury_assets_value',
            'bank_asset_value'
        ], inplace=True)

        # Log null values
        nullvalues = df.isnull().sum()
        logger.debug(f'Null values:\n{nullvalues}')

        # Clean categorical text
        for col in ['education', 'self_employed', 'loan_status']:
            df[col] = df[col].apply(clean_data)

        # Label encode categorical columns
        label_enc = LabelEncoder()
        categorical_cols = df.select_dtypes(include=['object']).columns

        for col in categorical_cols:
            df[col] = label_enc.fit_transform(df[col].astype(str))
            logger.debug(f'Label encoded column: {col}')

        logger.debug('Preprocessing complete')
        return df

    except KeyError as e:
        logger.error('Column not found: %s', e)
        raise
    except Exception as e:
        logger.error('Error during preprocessing: %s', e)
        raise


def main():
    """
    Main function to load raw data, preprocess it, and save the processed data.
    """
    try:
        # Fetch the data from data/raw
        train_data = pd.read_csv('../data/raw/train.csv')
        test_data = pd.read_csv('../data/raw/test.csv')
        logger.debug('Data loaded properly')

        # Transform the data
        train_processed_data = preprocess_df(train_data)
        test_processed_data = preprocess_df(test_data)

        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
        os.makedirs(data_path, exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)

        logger.debug('Processed data saved to %s', data_path)
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
    except pd.errors.EmptyDataError as e:
        logger.error('No data: %s', e)
    except Exception as e:
        logger.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()