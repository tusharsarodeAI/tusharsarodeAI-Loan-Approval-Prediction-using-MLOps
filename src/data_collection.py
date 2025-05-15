import pandas as pd
import numpy as np
from logger.logger_config import get_logger
import os
from sklearn.model_selection import train_test_split

# Constants
DATA_URL = "https://raw.githubusercontent.com/tusharsarodeAI/tusharsarodeAI-Loan-Approval-Prediction-using-MLOps/refs/heads/master/data/loan_approval_dataset.csv"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

# Logger
logger = get_logger('data_ingestion')


def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_url)
        logger.debug('Data loaded from %s', data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data by dropping unnecessary columns."""
    try:
        df.drop(columns="loan_id", inplace=True)
        logger.debug('loan_id column dropped')
        return df
    except KeyError as e:
        logger.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s', e)
        raise


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, save_dir: str) -> None:
    """Save the train and test datasets."""
    try:
        os.makedirs(save_dir, exist_ok=True)
        train_data.to_csv(os.path.join(save_dir, "train.csv"), index=False)
        test_data.to_csv(os.path.join(save_dir, "test.csv"), index=False)
        logger.debug('Train and test data saved to %s', save_dir)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise


# Main execution
if __name__ == "__main__":
    df = load_data(DATA_URL)
    final_df = drop_columns(df)
    train_data, test_data = train_test_split(final_df, test_size=0.2, random_state=2)
    save_data(train_data, test_data, LOCAL_DATA_DIR)
