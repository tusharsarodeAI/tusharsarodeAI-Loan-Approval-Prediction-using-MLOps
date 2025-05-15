import pandas as pd
import numpy as np
from logger.logger_config import get_logger
import os

# Construct normalized path relative to this file
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'loan_approval_dataset.csv')
normalized_path = os.path.abspath(DATA_PATH)

print(normalized_path)

logger = get_logger('data_ingestion')

def load_data(data_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_path)
        logger.debug('Data loaded from %s', data_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

# Example usage
if __name__ == "__main__":
    df = load_data(normalized_path)
    print(df.head())
