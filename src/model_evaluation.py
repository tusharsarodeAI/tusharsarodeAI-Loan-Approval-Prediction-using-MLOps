import pandas as pd
import os
import pickle
from sklearn.metrics import accuracy_score, classification_report
from logger.logger_config import get_logger

logger = get_logger('model_evaluation')


def load_data(file_path):
    logger.debug(f'Loading data from {file_path}')
    df = pd.read_csv(file_path)
    logger.debug(f'Data shape: {df.shape}')
    return df


def load_model(model_path):
    logger.debug(f'Loading model from {model_path}')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    logger.debug('Model loaded successfully')
    return model


def evaluate_model(model, df: pd.DataFrame, target_col='loan_status'):
    """
    Evaluate the model on the given dataset.
    """
    try:
        X = df.drop(columns=[target_col])
        y_true = df[target_col]

        y_pred = model.predict(X)
        acc = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred)

        logger.debug(f'Evaluation Accuracy: {acc:.4f}')
        logger.debug('Classification Report:\n' + report)

        print(f"Accuracy: {acc:.4f}")
        print("Classification Report:\n", report)

    except Exception as e:
        logger.error(f'Error during model evaluation: {e}')
        raise


def main():
    try:
        data_path = os.path.join('data', 'processed', 'test_processed.csv')
        model_path = os.path.join('models', 'random_forest_model.pkl')

        df = load_data(data_path)
        model = load_model(model_path)

        evaluate_model(model, df, target_col='loan_status')

    except FileNotFoundError as e:
        logger.error(f'File not found: {e}')
    except Exception as e:
        logger.error(f'Model evaluation failed: {e}')


if __name__ == '__main__':
    main()
