import pandas as pd
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from logger.logger_config import get_logger

logger = get_logger('model_training')


def load_data(file_path):
    logger.debug(f'Loading data from {file_path}')
    df = pd.read_csv(file_path)
    logger.debug(f'Data shape: {df.shape}')
    return df


def train_model(df: pd.DataFrame, target_col='loan_status'):
    """
    Train a Random Forest model on the processed data.
    """
    try:
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Split train/validation for quick evaluation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        logger.debug(f'Train shape: {X_train.shape}, Validation shape: {X_val.shape}')

        # Initialize and train the model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        logger.debug('Random Forest model training complete')

        # # Validate the model
        # y_pred = model.predict(X_val)
        # acc = accuracy_score(y_val, y_pred)
        # logger.debug(f'Validation Accuracy: {acc:.4f}')
        # logger.debug('Classification Report:\n' + classification_report(y_val, y_pred))

        return model

    except Exception as e:
        logger.error(f'Error during model training: {e}')
        raise


def save_model(model, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
    logger.debug(f'Model saved to {output_path}')


def main():
    try:
        data_path = os.path.join('data', 'processed', 'train_processed.csv')
        model_path = os.path.join('models', 'random_forest_model.pkl')

        df = load_data(data_path)
        model = train_model(df, target_col='loan_status')

        save_model(model, model_path)

    except FileNotFoundError as e:
        logger.error(f'File not found: {e}')
    except Exception as e:
        logger.error(f'Model training failed: {e}')


if __name__ == '__main__':
    main()
