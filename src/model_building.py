import os
import pickle
import logging
import yaml
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Ensure the "logs" directory exists
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)  # Creates the directory only if it doesn't already exist

# Setting up logger
logger = logging.getLogger("model_building")
logger.setLevel(logging.DEBUG)  # Log all messages from DEBUG level and above

# Console handler - displays logs on the terminal
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# File handler - saves logs inside a separate log file
log_file_path = os.path.join(log_dir, "model_building.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

# Define how the log messages will appear
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Attach the handlers to the logger.
# The if-condition prevents duplicate log messages if this file is imported elsewhere.
if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    """
    Load parameters from a YAML file.
    """

    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)

        logger.debug("Parameters retrieved from %s", params_path)
        return params

    except FileNotFoundError:
        logger.error("File not found: %s", params_path)
        raise

    except yaml.YAMLError as e:
        logger.error("YAML parsing error: %s", e)
        raise

    except Exception as e:
        logger.error("Unexpected error while loading parameters: %s", e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.

    :param file_path: Path to the CSV file.
    :return: Loaded DataFrame.
    """

    try:
        df = pd.read_csv(file_path)

        logger.debug(
            "Data loaded successfully from %s with shape %s",
            file_path,
            df.shape,
        )

        return df

    except pd.errors.ParserError as e:
        logger.error("Failed to parse the CSV file: %s", e)
        raise

    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        raise

    except Exception as e:
        logger.error(
            "Unexpected error occurred while loading the data: %s",
            e,
        )
        raise


def train_model(            # Using type-hinting style
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: dict,
) -> RandomForestClassifier:
    """
    Train the Random Forest model.

    :param X_train: Training features.
    :param y_train: Training labels.
    :param params: Dictionary containing model hyperparameters.
    :return: Trained RandomForestClassifier.
    """

    try:
        # Ensure both X_train and y_train contain the same
        # number of training samples.
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(
                "The number of samples in X_train and y_train must be the same."
            )

        logger.debug(
            "Initializing Random Forest model with parameters: %s",
            params,
        )

        # Initialize the Random Forest model using
        # the hyperparameters defined in params.yaml.
        clf = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            random_state=params["random_state"],
        )

        logger.debug(
            "Model training started with %d samples",
            X_train.shape[0],
        )

        # Train the model
        clf.fit(X_train, y_train)

        logger.debug("Model training completed successfully")

        return clf

    except ValueError as e:
        logger.error("ValueError during model training: %s", e)
        raise

    except Exception as e:
        logger.error("Error during model training: %s", e)
        raise


def save_model(model, file_path: str) -> None:
    """
    Save the trained model as a pickle (.pkl) file.

    :param model: Trained machine learning model.
    :param file_path: Path where the model should be saved.
    """

    try:
        # Create the directory if it doesn't exist.
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Save the trained model.
        with open(file_path, "wb") as file:
            pickle.dump(model, file)

        logger.debug("Model saved successfully to %s", file_path)

    except FileNotFoundError as e:
        logger.error("File path not found: %s", e)
        raise

    except Exception as e:
        logger.error("Error occurred while saving the model: %s", e)
        raise


def main():
    """
    Main function to train and save the machine learning model.
    """

    try:
        # Load the model hyperparameters from params.yaml
        params = load_params("params.yaml")["model_building"]

        # Load the processed training dataset
        train_data = load_data("./data/processed/train_tfidf.csv")

        # Select all columns except the last one as input features
        X_train = train_data.iloc[:, :-1].values

        # Select the last column as the target labels
        y_train = train_data.iloc[:, -1].values

        # Train the Random Forest model
        clf = train_model(
            X_train,
            y_train,
            params,
        )

        # Save the trained model inside the models directory
        model_save_path = os.path.join("models", "model.pkl")
        save_model(clf, model_save_path)

        logger.debug("Model training pipeline completed successfully")

    except Exception as e:
        logger.error(
            "Failed to complete the model building process: %s",
            e,
        )
        print(f"Error: {e}")


# Run the main() function only when this file is executed directly.
# If this file is imported into another Python file,
# the main() function will not execute automatically.
if __name__ == "__main__":
    main()