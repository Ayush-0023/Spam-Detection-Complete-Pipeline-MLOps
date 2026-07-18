import os
import json
import pickle
import logging
import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from dvclive import Live

# Ensure the "logs" directory exists
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)  # Creates the directory only if it doesn't already exist

# Setting up logger
logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)  # Log all messages from DEBUG level and above

# Console handler - displays logs on the terminal
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# File handler - saves logs inside a separate log file
log_file_path = os.path.join(log_dir, "model_evaluation.log")
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


def load_model(file_path: str):
    """
    Load the trained model from a pickle file.
    """

    try:
        with open(file_path, "rb") as file:
            model = pickle.load(file)

        logger.debug("Model loaded successfully from %s", file_path)
        return model

    except FileNotFoundError:
        logger.error("File not found: %s", file_path)
        raise

    except Exception as e:
        logger.error(
            "Unexpected error occurred while loading the model: %s",
            e,
        )
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
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

    except Exception as e:
        logger.error(
            "Unexpected error occurred while loading the data: %s",
            e,
        )
        raise


def evaluate_model(
    clf,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Evaluate the trained model and return the evaluation metrics.
    """

    try:
        # Predict the class labels
        y_pred = clf.predict(X_test)

        # Predict the probability of belonging to the positive class.
        # This is required for ROC-AUC calculation.
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        metrics_dict = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "auc": auc,
        }

        logger.debug("Model evaluation completed successfully")

        return metrics_dict

    except Exception as e:
        logger.error("Error during model evaluation: %s", e)
        raise


def save_metrics(metrics: dict, file_path: str) -> None:
    """
    Save the evaluation metrics to a JSON file.
    """

    try:
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Save the metrics as a JSON file
        with open(file_path, "w") as file:
            json.dump(metrics, file, indent=4)

        logger.debug("Metrics saved successfully to %s", file_path)

    except Exception as e:
        logger.error(
            "Error occurred while saving the metrics: %s",
            e,
        )
        raise


def main():
    """
    Main function to evaluate the trained model.
    """

    try:
        # Load parameters from params.yaml
        params = load_params("params.yaml")

        # Load the trained model
        clf = load_model("./models/model.pkl")

        # Load the processed testing dataset
        test_data = load_data("./data/processed/test_tfidf.csv")

        # Select all columns except the last one as input features
        X_test = test_data.iloc[:, :-1].values

        # Select the last column as the target labels
        y_test = test_data.iloc[:, -1].values

        # Evaluate the trained model
        metrics = evaluate_model(
            clf,
            X_test,
            y_test,
        )

        # Track experiments using DVCLive.
        # These metrics can later be visualized and compared across experiments.
        with Live(save_dvc_exp=True) as live:
            live.log_metric("accuracy", metrics["accuracy"])
            live.log_metric("precision", metrics["precision"])
            live.log_metric("recall", metrics["recall"])
            live.log_metric("auc", metrics["auc"])

            # Log all parameters used during the experiment
            live.log_params(params)

        # Save the evaluation metrics
        save_metrics(
            metrics,
            os.path.join("reports", "metrics.json"),
        )

        logger.debug("Model evaluation pipeline completed successfully")

    except Exception as e:
        logger.error(
            "Failed to complete the model evaluation process: %s",
            e,
        )
        print(f"Error: {e}")


# Run the main() function only when this file is executed directly.
# If this file is imported into another Python file,
# the main() function will not execute automatically.
if __name__ == "__main__":
    main()