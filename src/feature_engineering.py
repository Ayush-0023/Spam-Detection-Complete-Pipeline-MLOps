import os
import logging
import yaml
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure the "logs" directory exists
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)  # Creates the directory only if it doesn't already exist

# Setting up logger
logger = logging.getLogger("feature_engineering")
logger.setLevel(logging.DEBUG)  # Log all messages from DEBUG level and above

# Console handler - displays logs on the terminal
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# File handler - saves logs inside a separate log file
log_file_path = os.path.join(log_dir, "feature_engineering.log")
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
    """

    try:
        df = pd.read_csv(file_path)

        # Replace missing values with empty strings
        # so that TF-IDF doesn't throw an error.
        df.fillna("", inplace=True)

        logger.debug("Data loaded successfully from %s", file_path)
        return df

    except pd.errors.ParserError as e:
        logger.error("Failed to parse the CSV file: %s", e)
        raise

    except Exception as e:
        logger.error("Unexpected error occurred while loading the data: %s", e)
        raise


def apply_tfidf(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    max_features: int,
) -> tuple:
    """
    Apply TF-IDF vectorization on the training and testing datasets.
    """

    try:
        vectorizer = TfidfVectorizer(max_features=max_features)

        X_train = train_data["text"].values
        y_train = train_data["target"].values

        X_test = test_data["text"].values
        y_test = test_data["target"].values

        # Fit ONLY on training data
        X_train_tfidf = vectorizer.fit_transform(X_train)

        # Transform test data using the vocabulary learned from training data
        X_test_tfidf = vectorizer.transform(X_test)

        train_df = pd.DataFrame(
            X_train_tfidf.toarray()
        )
        train_df["target"] = y_train

        test_df = pd.DataFrame(
            X_test_tfidf.toarray()
        )
        test_df["target"] = y_test

        logger.debug(
            "TF-IDF vectorization completed successfully with %d features",
            max_features,
        )

        return train_df, test_df, vectorizer

    except Exception as e:
        logger.error("Error during TF-IDF transformation: %s", e)
        raise


def save_data(df: pd.DataFrame, file_path: str) -> None:
    """
    Save the DataFrame to a CSV file.
    """

    try:
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Save the DataFrame
        df.to_csv(file_path, index=False)

        logger.debug("Data saved to %s", file_path)

    except Exception as e:
        logger.error("Unexpected error occurred while saving the data: %s", e)
        raise


def main():
    """
    Main function to perform feature engineering.
    """

    try:
        params = load_params("params.yaml")

        max_features = params["feature_engineering"]["max_features"]

        train_data = load_data(
            "./data/interim/train_processed.csv"
        )

        test_data = load_data(
            "./data/interim/test_processed.csv"
        )

        train_df, test_df, vectorizer = apply_tfidf(
            train_data,
            test_data,
            max_features,
        )

        save_data(
            train_df,
            os.path.join(
                "data",
                "processed",
                "train_tfidf.csv",
            ),
        )

        save_data(
            test_df,
            os.path.join(
                "data",
                "processed",
                "test_tfidf.csv",
            ),
        )

        # Save the fitted vectorizer
        vectorizer_path = os.path.join(
            "models",
            "vectorizer.pkl",
        )

        os.makedirs("models", exist_ok=True)

        with open(vectorizer_path, "wb") as file:
            import pickle
            pickle.dump(vectorizer, file)

        logger.debug(
            "Vectorizer saved successfully to %s",
            vectorizer_path,
        )

        logger.debug(
            "Feature engineering process completed successfully"
        )

    except Exception as e:
        logger.error(
            "Failed to complete the feature engineering process: %s",
            e,
        )
        print(f"Error: {e}")

# Run the main() function only when this file is executed directly.
# If this file is imported into another Python file,
# the main() function will not execute automatically.
if __name__ == "__main__":
    main()