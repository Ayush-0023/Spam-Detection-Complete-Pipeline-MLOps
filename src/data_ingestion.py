import pandas as pd
import os
import logging
from sklearn.model_selection import train_test_split
import yaml

# First we need to make a logs directory, where we can save logs file. 
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True) #exist_ok=True means that this directory will be created only if it doesn't already exists

# Then we make logger object using logging module's getLogger() function
logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.DEBUG)  # If we set the level to DEBUG, it'll give all information from DEBUG to CRITICAL level

console_handler = logging.StreamHandler()   # For terminal logging
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, "data_ingestion.log") # For separate logging file
file_handler = logging.FileHandler(log_file_path, mode="w")
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(              # How the log messages will be shown
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


def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_url)
        logger.debug("Data loaded from %s", data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the CSV file: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error occurred while loading the data: %s", e)
        raise


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data."""
    try:
        df.drop(
            columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"],
            inplace=True,
        )
        df.rename(columns={"v1": "target", "v2": "text"}, inplace=True)
        logger.debug("Data preprocessing completed")
        return df
    except KeyError as e:
        logger.error("Missing column in the dataframe: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error during preprocessing: %s", e)
        raise


def save_data(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    data_path: str,
) -> None:
    """Save the train and test datasets."""
    try:
        raw_data_path = os.path.join(data_path, "raw")
        os.makedirs(raw_data_path, exist_ok=True)

        train_data.to_csv(
            os.path.join(raw_data_path, "train.csv"),
            index=False,
        )
        test_data.to_csv(
            os.path.join(raw_data_path, "test.csv"),
            index=False,
        )

        logger.debug("Train and test data saved to %s", raw_data_path)

    except Exception as e:
        logger.error("Unexpected error occurred while saving the data: %s", e)
        raise


def main():
    """
    Main function to load the dataset,
    preprocess it,
    split it into train and test sets,
    and save the datasets.
    """

    try:
        # Load parameters from params.yaml
        params = load_params("params.yaml")

        # Read the test size from params.yaml
        test_size = params["data_ingestion"]["test_size"]

        # URL of the dataset
        data_url = (
            "https://raw.githubusercontent.com/"
            "Ayush-0023/Datasets/refs/heads/main/spam.csv"
        )

        # Load the dataset
        df = load_data(data_url=data_url)

        # Perform preprocessing
        final_df = preprocess_data(df)

        # Split the dataset into training and testing sets.
        # random_state ensures that everyone gets the same split.
        train_data, test_data = train_test_split(
            final_df,
            test_size=test_size,
            random_state=42,
        )

        # Save the datasets
        save_data(
            train_data,
            test_data,
            data_path="./data",
        )

        logger.debug("Data ingestion pipeline completed successfully")

    except Exception as e:
        logger.error(
            "Failed to complete the data ingestion process: %s",
            e,
        )
        print(f"Error: {e}")

# Run the main() function only when this file is executed directly.
# If this file is imported into another Python file,
# the main() function will not execute automatically.
if __name__ == "__main__":
    main()