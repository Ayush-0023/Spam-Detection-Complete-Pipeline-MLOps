import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk

# Download the required NLTK resources only if they are not already present
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# Ensure the "logs" directory exists
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)  # Creates the directory only if it doesn't already exist

# Setting up logger
logger = logging.getLogger("data_preprocessing")
logger.setLevel(logging.DEBUG)  # Log all messages from DEBUG level and above

# Console handler - displays logs on the terminal
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# File handler - saves logs inside a separate log file
log_file_path = os.path.join(log_dir, "data_preprocessing.log")
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

# Create these objects only once instead of creating them for every text message.
# This improves performance.
ps = PorterStemmer()

# Convert the list of stopwords into a set for faster lookup.
STOP_WORDS = set(stopwords.words("english"))


def transform_text(text):
    """
    Transforms the input text by converting it to lowercase,
    tokenizing, removing stopwords, and stemming.
    """

    # Convert text to lowercase
    text = text.lower()

    # Tokenize the text into individual words
    text = nltk.word_tokenize(text)

    # Remove punctuation/non-alphanumeric words and stopwords
    # word.isalnum() removes punctuation such as !, ?, ., etc.
    text = [
        word
        for word in text
        if word.isalnum() and word not in STOP_WORDS
    ]

    # Apply stemming to each word
    text = [ps.stem(word) for word in text]

    # Join all words back into a single string
    return " ".join(text)


def preprocess_df(df, text_column="text", target_column="target"):
    """
    Preprocesses the DataFrame by encoding the target column,
    removing duplicates, and transforming the text column.
    """

    try:
        logger.debug("Starting preprocessing for DataFrame")

        # Encode the target column.
        # Example:
        # ham  -> 0
        # spam -> 1
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.debug("Target column encoded")

        # Remove duplicate rows while keeping the first occurrence
        df = df.drop_duplicates(keep="first")
        logger.debug("Duplicates removed")

        # Apply text preprocessing on the text column
        df.loc[:, text_column] = df[text_column].apply(transform_text)
        logger.debug("Text column transformed")

        return df

    except KeyError as e:
        logger.error("Column not found: %s", e)
        raise

    except Exception as e:
        logger.error("Error during text normalization: %s", e)
        raise


def main(text_column="text", target_column="target"):
    """
    Main function to load raw data,
    preprocess it,
    and save the processed data.
    """

    try:
        # Load the raw train and test datasets
        train_data = pd.read_csv("./data/raw/train.csv")
        test_data = pd.read_csv("./data/raw/test.csv")
        logger.debug("Training and testing data loaded successfully")

        # Preprocess both datasets
        train_processed_data = preprocess_df(
            train_data,
            text_column,
            target_column,
        )

        test_processed_data = preprocess_df(
            test_data,
            text_column,
            target_column,
        )

        # Store the processed datasets inside data/interim
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)

        train_processed_data.to_csv(
            os.path.join(data_path, "train_processed.csv"),
            index=False,
        )

        test_processed_data.to_csv(
            os.path.join(data_path, "test_processed.csv"),
            index=False,
        )

        logger.debug("Processed data saved to %s", data_path)

    except FileNotFoundError as e:
        logger.error("File not found: %s", e)

    except pd.errors.EmptyDataError as e:
        logger.error("Dataset is empty: %s", e)

    except Exception as e:
        logger.error(
            "Failed to complete the data preprocessing process: %s",
            e,
        )
        print(f"Error: {e}")


# Run the main() function only when this file is executed directly.
# If this file is imported into another Python file,
# the main() function will not execute automatically.
if __name__ == "__main__":
    main()