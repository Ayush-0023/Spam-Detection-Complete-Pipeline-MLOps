# SMS Spam Classification - Complete MLOps Pipeline

An end-to-end **SMS Spam Classification** project that combines machine learning, reproducible data pipelines, experiment tracking, and a Streamlit web application.

The project uses **TF-IDF feature extraction** and **Multinomial Naive Bayes** to classify SMS messages as **Spam** or **Not Spam**. The complete ML workflow is managed using **DVC**, while **DVCLive** is used for experiment tracking and model evaluation.

---

## Project Overview

The goal of this project is to build a reproducible machine learning pipeline for detecting spam SMS messages.

The project covers the complete ML lifecycle:

```text
Raw Dataset
     ↓
Data Ingestion
     ↓
Data Preprocessing
     ↓
TF-IDF Feature Engineering
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Trained Model + Vectorizer
     ↓
Streamlit Application
     ↓
Spam / Not Spam Prediction
```

---

## Tech Stack

* **Python**
* **Pandas** - data manipulation
* **NumPy** - numerical operations
* **NLTK** - text preprocessing
* **Scikit-learn** - TF-IDF and machine learning
* **Multinomial Naive Bayes** - final classification model
* **DVC** - data and pipeline versioning
* **DVCLive** - experiment tracking
* **PyYAML** - configuration management
* **Streamlit** - web application
* **Git/GitHub** - source code version control

---

## Dataset

The project uses the **SMS Spam Collection** dataset containing SMS messages labeled as:

* `ham` - legitimate message
* `spam` - unwanted/spam message

The target labels are encoded as:

```text
ham  → 0
spam → 1
```

The dataset is downloaded automatically during the data ingestion stage.

---

## Data Preprocessing

The text preprocessing pipeline performs the following operations:

1. Convert text to lowercase
2. Tokenize the text using NLTK
3. Remove punctuation and non-alphanumeric tokens
4. Remove English stopwords
5. Apply Porter stemming
6. Join the processed tokens back into a string

Example:

```text
Original:
"Congratulations! You have won a free prize."

Processed:
"congratul free prize"
```

Duplicate records are also removed during preprocessing.

---

## Feature Engineering

Text is converted into numerical features using **TF-IDF (Term Frequency–Inverse Document Frequency)**.

The final configuration uses:

```yaml
feature_engineering:
  max_features: 3000
```

The TF-IDF vectorizer is fitted only on the training data and then used to transform the test data.

The fitted vectorizer is saved as:

```text
models/vectorizer.pkl
```

This ensures that the same vocabulary and transformation process used during training is used during inference.

---

## Model

Several classification algorithms were explored during the experimentation phase, including:

* Support Vector Classifier
* K-Nearest Neighbors
* Multinomial Naive Bayes
* Decision Tree
* Logistic Regression
* Random Forest
* AdaBoost
* Bagging
* Extra Trees
* Gradient Boosting
* XGBoost

Based on the experimentation results, **Multinomial Naive Bayes** was selected as the final model.

Current configuration:

```yaml
model_building:
  alpha: 1.0
```

The trained model is saved as:

```text
models/model.pkl
```

---

## Model Performance

The current pipeline achieved the following results on the test dataset:

| Metric    |       Score |
| --------- | ----------: |
| Accuracy  |  **97.07%** |
| Precision | **100.00%** |
| Recall    |  **78.23%** |
| ROC-AUC   |  **98.16%** |

The particularly high precision means that messages predicted as spam are highly reliable, while the lower recall indicates that some spam messages are still classified as legitimate.

Metrics are stored in:

```text
reports/metrics.json
```

---

## MLOps Pipeline

DVC manages the complete machine learning pipeline.

The pipeline consists of five stages:

```text
data_ingestion
      ↓
data_preprocessing
      ↓
feature_engineering
      ↓
model_building
      ↓
model_evaluation
```

The pipeline is defined in:

```text
dvc.yaml
```

The exact state of the pipeline and its dependencies is tracked in:

```text
dvc.lock
```

### Run the complete pipeline

```bash
dvc repro
```

DVC automatically determines which stages need to be rerun based on changes to code, parameters, and dependencies.

---

## Configuration

Model and pipeline parameters are stored separately in:

```text
params.yaml
```

Current configuration:

```yaml
data_ingestion:
  test_size: 0.2

feature_engineering:
  max_features: 3000

model_building:
  alpha: 1.0
```

This makes it possible to change experiment parameters without modifying the pipeline code.

---

## Experiment Tracking

**DVCLive** is used to track model evaluation metrics.

The evaluation stage records:

* Accuracy
* Precision
* Recall
* ROC-AUC

Experiments can be run with:

```bash
dvc exp run
```

and compared using DVC's experiment functionality.

---

## Streamlit Application

The project includes a Streamlit application for real-time SMS classification.

Run the application with:

```bash
streamlit run app.py
```

The application:

1. Accepts an SMS message from the user
2. Applies the same text preprocessing used during training
3. Transforms the message using the trained TF-IDF vectorizer
4. Passes the resulting features to the trained Multinomial Naive Bayes model
5. Displays either **Spam** or **Not Spam**

The application loads the artifacts generated by the DVC pipeline:

```text
models/vectorizer.pkl
models/model.pkl
```

---

## Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Spam-Detection-Complete-Pipeline-MLOps
```

Create and activate a virtual environment:

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the ML Pipeline

Run the complete pipeline:

```bash
dvc repro
```

---

## Run the Streamlit App

After the pipeline has completed:

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

---

## Generated Artifacts

The pipeline generates the following important artifacts:

```text
models/model.pkl
models/vectorizer.pkl
reports/metrics.json
```

These artifacts allow the trained model to be used directly by the Streamlit application.

---

## Experimentation

The initial model experimentation was performed in:

```text
experiments/spam-classification.ipynb
```

The notebook was used to compare multiple classification algorithms and investigate different TF-IDF configurations.

The final production pipeline was then implemented as modular Python scripts under:

```text
src/
```

This separates experimentation from the reproducible production pipeline.

---

## Reproducibility

One of the main goals of this project is reproducibility.

The combination of:

* Git
* DVC
* `dvc.yaml`
* `dvc.lock`
* `params.yaml`
* DVCLive
* Versioned source code

allows the complete ML workflow to be reproduced from the project repository.

Running:

```bash
dvc repro
```

rebuilds only the pipeline stages affected by changes.

---

## Author

**Ayush**