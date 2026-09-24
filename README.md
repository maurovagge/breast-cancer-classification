# Breast Cancer Classification Pipeline

This repository contains a small, configuration-driven machine-learning pipeline for classifying breast-cancer observations as **malignant** or **benign**. The implementation is written in Python and is organized as a sequence of data loading, dataset splitting, model training, evaluation, serialization, and inference steps.

## Technology Stack

- **Python**: application and pipeline runtime.
- **pandas**: tabular data representation and CSV serialization.
- **NumPy**: generation of the sample feature vector used by the prediction script.
- **scikit-learn**:
  - `load_breast_cancer` provides the built-in Wisconsin Diagnostic Breast Cancer dataset.
  - `train_test_split` creates training and test partitions.
  - `RandomForestClassifier` trains the classification model.
  - `accuracy_score` and `classification_report` evaluate predictions.
- **PyYAML**: loads runtime and model hyperparameters from `config.yaml`.
- **Joblib**: serializes and deserializes the trained Random Forest model.

## Repository Structure

```text
.
├── config.yaml                 # Dataset paths, split settings, and model parameters
├── main.py                     # End-to-end training and evaluation entry point
├── prediction.py               # Loads the persisted model and performs inference
├── model/
│   └── random_forest.joblib    # Generated serialized model
├── data/
│   ├── raw/                    # Generated raw dataset CSV
│   └── processed/              # Reserved processed-data location
└── src/
    ├── config_loader.py        # YAML configuration loading
    ├── data_preprocessing.py   # Dataset retrieval, persistence, and splitting
    ├── model_training.py       # Random Forest construction and fitting
    ├── evaluation.py           # Accuracy and classification report generation
    └── model_io.py             # Joblib model persistence helpers
```

The `data/raw` and `data/processed` directories are kept in the repository with `.gitkeep` files, while generated data files are excluded from version control.

## Configuration

Runtime settings are centralized in `config.yaml`:

```yaml
raw_data_path: "data/raw/breast_cancer.csv"
processed_data_path: "data/processed/scaled_data.csv"
save_model_path: "model/random_forest.joblib"
test_size: 0.2
random_state: 42
n_estimators: 100
depth: 5
```

`test_size` reserves 20% of the observations for testing. `random_state` makes the split and Random Forest reproducible. The classifier is configured with 100 trees and a maximum tree depth of 5. `processed_data_path` is available as a configuration value but is not currently used by the pipeline because no scaling stage is implemented.

## Training and Evaluation Flow

Run the complete pipeline from the repository root:

```bash
python main.py
```

The execution flow is:

1. `config_loader.py` reads the YAML configuration.
2. `data_preprocessing.py` loads scikit-learn's breast-cancer dataset and converts it into a pandas `DataFrame`.
3. The dataset's 30 feature columns and `target` column are written to `data/raw/breast_cancer.csv`.
4. The `target` column is separated from the features, and `train_test_split` creates training and test partitions.
5. `model_training.py` creates and fits a `RandomForestClassifier` using the configured hyperparameters.
6. `evaluation.py` predicts the test labels and prints the accuracy and a detailed classification report.
7. `model_io.py` serializes the fitted model to `model/random_forest.joblib`, creating the destination directory when required.

The dataset target uses scikit-learn's original encoding: `0` represents **malignant** and `1` represents **benign**.

## Prediction Flow

After training has generated the model artifact, run:

```bash
python prediction.py
```

The prediction script:

1. Loads the same configuration used by the training pipeline.
2. Deserializes `model/random_forest.joblib` with Joblib.
3. Creates one 30-feature pandas `DataFrame` using randomly generated values and the official dataset feature names.
4. Calls the trained model's `predict` method.
5. Maps the predicted class to `Malignant` or `Benign` and prints the result.

The current prediction entry point is an inference demonstration: its feature vector is randomly generated and does not represent a validated clinical measurement or a user-provided patient record.

## Setup

Use a Python virtual environment, then install the runtime dependencies:

```bash
python -m venv .venv
```

Activate the environment according to your operating system and install the packages:

```bash
python -m pip install pandas numpy scikit-learn pyyaml joblib
```

No external dataset download is required; the dataset is provided by scikit-learn at runtime. The first training run creates the generated CSV and model files referenced by the configuration.

## Model Artifact

The trained estimator is stored as a Joblib file. `model_io.py` provides two focused helpers:

- `save_model(model, path)` creates the parent directory and writes the fitted estimator.
- `load_model(path)` checks that the artifact exists and then deserializes it.

The prediction script requires the artifact to exist, so `main.py` must be run before `prediction.py`.
