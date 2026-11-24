# Titanic Survival Prediction

A beginner machine learning project that predicts passenger survival labels in the public Titanic dataset. It uses passenger class, age, sex, family counts, fare, and embarkation port as model inputs.

## Internship ID

Add your own internship ID here before submitting this repository:

`CIS134`

## How it works

- Downloads the Titanic dataset from OpenML the first time the script runs.
- Uses only passenger details available before the voyage; it leaves out columns that directly reveal survival, such as lifeboat or body-recovery information.
- Fills missing numeric and categorical values, then scales numeric features and one-hot encodes categories.
- Trains a logistic regression classifier with a stratified train/test split.
- Prints accuracy, weighted F1 score, a classification report, a confusion matrix, and one example prediction.

## Run it

Requires Python 3.10 or newer and an internet connection for the first dataset download.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

On Windows, activate the environment with `.venv\\Scripts\\activate` instead.

## Dataset source

The project uses OpenML's Titanic dataset (data ID 40945), which includes passenger survival status and passenger information. [OpenML dataset page](https://www.openml.org/d/40945).

The project is an educational classification exercise. It does not make judgments about individuals or explain the historical causes of survival.

## Tools

Python, pandas, scikit-learn, OpenML, and logistic regression.
