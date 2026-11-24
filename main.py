"""Educational Titanic passenger survival classifier."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.datasets import fetch_openml


NUMERIC_FEATURES = ["pclass", "age", "sibsp", "parch", "fare"]
CATEGORICAL_FEATURES = ["sex", "embarked"]


def main():
    # OpenML downloads and caches this public dataset on the first run.
    dataset = fetch_openml(data_id=40945, as_frame=True)
    features = dataset.data.copy()
    target = dataset.target.squeeze()

    # Keep only information that would be known before the voyage.
    selected_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    features = features[selected_features]

    numeric_columns = [
        column for column in NUMERIC_FEATURES if column in features.columns
    ]
    categorical_columns = [
        column for column in CATEGORICAL_FEATURES if column in features.columns
    ]

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    StandardScaler(),
                ),
                numeric_columns,
            ),
            (
                "categorical",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder(handle_unknown="ignore"),
                ),
                categorical_columns,
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    model = make_pipeline(
        preprocessing,
        LogisticRegression(max_iter=1000, random_state=42),
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print("Titanic Survival Prediction")
    print(f"Passengers used: {len(features)}")
    print(f"Passenger details used: {len(features.columns)}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")
    print(f"Weighted F1 score: {f1_score(y_test, predictions, average='weighted'):.2f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix (class order shown below):")
    print(list(model.classes_))
    print(confusion_matrix(y_test, predictions, labels=model.classes_))

    example = features.iloc[[0]]
    prediction = model.predict(example)[0]
    print(f"\nExample passenger prediction: {prediction}")
    print("This model is for learning and does not explain or judge individual outcomes.")


if __name__ == "__main__":
    main()
