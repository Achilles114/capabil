
import hashlib
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from capabilume.mlops.tracking import (
    configure_tracking,
    log_training_run,
)

SPLITS = ROOT / "data" / "splits"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports" / "evaluation"

TARGET = "relation_type"
GROUP = "occupation_uri"
RANDOM_STATE = 42
LABELS = ["essential", "optional"]


def sha256_file(path):
    """Fingerprint a file to identify the exact dataset version."""
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split(name):
    path = SPLITS / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run scripts/prepare_data.py first."
        )
    return pd.read_csv(path).fillna("")


def make_text(df):
    return (
        "occupation: " + df["occupation_name"].astype(str)
        + " " + df["occupation_description"].astype(str)
        + " skill: " + df["skill_name"].astype(str)
        + " " + df["skill_description"].astype(str)
    )


def make_features(df):
    return pd.DataFrame({
        "text": make_text(df),
        "skill_type": df["skill_type"].fillna("unknown").astype(str),
    })


def calculate_metrics(y_true, predictions):
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, predictions)
        ),
        "macro_f1": float(
            f1_score(y_true, predictions, average="macro")
        ),
        "classification_report": classification_report(
            y_true,
            predictions,
            labels=LABELS,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_true, predictions, labels=LABELS
        ).tolist(),
    }


def main():
    configure_tracking(
        project_root=ROOT,
        config_path=ROOT / "configs" / "training.yaml",
    )

    for directory in [SPLITS, MODELS, REPORTS]:
        if not directory.exists():
            raise FileNotFoundError(
                f"Required directory not found: {directory}"
            )

    split_paths = {
        name: SPLITS / f"{name}.csv"
        for name in ["train", "validation", "test"]
    }

    dataset_hashes = {
        name: sha256_file(path)
        for name, path in split_paths.items()
    }

    train = load_split("train")
    validation = load_split("validation")
    test = load_split("test")

    required_columns = {
        "occupation_uri",
        "occupation_name",
        "occupation_description",
        "skill_uri",
        "skill_name",
        "skill_description",
        "skill_type",
        TARGET,
    }

    for name, frame in [
        ("train", train),
        ("validation", validation),
        ("test", test),
    ]:
        missing = required_columns - set(frame.columns)
        if missing:
            raise ValueError(
                f"{name} is missing columns: {sorted(missing)}"
            )

        if frame[list(required_columns)].isna().any().any():
            raise ValueError(f"Unexpected missing values in {name}.")

        if not set(frame[TARGET].unique()).issubset(LABELS):
            raise ValueError(f"Unexpected labels in {name}.")

    # Prevent occupation leakage across splits.
    train_groups = set(train[GROUP])
    val_groups = set(validation[GROUP])
    test_groups = set(test[GROUP])

    if (
        not train_groups.isdisjoint(val_groups)
        or not train_groups.isdisjoint(test_groups)
        or not val_groups.isdisjoint(test_groups)
    ):
        raise ValueError("Occupation leakage detected between splits.")

    X_train = make_features(train)
    X_val = make_features(validation)
    X_test = make_features(test)

    y_train = train[TARGET]
    y_val = validation[TARGET]
    y_test = test[TARGET]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "text",
                TfidfVectorizer(
                    max_features=100_000,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                ),
                "text",
            ),
            (
                "skill_type",
                OneHotEncoder(handle_unknown="ignore"),
                ["skill_type"],
            ),
        ]
    )

    model = Pipeline([
        ("features", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        )),
    ])

    split_metadata = {
        name: {
            "rows": len(frame),
            "occupations": int(frame[GROUP].nunique()),
            "label_counts": {
                str(key): int(value)
                for key, value in frame[TARGET].value_counts().items()
            },
            "sha256": dataset_hashes[name],
        }
        for name, frame in [
            ("train", train),
            ("validation", validation),
            ("test", test),
        ]
    }

    print("Training baseline model...")
    model.fit(X_train, y_train)

    print("Evaluating validation set...")
    val_metrics = calculate_metrics(y_val, model.predict(X_val))

    print("Evaluating test set...")
    test_metrics = calculate_metrics(y_test, model.predict(X_test))

    majority_baseline = float(
        y_train.value_counts(normalize=True).max()
    )

    results = {
        "model": "TF-IDF + Logistic Regression",
        "target": TARGET,
        "label_order": LABELS,
        "split_strategy": "unseen occupations",
        "majority_class_train_accuracy": majority_baseline,
        "dataset_hashes": dataset_hashes,
        "splits": split_metadata,
        "validation": val_metrics,
        "test": test_metrics,
    }

    model_path = MODELS / "esco_relation_baseline.joblib"
    metrics_path = REPORTS / "baseline_metrics.json"
    metadata_path = REPORTS / "dataset_split_metadata.json"

    joblib.dump(model, model_path)

    metrics_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    metadata_path.write_text(
        json.dumps(split_metadata, indent=2),
        encoding="utf-8",
    )

    params = {
        "model_type": "LogisticRegression",
        "vectorizer": "TfidfVectorizer",
        "max_features": 100_000,
        "ngram_range": "(1, 2)",
        "min_df": 2,
        "sublinear_tf": True,
        "max_iter": 1000,
        "random_state": RANDOM_STATE,
        "split_strategy": "GroupShuffleSplit by occupation_uri",
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
    }
    params.update({
        f"sha256_{name}": digest
        for name, digest in dataset_hashes.items()
    })

    metrics = {
        "validation_accuracy": val_metrics["accuracy"],
        "validation_balanced_accuracy": val_metrics["balanced_accuracy"],
        "validation_macro_f1": val_metrics["macro_f1"],
        "test_accuracy": test_metrics["accuracy"],
        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "train_majority_class_accuracy": majority_baseline,
    }

    run_id = log_training_run(
        run_name="tfidf-logistic-regression",
        params=params,
        metrics=metrics,
        artifacts={
            "evaluation": metrics_path,
            "data-metadata": metadata_path,
            "model-files": model_path,
        },
        model=model,
        model_name="esco-relation-classifier",
        input_example=X_train.head(3),
    )

    print(f"\nMLflow run ID: {run_id}")
    print(f"Training majority-class accuracy: {majority_baseline:.4f}")

    for name, metric_set in [
        ("VALIDATION", val_metrics),
        ("TEST", test_metrics),
    ]:
        print(f"\n{name} METRICS")
        print(json.dumps({
            key: metric_set[key]
            for key in [
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "confusion_matrix",
            ]
        }, indent=2))
        print("\nPer-class report:")
        print(
            pd.DataFrame(metric_set["classification_report"])
            .T.round(3)
        )

    print(f"\nModel: {model_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Dataset metadata: {metadata_path}")


if __name__ == "__main__":
    main()
