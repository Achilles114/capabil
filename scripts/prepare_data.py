
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "occupation_skill_dataset.csv"
OUTPUT = ROOT / "data" / "splits"

TARGET = "relation_type"
GROUP = "occupation_uri"
RANDOM_STATE = 42

REQUIRED_COLUMNS = [
    "occupation_uri",
    "occupation_name",
    "occupation_description",
    "skill_uri",
    "skill_name",
    "skill_description",
    "relation_type",
    "skill_type",
]


def main():
    df = pd.read_csv(INPUT)

    # Validate schema and essential fields.
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    essential_fields = [
        "occupation_uri",
        "occupation_name",
        "skill_uri",
        "skill_name",
        TARGET,
    ]
    if df[essential_fields].isna().any().any():
        raise ValueError("Missing values found in essential fields.")

    if not set(df[TARGET].unique()).issubset({"essential", "optional"}):
        raise ValueError("Unexpected target labels found.")

    if df.duplicated(["occupation_uri", "skill_uri"]).any():
        raise ValueError("Duplicate occupation-skill pairs found.")

    # Treat missing skill types as an explicit category.
    df["skill_type"] = df["skill_type"].fillna("unknown")

    # Split into 70% train, 15% validation, 15% test,
    # keeping every occupation in only one split.
    first_split = GroupShuffleSplit(
        n_splits=1, test_size=0.30, random_state=RANDOM_STATE
    )
    train_idx, remaining_idx = next(
        first_split.split(df, groups=df[GROUP])
    )

    train = df.iloc[train_idx].copy()
    remaining = df.iloc[remaining_idx].copy()

    second_split = GroupShuffleSplit(
        n_splits=1, test_size=0.50, random_state=RANDOM_STATE
    )
    val_idx, test_idx = next(
        second_split.split(remaining, groups=remaining[GROUP])
    )

    validation = remaining.iloc[val_idx].copy()
    test = remaining.iloc[test_idx].copy()

    # Confirm no occupation appears in more than one split.
    train_groups = set(train[GROUP])
    val_groups = set(validation[GROUP])
    test_groups = set(test[GROUP])

    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)

    OUTPUT.mkdir(parents=True, exist_ok=True)

    train.to_csv(OUTPUT / "train.csv", index=False)
    validation.to_csv(OUTPUT / "validation.csv", index=False)
    test.to_csv(OUTPUT / "test.csv", index=False)

    report = {
        "input_rows": len(df),
        "missing_skill_type_filled": int(
            df["skill_type"].eq("unknown").sum()
        ),
        "splits": {},
        "random_state": RANDOM_STATE,
        "group_column": GROUP,
        "target_column": TARGET,
        "split_method": "GroupShuffleSplit by occupation_uri",
    }

    for name, part in [
        ("train", train),
        ("validation", validation),
        ("test", test),
    ]:
        report["splits"][name] = {
            "rows": len(part),
            "occupations": int(part[GROUP].nunique()),
            "label_counts": {
                str(k): int(v)
                for k, v in part[TARGET].value_counts().items()
            },
        }

    with open(OUTPUT / "split_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nSaved splits to: {OUTPUT}")


if __name__ == "__main__":
    main()
