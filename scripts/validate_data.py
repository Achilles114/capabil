
from pathlib import Path
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "occupation_skill_dataset.csv"
REPORT_DIR = ROOT / "reports" / "eda"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\n"
            "Run python scripts/ingest_data.py first."
        )

    df = pd.read_csv(DATA_FILE)

    print("\n--- DATASET OVERVIEW ---")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Unique occupations: {df['occupation_uri'].nunique():,}")
    print(f"Unique skills: {df['skill_uri'].nunique():,}")
    print(f"Unique occupation-skill pairs: "
          f"{df[['occupation_uri', 'skill_uri']].drop_duplicates().shape[0]:,}")

    print("\n--- MISSING VALUES ---")
    print(df.isna().sum().to_string())

    print("\n--- DUPLICATE RELATIONSHIPS ---")
    print(
        "Exact duplicate rows:",
        int(df.duplicated().sum()),
    )
    print(
        "Duplicate occupation-skill pairs:",
        int(df.duplicated(["occupation_uri", "skill_uri"]).sum()),
    )

    print("\n--- RELATION TYPES ---")
    print(df["relation_type"].value_counts(dropna=False).to_string())

    print("\n--- SKILL TYPES ---")
    print(df["skill_type"].value_counts(dropna=False).to_string())

    print("\n--- TOP 15 OCCUPATIONS BY RELATION COUNT ---")
    print(
        df["occupation_name"]
        .value_counts()
        .head(15)
        .to_string()
    )

    print("\n--- TOP 15 SKILLS BY RELATION COUNT ---")
    print(
        df["skill_name"]
        .value_counts()
        .head(15)
        .to_string()
    )

    # Distribution of skills associated with each occupation.
    skills_per_occupation = df.groupby("occupation_uri")[
        "skill_uri"
    ].nunique()

    print("\n--- SKILLS PER OCCUPATION ---")
    print(skills_per_occupation.describe().round(2).to_string())

    report = {
        "rows": int(len(df)),
        "columns": df.columns.tolist(),
        "unique_occupations": int(df["occupation_uri"].nunique()),
        "unique_skills": int(df["skill_uri"].nunique()),
        "unique_occupation_skill_pairs": int(
            df[["occupation_uri", "skill_uri"]]
            .drop_duplicates()
            .shape[0]
        ),
        "missing_values": {
            k: int(v) for k, v in df.isna().sum().items()
        },
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "duplicate_occupation_skill_pairs": int(
            df.duplicated(["occupation_uri", "skill_uri"]).sum()
        ),
        "relation_type_counts": {
            str(k): int(v)
            for k, v in df["relation_type"]
            .value_counts(dropna=False).items()
        },
        "skill_type_counts": {
            str(k): int(v)
            for k, v in df["skill_type"]
            .value_counts(dropna=False).items()
        },
        "skills_per_occupation": {
            str(k): float(v)
            for k, v in skills_per_occupation.describe().items()
        },
    }

    output = REPORT_DIR / "data_quality_report.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nReport saved to: {output}")


if __name__ == "__main__":
    main()
