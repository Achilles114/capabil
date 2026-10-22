
from pathlib import Path
import json

import pandas as pd


# Project root: capabilume/
ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
INTERIM_DIR = ROOT / "data" / "interim"
PROCESSED_DIR = ROOT / "data" / "processed"

OCCUPATIONS_FILE = RAW_DIR / "occupations_en.csv"
SKILLS_FILE = RAW_DIR / "skills_en.csv"
RELATIONS_FILE = RAW_DIR / "occupationSkillRelations_en.csv"

INTERIM_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    """Fail early when an input CSV has an unexpected schema."""
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(
            f"{name} is missing columns: {sorted(missing)}\n"
            f"Available columns: {df.columns.tolist()}"
        )


def clean_text(series: pd.Series) -> pd.Series:
    """Trim whitespace while preserving missing values."""
    return series.astype("string").str.strip()


def main() -> None:
    # 1. Confirm all source files exist.
    for path in [OCCUPATIONS_FILE, SKILLS_FILE, RELATIONS_FILE]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing file: {path}\n"
                "Place the original ESCO CSV files inside data/raw/."
            )

    # 2. Load source data.
    occupations = pd.read_csv(OCCUPATIONS_FILE, dtype="string")
    skills = pd.read_csv(SKILLS_FILE, dtype="string")
    relations = pd.read_csv(RELATIONS_FILE, dtype="string")

    print("Loaded ESCO source files")
    print(f"Occupations: {len(occupations):,}")
    print(f"Skills: {len(skills):,}")
    print(f"Occupation-skill relations: {len(relations):,}")

    # 3. Validate expected schemas.
    require_columns(
        occupations,
        ["conceptUri", "preferredLabel", "description"],
        "occupations_en.csv",
    )
    require_columns(
        skills,
        ["conceptUri", "preferredLabel", "description"],
        "skills_en.csv",
    )
    require_columns(
        relations,
        [
            "occupationUri",
            "occupationLabel",
            "relationType",
            "skillType",
            "skillUri",
            "skillLabel",
        ],
        "occupationSkillRelations_en.csv",
    )

    # 4. Save source-level quality information.
    quality = {
        "source": "ESCO",
        "occupation_rows": int(len(occupations)),
        "skill_rows": int(len(skills)),
        "relation_rows": int(len(relations)),
        "occupation_duplicate_uris": int(
            occupations["conceptUri"].duplicated().sum()
        ),
        "skill_duplicate_uris": int(
            skills["conceptUri"].duplicated().sum()
        ),
        "missing_occupation_uris": int(
            relations["occupationUri"].isna().sum()
        ),
        "missing_skill_uris": int(
            relations["skillUri"].isna().sum()
        ),
    }

    # 5. Clean text fields and retain stable ESCO identifiers.
    for df, columns in [
        (
            occupations,
            ["conceptUri", "preferredLabel", "description"],
        ),
        (
            skills,
            ["conceptUri", "preferredLabel", "description"],
        ),
        (
            relations,
            [
                "occupationUri",
                "occupationLabel",
                "relationType",
                "skillType",
                "skillUri",
                "skillLabel",
            ],
        ),
    ]:
        for column in columns:
            df[column] = clean_text(df[column])

    # 6. Keep only the fields needed for the first version.
    occupation_lookup = (
        occupations[
            ["conceptUri", "preferredLabel", "description"]
        ]
        .drop_duplicates(subset=["conceptUri"])
        .rename(
            columns={
                "conceptUri": "occupation_uri",
                "preferredLabel": "occupation_name",
                "description": "occupation_description",
            }
        )
    )

    skill_lookup = (
        skills[
            ["conceptUri", "preferredLabel", "description"]
        ]
        .drop_duplicates(subset=["conceptUri"])
        .rename(
            columns={
                "conceptUri": "skill_uri",
                "preferredLabel": "skill_name",
                "description": "skill_description",
            }
        )
    )

    # 7. Normalize the relationship table.
    links = relations.rename(
        columns={
            "occupationUri": "occupation_uri",
            "occupationLabel": "relation_occupation_name",
            "relationType": "relation_type",
            "skillType": "skill_type",
            "skillUri": "skill_uri",
            "skillLabel": "relation_skill_name",
        }
    )

    # 8. Join using URIs, not display names.
    dataset = links.merge(
        occupation_lookup,
        on="occupation_uri",
        how="left",
        validate="many_to_one",
    )

    dataset = dataset.merge(
        skill_lookup,
        on="skill_uri",
        how="left",
        validate="many_to_one",
    )

    # 9. Report unresolved references before filtering anything.
    quality["unmatched_occupation_uris"] = int(
        dataset["occupation_name"].isna().sum()
    )
    quality["unmatched_skill_uris"] = int(
        dataset["skill_name"].isna().sum()
    )

    # Relationship labels are retained from ESCO for traceability.
    # The source lookup is preferred, but the relationship label is
    # a useful fallback when a lookup is missing.
    dataset["occupation_name"] = dataset[
        "occupation_name"
    ].fillna(dataset["relation_occupation_name"])

    dataset["skill_name"] = dataset[
        "skill_name"
    ].fillna(dataset["relation_skill_name"])

    # 10. Remove unusable rows and exact duplicate relationships.
    dataset = dataset.dropna(
        subset=["occupation_uri", "skill_uri"]
    )
    dataset = dataset.drop_duplicates(
        subset=["occupation_uri", "skill_uri", "relation_type"]
    )

    # 11. Select a consistent, documented schema.
    output_columns = [
        "occupation_uri",
        "occupation_name",
        "occupation_description",
        "skill_uri",
        "skill_name",
        "skill_description",
        "relation_type",
        "skill_type",
    ]

    dataset = dataset[output_columns].sort_values(
        ["occupation_name", "skill_name"]
    )

    # 12. Write reproducible intermediate and processed artifacts.
    dataset.to_csv(
        INTERIM_DIR / "occupation_skill_relationships.csv",
        index=False,
    )

    dataset.to_csv(
        PROCESSED_DIR / "occupation_skill_dataset.csv",
        index=False,
    )

    quality["processed_rows"] = int(len(dataset))
    quality["unique_occupations"] = int(
        dataset["occupation_uri"].nunique()
    )
    quality["unique_skills"] = int(dataset["skill_uri"].nunique())
    quality["unique_occupation_skill_pairs"] = int(
        dataset[["occupation_uri", "skill_uri"]]
        .drop_duplicates()
        .shape[0]
    )

    (PROCESSED_DIR / "ingestion_report.json").write_text(
        json.dumps(quality, indent=2),
        encoding="utf-8",
    )

    print("\nIngestion complete")
    print(f"Processed dataset: {PROCESSED_DIR / 'occupation_skill_dataset.csv'}")
    print(f"Quality report: {PROCESSED_DIR / 'ingestion_report.json'}")
    print(f"Processed rows: {len(dataset):,}")
    print(f"Unique occupations: {quality['unique_occupations']:,}")
    print(f"Unique skills: {quality['unique_skills']:,}")


if __name__ == "__main__":
    main()
