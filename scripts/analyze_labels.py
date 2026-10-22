
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "occupation_skill_dataset.csv"

df = pd.read_csv(DATA)

print("RELATION TYPE COUNTS")
print(df["relation_type"].value_counts())

print("\nRELATION TYPE PROPORTIONS")
print(df["relation_type"].value_counts(normalize=True).round(4))

print("\nRELATION TYPE BY SKILL TYPE")
print(pd.crosstab(
    df["skill_type"],
    df["relation_type"],
    margins=True
))

print("\nOCCUPATIONS WITH MOST ESSENTIAL RELATIONSHIPS")
essential = df[df["relation_type"] == "essential"]
print(essential["occupation_name"].value_counts().head(10))

print("\nSAMPLE RELATIONSHIPS")
print(
    df[[
        "occupation_name",
        "skill_name",
        "skill_type",
        "relation_type"
    ]].sample(min(10, len(df)), random_state=42).to_string(index=False)
)
