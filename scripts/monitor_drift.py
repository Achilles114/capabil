
import json
from pathlib import Path

import pandas as pd
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "splits" / "train.csv"
LOG = ROOT / "reports" / "monitoring" / "predictions.jsonl"
OUTPUT = ROOT / "reports" / "monitoring" / "drift_report.json"


def main():
    if not TRAIN.exists():
        raise FileNotFoundError(f"Training data missing: {TRAIN}")
    if not LOG.exists():
        raise FileNotFoundError(
            f"No prediction logs yet. Make predictions through the API first: {LOG}"
        )

    train = pd.read_csv(TRAIN)
    production = pd.read_json(LOG, lines=True)

    # Compare the text-length distributions.
    text_columns = [
        col for col in [
            "occupation_name",
            "occupation_description",
            "skill_name",
            "skill_description",
        ]
        if col in train.columns
    ]

    if not text_columns:
        raise ValueError(f"No text columns found in training data: {train.columns.tolist()}")

    train_text = train[text_columns].fillna("").astype(str).agg(" ".join, axis=1)
    train_lengths = train_text.str.len()
    live_lengths = production["text_length"].dropna()

    ks = ks_2samp(train_lengths, live_lengths)
    text_drift = bool(ks.pvalue < 0.05)

    # Compare categorical distributions using total variation distance.
    def distribution_distance(reference, current):
        ref = reference.fillna("unknown").astype(str).value_counts(normalize=True)
        cur = current.fillna("unknown").astype(str).value_counts(normalize=True)
        return float(
            0.5 * ref.sub(cur, fill_value=0).abs().sum()
        )

    skill_type_distance = distribution_distance(
        train["skill_type"], production["skill_type"]
    )
    prediction_rate = production["prediction"].value_counts(normalize=True).to_dict()

    report = {
        "training_rows": int(len(train)),
        "logged_predictions": int(len(production)),
        "text_length_ks_statistic": float(ks.statistic),
        "text_length_ks_pvalue": float(ks.pvalue),
        "text_length_drift_detected": text_drift,
        "skill_type_distribution_distance": skill_type_distance,
        "prediction_distribution": prediction_rate,
        "low_confidence_rate_below_0_60": float(
            (production["confidence"] < 0.60).mean()
        ),
        "interpretation": (
            "Potential input drift detected"
            if text_drift or skill_type_distance >= 0.20
            else "No drift threshold triggered"
        ),
        "warning": (
            "Small or unrepresentative production samples can produce "
            "misleading drift results. Drift does not prove model degradation."
        ),
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
