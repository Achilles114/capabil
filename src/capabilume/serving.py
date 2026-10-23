
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from functools import lru_cache

import json
from datetime import datetime, timezone

from capabilume.agent.workflow import agent_graph

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "esco_relation_baseline.joblib"
PREDICTION_LOG = ROOT / "reports" / "monitoring" / "predictions.jsonl"

@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

app = FastAPI(
    title="Capabilume ML API",
    description="Predicts ESCO occupation-skill relationship types.",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    occupation_name: str = Field(min_length=1, max_length=300)
    occupation_description: str = Field(default="", max_length=5000)
    skill_name: str = Field(min_length=1, max_length=300)
    skill_description: str = Field(default="", max_length=5000)
    skill_type: str = Field(default="unknown", max_length=100)


class PredictionResponse(BaseModel):
    predicted_relation: str
    confidence: float
    model: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL_PATH.exists(),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model file not found.")

    try:
        model = load_model()
        row = pd.DataFrame([{
            "text": (
                f"occupation: {request.occupation_name} "
                f"{request.occupation_description} "
                f"skill: {request.skill_name} "
                f"{request.skill_description}"
            ),
            "skill_type": request.skill_type,
        }])

        prediction = model.predict(row)[0]
        probabilities = model.predict_proba(row)[0]
        classes = list(model.classes_)
        confidence = float(probabilities[classes.index(prediction)])

        PREDICTION_LOG.parent.mkdir(parents=True, exist_ok=True)

        with PREDICTION_LOG.open("a", encoding="utf-8") as file:
            file.write(json.dumps({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "text_length": len(row.iloc[0]["text"]),
                "skill_type": request.skill_type,
                "prediction": str(prediction),
                "confidence": confidence,
            }) + "\n")

        return PredictionResponse(
            predicted_relation=str(prediction),
            confidence=confidence,
            model="esco_relation_baseline",
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed.",
        ) from exc

@app.post("/agent/analyze")
def analyze_with_agent(request: PredictionRequest):
    try:
        result = agent_graph.invoke({
            **request.model_dump(),
            "prediction": "",
            "confidence": 0.0,
            "model": "",
            "human_review_required": False,
            "review_reason": "",
        })

        return {
            "predicted_relation": result["prediction"],
            "confidence": result["confidence"],
            "model": result["model"],
            "human_review_required": result["human_review_required"],
            "review_reason": result["review_reason"],
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Agent workflow failed.",
        ) from exc
