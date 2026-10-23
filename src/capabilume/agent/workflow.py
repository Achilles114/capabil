from typing import TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from functools import lru_cache
from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = ROOT / "models" / "esco_relation_baseline.joblib"

@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)



CONFIDENCE_THRESHOLD = 0.60


class AgentState(TypedDict):
    occupation_name: str
    occupation_description: str
    skill_name: str
    skill_description: str
    skill_type: str
    prediction: str
    confidence: float
    model: str
    human_review_required: bool
    review_reason: str


def predict_skill_relation(state: AgentState) -> dict:
    model = load_model()

    row = pd.DataFrame([{
        "text": (
            f"occupation: {state['occupation_name']} "
            f"{state['occupation_description']} "
            f"skill: {state['skill_name']} "
            f"{state['skill_description']}"
        ),
        "skill_type": state["skill_type"],
    }])

    prediction = model.predict(row)[0]
    probabilities = model.predict_proba(row)[0]
    classes = list(model.classes_)
    confidence = float(probabilities[classes.index(prediction)])

    return {
        "prediction": str(prediction),
        "confidence": confidence,
        "model": "esco_relation_baseline",
    }


def route_by_confidence(state: AgentState) -> str:
    if state["confidence"] < CONFIDENCE_THRESHOLD:
        return "review"
    return "complete"


def flag_for_review(state: AgentState) -> dict:
    return {
        "human_review_required": True,
        "review_reason": (
            f"Confidence {state['confidence']:.3f} is below "
            f"the threshold {CONFIDENCE_THRESHOLD:.2f}."
        ),
    }


def mark_complete(state: AgentState) -> dict:
    return {
        "human_review_required": False,
        "review_reason": "",
    }


builder = StateGraph(AgentState)

builder.add_node("predict", predict_skill_relation)
builder.add_node("flag_review", flag_for_review)
builder.add_node("complete", mark_complete)

builder.add_edge(START, "predict")
builder.add_conditional_edges(
    "predict",
    route_by_confidence,
    {
        "review": "flag_review",
        "complete": "complete",
    },
)
builder.add_edge("flag_review", END)
builder.add_edge("complete", END)

agent_graph = builder.compile()


