from pathlib import Path
from datetime import datetime, timezone
import json, sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump
import mlflow
import mlflow.sklearn

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/"models"; OUT.mkdir(exist_ok=True)
rng=np.random.default_rng(42); X=[]; y=[]
# Synthetic rule-generated labels validate the training pipeline only.
for _ in range(500):
    mean_gap=float(rng.uniform(0,4)); max_gap=float(rng.uniform(0,5))
    missing=int(rng.integers(0,7)); observed=int(rng.integers(0,10))
    risk=.55*mean_gap+.25*max_gap+.35*missing-.08*observed
    label="high" if risk>=3 else ("medium" if risk>=1.7 else "low")
    if rng.random()<.08: label=str(rng.choice(["low","medium","high"]))
    X.append([mean_gap,max_gap,missing,observed]); y.append(label)
X=np.asarray(X); y=np.asarray(y)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
model=RandomForestClassifier(n_estimators=80,max_depth=6,random_state=42,class_weight="balanced")
model.fit(Xtr,ytr); pred=model.predict(Xte)
acc=float(accuracy_score(yte,pred))
version=datetime.now(timezone.utc).strftime("demo-%Y%m%dT%H%M%SZ")
dump(model,OUT/"intervention_model.joblib")
meta={"version":version,"created_at":datetime.now(timezone.utc).isoformat(),"algorithm":"RandomForestClassifier",
 "features":["mean_gap","max_gap","missing_skill_count","observed_capability_count"],"training_data":"synthetic_rule_generated",
 "holdout_accuracy":round(acc,4),"classification_report":classification_report(yte,pred,output_dict=True,zero_division=0),
 "warning":"Synthetic data/labels: metric does not indicate real-world predictive quality."}
(OUT/"model_metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
mlflow.set_tracking_uri((ROOT/"mlruns").as_uri())
with mlflow.start_run(run_name="capabilume-synthetic-training"):
    mlflow.log_param("algorithm","RandomForestClassifier"); mlflow.log_param("n_estimators",80)
    mlflow.log_param("max_depth",6); mlflow.log_param("data_type","synthetic_rule_generated")
    mlflow.log_metric("holdout_accuracy",acc); mlflow.sklearn.log_model(model,"model")
print(f"Saved model; version={version}; synthetic holdout accuracy={acc:.3f}")
print("This validates the demo pipeline only, not real-world predictive quality.")
