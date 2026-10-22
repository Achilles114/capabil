from pathlib import Path
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from joblib import load
from app.config import MODEL_PATH, MODEL_META_PATH

ROLE_FRAMEWORK = {
    "AI/ML Engineer": {"Python":4, "SQL":3, "Machine Learning":4, "Model Deployment":4, "NLP and Retrieval":3, "MLOps":3},
    "Data Analyst": {"SQL":4, "Data Visualization":4, "Statistics":3, "Python":3, "Communication":3},
    "Learning Designer": {"Instructional Design":4, "Learning Evaluation":4, "Content Development":4, "Data Literacy":2, "Accessibility":3}
}
COURSES = [
 {"id":"C-101","title":"Python for Applied Data Work","skills":["Python","data processing"],"hours":6,"description":"Python, pandas and practical data cleaning."},
 {"id":"C-102","title":"SQL and Data Modelling","skills":["SQL","database"],"hours":5,"description":"Queries, joins, aggregation and relational design."},
 {"id":"C-103","title":"Applied Machine Learning","skills":["Machine Learning","statistics","model evaluation"],"hours":10,"description":"Supervised learning, validation, metrics and feature engineering."},
 {"id":"C-104","title":"Deploying ML with APIs and Docker","skills":["Model Deployment","MLOps","FastAPI","Docker"],"hours":8,"description":"Serve models with REST APIs, test and deploy with Docker."},
 {"id":"C-105","title":"NLP, Embeddings and Retrieval","skills":["NLP and Retrieval","embeddings","RAG"],"hours":8,"description":"NLP, embeddings, semantic search and RAG evaluation."},
 {"id":"C-106","title":"MLOps Fundamentals","skills":["MLOps","monitoring","model versioning"],"hours":7,"description":"Experiment tracking, pipelines, model versioning and monitoring."},
 {"id":"C-107","title":"Statistics for Decisions","skills":["Statistics","data literacy"],"hours":5,"description":"Sampling, uncertainty and evidence interpretation."},
 {"id":"C-108","title":"Learning Experience Design","skills":["Instructional Design","Content Development","Learning Evaluation"],"hours":7,"description":"Learning objectives, assessment design and evaluation."},
 {"id":"C-109","title":"Accessible Digital Content","skills":["Accessibility","Content Development"],"hours":4,"description":"Inclusive design and accessibility checks."},
 {"id":"C-110","title":"Data Storytelling","skills":["Data Visualization","Communication"],"hours":4,"description":"Presenting findings with clear visuals and evidence."}
]

def model_metadata():
    p = Path(MODEL_META_PATH)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"version":"not-trained","status":"run python scripts/train.py"}

def analyze(subject_id, current_capabilities, target):
    framework = ROLE_FRAMEWORK.get(target)
    if framework is None:
        return {"subject_id":subject_id,"target":target,"status":"unsupported_target",
                "message":f"No approved demo framework. Available targets: {', '.join(ROLE_FRAMEWORK)}",
                "gaps":[],"recommendations":[],"governance":{"human_review_required":True}}
    current = {c["name"].strip().lower():c for c in current_capabilities}
    names = list(current)
    docs = names + list(framework)
    vec = TfidfVectorizer(ngram_range=(1,2), stop_words="english")
    mat = vec.fit_transform(docs) if docs else None
    gaps = []
    for idx, (skill, required) in enumerate(framework.items()):
        if skill.lower() in current:
            c = current[skill.lower()]
            observed, mapping, conf, evidence = c["proficiency"], "exact", 1.0, c.get("evidence","")
        elif names:
            q_idx = len(names) + idx
            sims = cosine_similarity(mat[q_idx:q_idx+1], mat[:len(names)]).flatten()
            j = int(np.argmax(sims))
            if float(sims[j]) >= 0.30:
                c = current[names[j]]
                observed, mapping, conf, evidence = c["proficiency"], names[j], round(float(sims[j]),3), c.get("evidence","")
            else:
                observed, mapping, conf, evidence = 0, None, 0.0, ""
        else:
            observed, mapping, conf, evidence = 0, None, 0.0, ""
        gap = max(0, required-observed)
        gaps.append({"skill":skill,"required_level":required,"observed_level":observed,"gap":gap,
                     "mapping":mapping or "no matching evidence","mapping_confidence":conf,"evidence":evidence,
                     "rationale":f"Required {required}/5; observed {observed}/5 from supplied demo profile."})
    gaps.sort(key=lambda x:(x["gap"],x["required_level"]), reverse=True)
    ranked = []
    for course in COURSES:
        addressed = [g for g in gaps if g["gap"]>0 and any(
            s.lower() in g["skill"].lower() or g["skill"].lower() in s.lower() for s in course["skills"])]
        # Fallback similarity connects related wording without requiring exact skill strings.
        if not addressed and any(g["gap"]>0 for g in gaps):
            gap_text = " ".join(g["skill"] for g in gaps if g["gap"]>0)
            course_text = " ".join(course["skills"]+[course["title"],course["description"]])
            pair = TfidfVectorizer(ngram_range=(1,2),stop_words="english").fit_transform([gap_text,course_text])
            sim = float(cosine_similarity(pair[0:1],pair[1:2])[0][0])
            if sim > .08:
                addressed = [g for g in gaps if g["gap"]>0][:1]
        if addressed:
            gap_score = sum(g["gap"] for g in addressed)/len(addressed)
            score = round(.7*(gap_score/5)+.3*min(1,len(addressed)/2),3)
            ranked.append({"recommendation_id":course["id"],"course_id":course["id"],"title":course["title"],
                           "score":score,"duration_hours":course["hours"],"addresses":[g["skill"] for g in addressed],
                           "rationale":"Ranked by competency-gap severity and declared course skill coverage."})
    ranked.sort(key=lambda x:x["score"],reverse=True)
    model = None
    mp = Path(MODEL_PATH)
    if mp.exists():
        model = load(mp)
    meta = model_metadata()
    priority = None
    if model:
        mean_gap = float(np.mean([g["gap"] for g in gaps])) if gaps else 0
        max_gap = float(max([g["gap"] for g in gaps],default=0))
        missing = sum(g["observed_level"]==0 for g in gaps)
        priority = str(model.predict([[mean_gap,max_gap,missing,len(current_capabilities)]])[0])
    return {"subject_id":subject_id,"target":target,"status":"ok","gaps":gaps,"recommendations":ranked[:5],
            "ml_signal":{"predicted_intervention_priority":priority,"model_version":meta.get("version","unknown"),
                         "note":"Synthetic-label demo classifier; not validated for real workforce prediction."},
            "governance":{"human_review_required":True,"decision_scope":"learning recommendations only",
                          "traceability":"Includes observed/required levels, mapping, evidence and rationale.",
                          "warning":"Never use this prototype for hiring, promotion or employee ranking."}}
