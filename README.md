# Capabilume — Intelligent Enterprise Platform

Local-first interview prototype combining ML inference, competency mapping, recommendations, RAG, LangGraph, Ollama, FastAPI, evaluation, and basic observability.

**Important:** Uses synthetic data and illustrative competency requirements. It is not validated for real workforce decisions. Human review is required.

## Setup (Windows PowerShell)
Requires Python 3.11/3.12 and Ollama. First check your model name:
```powershell
ollama list
```
Then:
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```
Edit `.env` and set `OLLAMA_MODEL` to the exact model name from `ollama list`.

Train and start:
```powershell
python scripts/train.py
uvicorn app.main:app --reload
```
Open http://127.0.0.1:8000/docs.

If activation is blocked:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\train.py
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

## Demo sequence
1. `GET /health`
2. `GET /model-info`
3. `POST /analyze`
4. `POST /assistant` — LangGraph runs analysis → retrieval → Ollama explanation.
5. `POST /feedback`
6. `GET /metrics`
7. `python -m app.evaluate`

Example JSON for `/analyze` or `/assistant`:
```json
{
  "subject_id": "demo-001",
  "target": "AI/ML Engineer",
  "current_capabilities": [
    {"name": "Python", "proficiency": 3, "evidence": "Automation project"},
    {"name": "SQL", "proficiency": 2, "evidence": "Basic reporting"},
    {"name": "Machine Learning", "proficiency": 1, "evidence": "Introductory course"}
  ],
  "question": "Explain the most important gaps and recommend learning interventions."
}
```
For `/analyze`, omit `question`.

## What this demonstrates
- Synthetic-data scikit-learn training and versioned artifact
- Transparent competency gaps and course ranking
- FastAPI inference endpoints
- TF-IDF knowledge retrieval with source IDs
- LangGraph orchestration with local Ollama; deterministic fallback if unavailable
- MLflow training run, offline ranking evaluation, latency/error metrics, feedback log
- Traceability, explicit human-review flag and prohibited-use safeguards

The synthetic classifier metric only verifies the pipeline, not real-world predictive validity. The core scoring/recommendation engine works independently of Ollama. For production, add authenticated identities, authorization, tenant isolation, durable audit storage, CI quality gates, approved data retention, real representative evaluation, drift monitoring, and deployment approvals.
