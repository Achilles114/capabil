from uuid import uuid4
from datetime import datetime, timezone

def trace_id():
    return f"tr-{uuid4().hex[:12]}"

def utc_now():
    return datetime.now(timezone.utc).isoformat()

def governance_record(trace, method, rationale, sources=None):
    return {
        "trace_id": trace, "timestamp": utc_now(), "method": method,
        "rationale": rationale, "sources": sources or [], "model_version": "unknown",
        "human_review_required": True,
        "decision_scope": "learning/intervention support only",
        "prohibited_uses": ["automated hiring", "promotion decisions", "employee ranking"]
    }
