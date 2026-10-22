import json, logging, time
from collections import Counter
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from app.schemas import AnalyzeRequest, AssistantRequest, FeedbackRequest
from app.engine import analyze, model_metadata
from app.agent import run_assistant
from app.retrieval import retrieve
from app.governance import trace_id, governance_record, utc_now

logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
logger=logging.getLogger("capabilume")
app=FastAPI(title="Capabilume API",version="0.1.0",
 description="Local-first enterprise intelligence prototype: ML, recommendations, RAG, LangGraph and governance.")
METRICS=Counter()
LATENCIES=[]
FEEDBACK=Path("data/feedback.jsonl")
FEEDBACK.parent.mkdir(parents=True,exist_ok=True)

@app.middleware("http")
async def observe(request: Request, call_next):
    start=time.perf_counter(); METRICS["requests_total"]+=1
    try:
        response=await call_next(request)
        if response.status_code>=400: METRICS["errors_total"]+=1
        return response
    except Exception:
        METRICS["errors_total"]+=1
        logger.exception("request_failed path=%s",request.url.path)
        raise
    finally:
        elapsed=(time.perf_counter()-start)*1000
        LATENCIES.append(elapsed)
        if len(LATENCIES)>500: del LATENCIES[:-500]
        logger.info("request path=%s method=%s latency_ms=%.2f",request.url.path,request.method,elapsed)

@app.get("/health")
def health():
    return {"status":"ok","service":"capabilume-api","timestamp":utc_now()}

@app.get("/model-info")
def model_info():
    return model_metadata()

@app.post("/analyze")
def analyze_api(payload: AnalyzeRequest):
    tid=trace_id()
    result=analyze(payload.subject_id,[c.model_dump() for c in payload.current_capabilities],payload.target)
    gov=governance_record(tid,"competency-gap scoring + course ranking",
        "Gap=max(0, required proficiency-observed proficiency); ranking uses gap severity and declared course skill coverage.")
    gov["model_version"]=result.get("ml_signal",{}).get("model_version",model_metadata().get("version","unknown"))
    result["governance_record"]=gov
    logger.info("analysis trace_id=%s target=%s status=%s",tid,payload.target,result.get("status"))
    return result

@app.post("/assistant")
def assistant_api(payload: AssistantRequest):
    return run_assistant(payload.question,payload.subject_id,[c.model_dump() for c in payload.current_capabilities],payload.target)

@app.post("/feedback")
def feedback_api(payload: FeedbackRequest):
    record={"timestamp":utc_now(),**payload.model_dump()}
    with FEEDBACK.open("a",encoding="utf-8") as f: f.write(json.dumps(record,ensure_ascii=False)+"\n")
    logger.info("feedback trace_id=%s decision=%s",payload.trace_id,payload.decision)
    return {"status":"recorded","record":record,"note":"Demo log; production requires authenticated reviewer identity and retention controls."}

@app.get("/knowledge-search")
def knowledge_search(q: str, k: int=4):
    if not q.strip(): raise HTTPException(status_code=400,detail="q must not be empty")
    return {"query":q,"results":retrieve(q,max(1,min(k,8)))}

@app.get("/metrics")
def metrics():
    vals=sorted(LATENCIES)
    p95=vals[min(len(vals)-1,int(.95*len(vals)))] if vals else 0
    return {"requests_total":METRICS["requests_total"],"errors_total":METRICS["errors_total"],
            "error_rate":round(METRICS["errors_total"]/max(1,METRICS["requests_total"]),4),
            "latency_ms":{"count":len(vals),"mean":round(sum(vals)/len(vals),2) if vals else 0,"p95":round(p95,2)},
            "note":"In-memory demo metrics reset on restart; use durable telemetry in production."}
