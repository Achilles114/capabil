from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL
from app.engine import analyze
from app.retrieval import retrieve
from app.governance import governance_record, trace_id

class State(TypedDict, total=False):
    question: str
    subject_id: str
    current_capabilities: list[dict]
    target: str
    analysis: dict
    sources: list[dict]
    answer: str
    llm_status: str

def analyze_node(s: State):
    return {"analysis":analyze(s["subject_id"],s["current_capabilities"],s["target"])}

def retrieve_node(s: State):
    gaps=[g["skill"] for g in s.get("analysis",{}).get("gaps",[]) if g.get("gap",0)>0]
    return {"sources":retrieve(f"{s.get('question','')} {s.get('target','')} {' '.join(gaps)}",4)}

def explain_node(s: State):
    analysis=s.get("analysis",{})
    sources=s.get("sources",[])
    context="\n\n".join(f"[{d['source_id']}] {d['title']}: {d['text']}" for d in sources)
    brief={"gaps":[{k:g.get(k) for k in ("skill","required_level","observed_level","gap","evidence","rationale")} for g in analysis.get("gaps",[])[:5]],
           "recommendations":analysis.get("recommendations",[])[:5]}
    prompt=f"""You explain a responsible enterprise intelligence prototype.
Use only the analysis and retrieved context. Explain key gaps and recommendations, cite source IDs in square brackets, and say requirements are illustrative demo data.
Do not infer a person's worth, potential, or employment outcome. If evidence is missing, say so.
Question: {s.get('question')}
Analysis: {brief}
Retrieved context:
{context}
Keep it concise."""
    try:
        llm=ChatOllama(model=OLLAMA_MODEL,base_url=OLLAMA_BASE_URL,temperature=0)
        response=llm.invoke(prompt)
        return {"answer":response.content if isinstance(response.content,str) else str(response.content),"llm_status":f"ollama:{OLLAMA_MODEL}"}
    except Exception as exc:
        gaps=[g for g in analysis.get("gaps",[]) if g.get("gap",0)>0][:3]
        courses=analysis.get("recommendations",[])[:3]
        answer="Local LLM unavailable; deterministic fallback.\n"
        answer+="Priority gaps: "+(", ".join(f"{g['skill']} (gap {g['gap']})" for g in gaps) or "none detected")+".\n"
        answer+="Interventions: "+(", ".join(c["title"] for c in courses) or "none ranked")+".\n"
        answer+="Sources: "+(", ".join(d["source_id"] for d in sources) or "none")+". Human review required."
        return {"answer":answer,"llm_status":f"fallback:{type(exc).__name__}"}

graph=StateGraph(State)
graph.add_node("analyze",analyze_node)
graph.add_node("retrieve",retrieve_node)
graph.add_node("explain",explain_node)
graph.set_entry_point("analyze")
graph.add_edge("analyze","retrieve")
graph.add_edge("retrieve","explain")
graph.add_edge("explain",END)
WORKFLOW=graph.compile()

def run_assistant(question, subject_id, current_capabilities, target):
    tid=trace_id()
    result=WORKFLOW.invoke({"question":question,"subject_id":subject_id,"current_capabilities":current_capabilities,"target":target})
    sources=result.get("sources",[])
    analysis=result.get("analysis",{})
    gov=governance_record(tid,"LangGraph: analyze -> retrieve -> explain",
        "Combines deterministic competency analysis with retrieved supporting documents.",[s["source_id"] for s in sources])
    gov["model_version"]=analysis.get("ml_signal",{}).get("model_version","unknown")
    return {"trace_id":tid,"answer":result.get("answer",""),"analysis":analysis,"sources":sources,
            "llm_status":result.get("llm_status","unknown"),"governance":gov}
