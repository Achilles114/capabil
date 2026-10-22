from app.engine import analyze
from app.retrieval import retrieve

CASES=[
 {"subject_id":"eval-1","target":"AI/ML Engineer","current_capabilities":[{"name":"Python","proficiency":3,"evidence":"demo"},{"name":"SQL","proficiency":2,"evidence":"demo"}],"relevant":{"C-103","C-104","C-105","C-106"}},
 {"subject_id":"eval-2","target":"Learning Designer","current_capabilities":[{"name":"Content Development","proficiency":2,"evidence":"demo"}],"relevant":{"C-108","C-109"}}
]
def main():
    ps=[]; rrs=[]
    for c in CASES:
        out=analyze(c["subject_id"],c["current_capabilities"],c["target"])
        recs=out["recommendations"][:3]
        ps.append(sum(r["course_id"] in c["relevant"] for r in recs)/3)
        rr=next((1/i for i,r in enumerate(out["recommendations"],1) if r["course_id"] in c["relevant"]),0)
        rrs.append(rr)
        print(f"{c['subject_id']} | Precision@3={ps[-1]:.3f} | reciprocal_rank={rr:.3f}")
    print(f"Mean Precision@3: {sum(ps)/len(ps):.3f}; MRR: {sum(rrs)/len(rrs):.3f}")
    print("Tiny hand-labelled demo set only; expand before drawing conclusions.")
    print("Retrieval smoke test:",[x["source_id"] for x in retrieve("human review and evidence for workforce recommendations")])
if __name__=="__main__": main()
