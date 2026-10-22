from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from app.engine import ROLE_FRAMEWORK, COURSES

DOCS = []
for role, skills in ROLE_FRAMEWORK.items():
    DOCS.append({"source_id":f"framework:{role}","title":f"Demo framework: {role}",
                 "text":f"{role} requires " + ", ".join(f"{s} at level {v}/5" for s,v in skills.items()) + ". Illustrative demo requirements."})
for c in COURSES:
    DOCS.append({"source_id":f"course:{c['id']}","title":c["title"],
                 "text":f"{c['title']}. Skills: {', '.join(c['skills'])}. {c['description']} Duration: {c['hours']} hours."})
DOCS += [
 {"source_id":"governance:human-review","title":"Human review and responsible use",
  "text":"Recommendations support decisions only. A qualified human must review evidence and context. Do not use this prototype to rank employees, make hiring decisions or determine promotions."},
 {"source_id":"mlops:validation","title":"Model validation and monitoring",
  "text":"Validate on held-out representative data. Track data quality, prediction distributions, latency, failures and model version. Use evaluation thresholds and human approval before model promotion."}
]

def retrieve(query, k=4):
    texts=[d["text"] for d in DOCS]
    vec=TfidfVectorizer(ngram_range=(1,2),stop_words="english")
    mat=vec.fit_transform(texts+[query])
    sims=cosine_similarity(mat[-1],mat[:-1]).flatten()
    order=sims.argsort()[::-1][:k]
    return [{**DOCS[int(i)],"score":round(float(sims[int(i)]),4)} for i in order if sims[int(i)]>0]
