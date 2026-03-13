import json, sys
sys.path.insert(0,'.')
from app.core.config import get_settings
from app.services.ingest_service import run_ingest
from app.services.chat_service import chat

question = "find similar issues discussing errors or failures due to package or memory issues and give me issues summary, resolution, who resolved it, how long the issue was open, issue id"
settings = get_settings()
summary = {
    "llava_primary_model": settings.phi3_vision_model,
    "mistral_model": settings.mistral_model,
}
try:
    ingest = run_ingest(settings)
    summary.update({"ingest_ran": True, **ingest})
except Exception as e:
    summary.update({"ingest_ran": False, "ingest_error": str(e)})

resp = chat(settings=settings, query=question, top_k=15)
summary["question"] = question
summary["final_answer"] = resp.get("answer")
summary["issue_results"] = resp.get("issue_results", [])
summary["evidence"] = resp.get("evidence", [])
print(json.dumps(summary, indent=2, ensure_ascii=False))
