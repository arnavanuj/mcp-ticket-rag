# mcp-ticket-rag

Lightweight interview-friendly MVP that ingests GitHub issues/comments, indexes with Chroma, and answers with evidence-backed RAG using local Mistral.

## Quick run

1. Install Python 3.11+
2. `pip install -r requirements.txt`
3. (Optional) set `GITHUB_TOKEN` to avoid low rate limits
4. Run pipeline:

```bash
python scripts/bootstrap_ingest.py
```

## FastAPI

```bash
uvicorn app.main:app --reload
```

APIs:
- `GET /health`
- `POST /ingest`
- `POST /chat`
- `GET /ticket/{issue_number}`
- `GET /search?q=...`

## Notes

- Default ingestion is capped for 16 GB RAM safety.
- OCR is capped by `MAX_OCR_IMAGES`.
- If Chroma or sentence-transformers are unavailable, code falls back and logs warnings.
