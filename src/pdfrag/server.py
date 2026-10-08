import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# Import the exact entrypoint function from your pipeline.py file
from pdfrag.pipeline import answer_question

app = FastAPI(title="pdfDocRag Production API Server")

class QueryRequest(BaseModel):
    question: str

@app.get("/health")
def health_check():
    """Deterministic endpoint for AWS environment and Load Balancer health hooks."""
    return {"status": "healthy", "service": "pdfDocRag"}

@app.post("/api/v1/query")
def run_query(payload: QueryRequest):
    """Exposes the guarded RAG pipeline via standard HTTP POST fabrics."""
    try:
        # Pass the question string directly into your custom LCEL query runner
        answer = answer_question(payload.question)
        return {"query": payload.question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
