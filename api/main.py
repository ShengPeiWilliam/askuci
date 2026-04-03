"""
Step 3 — FastAPI backend with /query endpoint.
Usage: uvicorn api.main:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag.chain import query as rag_query

app = FastAPI(title="AskUCI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str
    k: int = 5


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    result = rag_query(request.question, k=request.k)
    return QueryResponse(answer=result["answer"], sources=result["sources"])
