from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from .vector_ops import VectorStore
from providers.system_prompt import SYSTEM_PROMPT, CANNED_RESPONSES

app = FastAPI(title="Vector Store Service")

vector_store = VectorStore()

class AddTextRequest(BaseModel):
    text: str
    metadata: Dict[str, Any]

class AddVectorRequest(BaseModel):
    vector: List[float]
    metadata: Dict[str, Any]

class SearchTextRequest(BaseModel):
    query: str
    k: Optional[int] = 5

class SearchVectorRequest(BaseModel):
    query_vector: List[float]
    k: Optional[int] = 5

class DeleteRequest(BaseModel):
    id: int

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/about")
def about(detail: str = "short"):
    level = (detail or "").lower()
    if level not in ("short", "medium", "detailed"):
        return {"error": "detail must be one of: short, medium, detailed"}
    resp = CANNED_RESPONSES.get(level, CANNED_RESPONSES["short"])
    return {"level": level, "response": resp}

@app.post("/add_text")
def add_text(request: AddTextRequest):
    try:
        id = vector_store.add_text(request.text, request.metadata)
        return {"id": id, "status": "added"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add_vector")
def add_vector(request: AddVectorRequest):
    try:
        id = vector_store.add_vector(request.vector, request.metadata)
        return {"id": id, "status": "added"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search_text")
def search_text(request: SearchTextRequest):
    try:
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"VECTOR_STORE: Searching for query: '{request.query}' with k={request.k}")
        results = vector_store.search_text(request.query, request.k)
        logger.info(f"VECTOR_STORE: Found {len(results)} results")
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search_vector")
def search_vector(request: SearchVectorRequest):
    try:
        results = vector_store.search_vectors(request.query_vector, request.k)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metadata")
def get_metadata():
    try:
        return vector_store.get_all_metadata()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/delete")
def delete_vector(request: DeleteRequest):
    try:
        vector_store.delete_vector(request.id)
        return {"status": "deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rebuild")
def rebuild_index():
    try:
        vector_store.rebuild_index()
        return {"status": "rebuilt"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))