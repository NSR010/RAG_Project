from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
import pinecone

# ---------------- PATH SETUP ----------------
current_dir   = os.path.dirname(os.path.abspath(__file__))
root_dir      = os.path.abspath(os.path.join(current_dir, '..'))
ai_engine_dir = os.path.join(root_dir, 'AI_Engine')

if root_dir not in sys.path:
    sys.path.append(root_dir)
if ai_engine_dir not in sys.path:
    sys.path.append(ai_engine_dir)

# ---------------- IMPORT AI MODULES ----------------
from AI_Engine.LLM_Rag_implementation2 import (
    generate_answer,
    embedder,
    db,
    PINECONE_API_KEY,
)

# ---------------- LIFESPAN ----------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 MNNIT Study Mate Backend is starting up...")
    yield
    print("🧹 Backend is shutting down... Cleaning up 'query-index'.")
    try:
        pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
        if "query-index" in pc.list_indexes().names():
            pc.delete_index("query-index")
            print("✅ 'query-index' deleted successfully.")
        else:
            print("⚠️ 'query-index' not found. Nothing to delete.")
    except Exception as e:
        print(f"❌ Error during shutdown cleanup: {e}")

# ---------------- FASTAPI APP ----------------
app = FastAPI(
    title="MNNIT Study Mate API",
    description="Backend API for Hybrid RAG-based study platform using Gemini, Groq, and Pinecone.",
    lifespan=lifespan,
)

# ---------------- DATA MODELS ----------------
class QueryRequest(BaseModel):
    question:      str
    subject:       str  = "General"
    response_type: str  = "Detailed Explanation"
    # NEW: client can opt out of diagram generation (default: True)
    include_diagram: bool = True


class QueryResponse(BaseModel):
    answer:  str
    # NEW: Mermaid diagram string — None when no diagram was generated
    mermaid: Optional[str] = None


class DocumentInput(BaseModel):
    page_content: str
    metadata: Dict[str, Any] = {}


class IngestRequest(BaseModel):
    documents: List[DocumentInput]


class IngestResponse(BaseModel):
    message: str
    status:  str


# ---------------- ENDPOINTS ----------------

@app.get("/")
def read_root():
    return {"status": "Backend is running! MNNIT Tutor AI is ready. 🚀"}


@app.post("/api/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """
    Hybrid Search (Dense + Sparse) se relevant chunks retrieve karke
    Groq se text answer + optional Mermaid diagram generate karta hai.

    Response fields:
      - answer  : plain-text / markdown answer string
      - mermaid : Mermaid diagram syntax string (or null if not generated)
                  Render this directly with any Mermaid-compatible renderer.
    """
    try:
        result = generate_answer(
            question=request.question,
            subject=request.subject,
            response_type=request.response_type,
            include_diagram=request.include_diagram,   # NEW
        )
        return QueryResponse(
            answer=result["answer"],
            mermaid=result.get("mermaid"),             # NEW
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Generation Error: {str(e)}")


@app.post("/api/ingest", response_model=IngestResponse)
async def ingest_documents(request: IngestRequest):
    """
    Naye text documents ko Gemini aur BM25 se embed karke Pinecone mein save karta hai.
    """
    try:
        bm25_absolute_path = os.path.join(root_dir, "bm25_encoder.json")
        texts = [doc.page_content for doc in request.documents]

        if os.path.exists(bm25_absolute_path):
            embedder.load_bm25(bm25_absolute_path)

        embedder.fit_bm25(texts)
        embedder.save_bm25(bm25_absolute_path)

        dense_embeddings  = embedder.generate_dense_embeddings(texts)
        sparse_embeddings = embedder.generate_sparse_embeddings(texts)

        db.add_documents(
            chunks=request.documents,
            dense_embeddings=dense_embeddings,
            sparse_embeddings=sparse_embeddings,
        )

        return IngestResponse(
            status="success",
            message=f"{len(texts)} documents successfully processed and saved to Pinecone.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database Ingestion Error: {str(e)}")


# ---------------- SERVER TRIGGER ----------------
if __name__ == "__main__":
    import uvicorn
    print("Triggering Uvicorn Server on port 8000...")
    uvicorn.run(app, host="127.0.0.1", port=8000)