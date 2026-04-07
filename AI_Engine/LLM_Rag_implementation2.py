import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from google import genai
from groq import Groq
from Ingestion_Pipeline2 import EmbeddingManager, PineconeVectorStore, BM25SparseEncoder
from dotenv import load_dotenv
load_dotenv()
# ---------------- CONFIGURATION ----------------
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

GEMINI_API_KEY            = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY          = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME       = "my-index"
PINECONE_QUERY_INDEX_NAME = "query-index"
BASE_DIR                  = os.path.dirname(os.path.abspath(__file__))
BM25_PATH                 = os.path.join(BASE_DIR, "datas", "bm25_encoder.json")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL   = "llama-3.1-8b-instant"

# ---------------- CLIENT SETUP ----------------
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
groq_client   = Groq(api_key=GROQ_API_KEY)

embedder = EmbeddingManager()
embedder.load_bm25(BM25_PATH)

db = PineconeVectorStore(
    api_key=PINECONE_API_KEY,
    index_name=PINECONE_INDEX_NAME
)

qdb = PineconeVectorStore(
    api_key=PINECONE_API_KEY,
    index_name=PINECONE_QUERY_INDEX_NAME
)


# ---------------- PROMPT BUILDER ----------------
def build_prompt(
    context: str,
    question: str,
    response_type: str,
    Previous_queries: str = None
) -> str:
    style_map = {
        "Detailed Explanation": "Give a detailed explanation. In 500-600 words",
        "Step-by-Step":         "Explain step-by-step. In 350 words",
        "Short Exam Answer":    "Give a medium length, exam-oriented answer. In 150 words"
    }
    return f"""
You are a helpful tutor for MNNIT students.
Answer ONLY based on the provided context.
If the context does not contain enough information to answer the question,
say: "I don't have information about that in the uploaded documents."
Do NOT use any outside knowledge.
Do NOT mention chunk numbers, scores, or refer to the context explicitly. Answer naturally.

Context:
{context}

Previous Queries:
{Previous_queries}

Question:
{question}

Instruction:
{style_map.get(response_type, "Explain clearly.")}

"""


# ============================================================
# NEW: MERMAID DIAGRAM SUPPORT
# ============================================================

# Diagram types Groq choose kar sakta hai context ke hisaab se
MERMAID_DIAGRAM_TYPES = {
    "flowchart":  "flowchart TD",
    "sequence":   "sequenceDiagram",
    "class":      "classDiagram",
    "state":      "stateDiagram-v2",
    "er":         "erDiagram",
    "mindmap":    "mindmap",
    "timeline":   "timeline",
}

# ============================================================
# NEW: MERMAID DIAGRAM SUPPORT (STRICTLY MINDMAPS)
# ============================================================

def build_mermaid_prompt(context: str, question: str, response_type: str) -> str:
    # Ab humein diagram_hint_map ki zaroorat nahi hai, kyunki hamesha mindmap banega.
    return f"""You are a Mermaid diagram generator.

Output ONLY raw Mermaid syntax for a MINDMAP. No explanation. No markdown fences. No backticks.

Rules:
1. Use ONLY the 'mindmap' diagram type.
2. Mermaid mindmaps rely on strict INDENTATION (spaces or tabs) to show hierarchy. Do NOT use arrows (-->).
3. Do not use special characters or brackets like ( ) : / # & < >.
4. Keep node text under 5 words. Use simple words only.
5. Maximum 10-12 nodes.
6. If context is insufficient, output exactly: NONE

Example of correct mindmap format:
mindmap
  Root Concept
    Sub Topic 1
      Detail A
      Detail B
    Sub Topic 2
      Detail C

Context:
{context}

Question:
{question}

Begin your output directly with the keyword 'mindmap':"""

import re

def generate_mermaid_diagram(context: str, question: str, response_type: str) -> str | None:
    try:
        prompt = build_mermaid_prompt(context, question, response_type)
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=600,
        )
        raw = response.choices[0].message.content.strip()

        if not raw or raw.upper() == "NONE":
            return None

        # ── Step 1: Strip ALL markdown fences robustly ──────────
        raw = re.sub(r"```[a-zA-Z]*", "", raw)   # remove opening fences
        raw = re.sub(r"```", "", raw)             # remove closing fences
        raw = raw.strip()

        # ── Step 2: Validate starts with 'mindmap' keyword ───
        valid_starts = ("mindmap", "mindmap\n")
        if not any(raw.lower().startswith(kw) for kw in valid_starts):
            print(f"[Mermaid] Invalid diagram start, discarding: {raw[:80]}")
            return None

        # ── Step 3: ULTRA-STRICT SANITIZATION (Bomb Fix) ──────────
        # 1. Convert all Tabs to double spaces (Mermaid hates tabs)
        raw = raw.replace("\t", "  ")
        
        # 2. Remove markdown bold/italics
        raw = raw.replace("**", "").replace("*", "").replace("__", "")
        
        # 3. Remove accidental bullet points (e.g., "- Node" becomes "  Node")
        raw = re.sub(r"^\s*[-+*]\s+", "  ", raw, flags=re.MULTILINE)
        
        # 4. Remove colons and dangerous brackets that break mindmaps
        raw = raw.replace(":", " ").replace('"', "'").replace("[", "").replace("]", "").replace("(", "").replace(")", "")
        
        # 5. Remove multiple consecutive empty lines
        raw = re.sub(r'\n\s*\n', '\n', raw)

        # ── Step 4: Final sanity — must have at least 2 lines ────
        lines = [l for l in raw.splitlines() if l.strip()]
        if len(lines) < 2:
            print(f"[Mermaid] Diagram too short, discarding.")
            return None

        return raw

    except Exception as e:
        print(f"[Mermaid] Diagram generation failed: {e}")
        return None

# ============================================================
# END OF MERMAID SUPPORT
# ============================================================


# ---------------- GROQ GENERATION ----------------
def generate_with_groq(prompt: str) -> str:
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


# ---------------- MAIN RETRIEVAL FUNCTION ----------------
def generate_answer(
    question: str,
    subject: str = "",
    response_type: str = "Detailed Explanation",
    top_k: int = 5,
    alpha: float = 0.5,
    include_diagram: bool = True,          # NEW: set False to skip diagram
) -> dict:
    """
    Retrieves relevant chunks from Pinecone using hybrid search
    (Gemini dense + BM25 sparse), then generates an answer via Groq (Llama 3.1).
    Optionally also generates a Mermaid diagram for the same context.

    Returns:
        {
            "answer":  str,          # text answer from Groq
            "mermaid": str | None    # Mermaid diagram string, or None
        }
    """
    try:
        # STEP 1: Embed the query
        print("Generating hybrid query embeddings...")
        dense_vec, sparse_vec = embedder.generate_query_embeddings(question)

        # STEP 2: Hybrid search in Pinecone
        print(f"Querying Pinecone (top_k={top_k}, alpha={alpha})...")
        matches = db.query(
            dense_vec=dense_vec,
            sparse_vec=sparse_vec,
            top_k=top_k,
            alpha=alpha
        )

        if not matches:
            return {
                "answer":  "No relevant information found in the database. Please re-run the ingestion pipeline.",
                "mermaid": None,
            }

        # STEP 3: Build context with relative relevance filter
        max_score = max(m.get("score", 0) for m in matches)
        REL_THRESHOLD = 0.6

        context_parts = [
            f"[Chunk {i+1} | score: {m['score']:.3f}]\n{m['text']}"
            for i, m in enumerate(matches)
            if m.get("score", 0) >= max_score * REL_THRESHOLD
        ]

        # Fallback: always keep at least the top chunk
        if not context_parts:
            top = matches[0]
            context_parts = [f"[Chunk 1 | score: {top['score']:.3f}]\n{top['text']}"]

        context = "\n\n".join(context_parts)

        # STEP 4: Fetch previous queries for context enrichment
        previous_queries = qdb.get_queries(question, embedder)

        # STEP 5: Generate text answer
        print(f"Generating text answer with {GROQ_MODEL}...")
        prompt = build_prompt(context, question, response_type, previous_queries)
        answer = generate_with_groq(prompt)

        # STEP 6: Optionally generate Mermaid diagram
        mermaid = None
        if include_diagram:
            print("Generating Mermaid diagram...")
            mermaid = generate_mermaid_diagram(context, question, response_type)
            if mermaid:
                print("Mermaid diagram generated successfully.")
            else:
                print("No diagram generated for this query.")

        return {"answer": answer, "mermaid": mermaid}

    except Exception as e:
        return {
            "answer":  f"Error during retrieval/generation: {str(e)}",
            "mermaid": None,
        }


# ---------------- TERMINAL TESTING ----------------
if __name__ == "__main__":
    test_q = "What is AI agent?"
    print("\n--- Testing Pinecone Hybrid RAG with Groq (Llama 3.1) + Mermaid ---\n")
    result = generate_answer(
        question=test_q,
        subject="Computer Science",
        response_type="Step-by-Step",
        include_diagram=True,
    )
    print("\n=== ANSWER ===")
    print(result["answer"])
    print("\n=== MERMAID DIAGRAM ===")
    print(result["mermaid"] if result["mermaid"] else "(No diagram generated)")