# evaluator.py

from AI_Engine.LLM_Rag_implementation2 import generate_answer, db
from AI_Engine.Ingestion_Pipeline2 import EmbeddingManager, PineconeVectorStore

def collect_rag_output(question, top_k=5, alpha=0.5):
    """
    Runs the RAG pipeline and collects:
    - generated answer
    - retrieved context chunks
    """

    # Get dense and sparse vectors
    manager=EmbeddingManager()
    dense_vec, sparse_vec = manager.generate_query_embeddings(question)
    # Get matches from Pinecone
    matches = db.query(
        dense_vec=dense_vec,
        sparse_vec=sparse_vec,
        top_k=top_k,
        alpha=alpha
    )

    # Collect context texts separately
    contexts = [match.get("text", "") for match in matches]

    # Generate answer
    answer = generate_answer(question)

    return {
        "question":  question,
        "answer":    answer,
        "contexts":  contexts   # list of retrieved chunks
    }