import os
import re
import numpy as np
import hashlib
import json
import time
from google import genai
from google.genai import types
from pinecone import Pinecone, ServerlessSpec
from rank_bm25 import BM25Okapi
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

class BM25SparseEncoder:
    """
    Drop-in replacement for pinecone_text's BM25Encoder.
    Works on Python 3.12+ where pinecone-text breaks.
    """

    def __init__(self):
        self.bm25 = None
        self.vocabulary: Dict[str, int] = {}
        self._tokenized_corpus: List[List[str]] = []

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def fit(self, texts: List[str]):
        print("Fitting BM25 on corpus...")
        new_tokenized = [self._tokenize(t) for t in texts]  
        existing = set(tuple(d) for d in self._tokenized_corpus)
        new_tokenized = [t for t in new_tokenized if tuple(t) not in existing]
        self._tokenized_corpus.extend(new_tokenized)

        new_words = set(word for doc in new_tokenized for word in doc)

        if self.vocabulary:
            next_idx = max(self.vocabulary.values()) + 1
            added = 0
            for word in sorted(new_words):
                if word not in self.vocabulary:
                    self.vocabulary[word] = next_idx
                    next_idx += 1
                    added += 1
            print(f"Merged vocabulary: {added} new tokens added "
                  f"(total: {len(self.vocabulary)})")
        else:
            self.vocabulary = {
                word: idx for idx, word in enumerate(sorted(new_words))
            }
            print(f"BM25 ready. Vocabulary size: {len(self.vocabulary)}")

        self.bm25 = BM25Okapi(self._tokenized_corpus)

    @staticmethod
    def _l2_normalize(indices: List[int], values: List[float]) -> Dict:
        """L2-normalize sparse vector values for consistent dotproduct scoring."""
        if not values:
            return {"indices": indices, "values": values}
        arr = np.array(values, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return {"indices": indices, "values": arr.tolist()}

    def encode_documents(self, texts: List[str]) -> List[Dict]:
        if not self.bm25:
            raise ValueError("BM25 not fitted. Call fit() first.")

        k1 = self.bm25.k1
        b  = self.bm25.b
        avgdl = self.bm25.avgdl
        results = []

        for text in texts:
            tokens = self._tokenize(text)
            dl = len(tokens)

            term_counts: Dict[str, int] = {}
            for t in tokens:
                if t in self.vocabulary:
                    term_counts[t] = term_counts.get(t, 0) + 1

            indices, values = [], []
            for term, tf in term_counts.items():
                idf = self.bm25.idf.get(term, 0.0)
                weight = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
                if weight > 0:
                    indices.append(self.vocabulary[term])
                    values.append(float(weight))

            results.append(self._l2_normalize(indices, values))

        return results

    def encode_queries(self, query: str) -> Dict:
        """
        Encode a query using IDF-weighted term frequency.
        FIX: now uses idf * tf instead of raw tf for consistent scoring with documents.
        """
        if not self.vocabulary:
            raise ValueError("Vocabulary is empty. Call fit() or load() first.")
        if not self.bm25:
            raise ValueError("BM25 not fitted. Call fit() or load() first.")

        tokens = self._tokenize(query)
        term_counts: Dict[str, int] = {}
        for token in tokens:
            if token in self.vocabulary:
                term_counts[token] = term_counts.get(token, 0) + 1

        indices, values = [], []
        for term, tf in term_counts.items():
            idf = self.bm25.idf.get(term, 0.0)
            weight = idf * tf
            if weight > 0:
                indices.append(self.vocabulary[term])
                values.append(float(weight))

        return self._l2_normalize(indices, values)

    def save(self, path: str = "bm25_encoder.json"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "vocabulary": self.vocabulary,
            "corpus": self._tokenized_corpus
        }
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"BM25 encoder saved to {path}")

    def load(self, path: str = "bm25_encoder.json"):
        with open(path, "r") as f:
            data = json.load(f)

        self.vocabulary = data.get("vocabulary", {})
        self._tokenized_corpus = data.get("corpus", [])

        if self._tokenized_corpus:
            self.bm25 = BM25Okapi(self._tokenized_corpus)

        print(f"BM25 encoder loaded from {path}")
        return self


class EmbeddingManager:
    GEMINI_EMBEDDING_DIM = 3072
    GEMINI_MODEL = "gemini-embedding-001"

    def __init__(self):
        self.bm25 = BM25SparseEncoder()
        self.client = genai.Client(api_key="AIzaSyCnkSUJiDhIZy5gaLvGo7PxILQUy2uitwo")
        print("EmbeddingManager ready (Gemini dense + BM25 sparse).")

    def fit_bm25(self, texts: List[str]):
        self.bm25.fit(texts)

    def save_bm25(self, path: str = "bm25_encoder.json"):
        self.bm25.save(path)

    def load_bm25(self, path: str = "bm25_encoder.json"):
        self.bm25.load(path)

    @staticmethod
    def _l2_normalize(vector: list) -> list:
        v = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(v)
        return (v / norm).tolist() if norm > 0 else v.tolist()

    def _embed_with_retry(self, contents, task_type: str) -> list:
        """Shared retry logic for all Gemini embed calls."""
        while True:
            try:
                result = self.client.models.embed_content(
                    model=self.GEMINI_MODEL,
                    contents=contents,
                    config=types.EmbedContentConfig(task_type=task_type)
                )
                return [EmbeddingManager._l2_normalize(e.values) for e in result.embeddings]
            except Exception as e:
                if "429" in str(e):
                    print("Quota hit! Sleeping for 60 seconds...")
                    time.sleep(60)
                    continue
                raise

    def generate_dense_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using Gemini in batches."""
        all_embeddings = []
        batch_size = 50
        print(f"Total chunks: {len(texts)}. Processing in batches of {batch_size}...")

        i = 0
        while i < len(texts):
            batch = texts[i: i + batch_size]
            embeddings = self._embed_with_retry(batch, "RETRIEVAL_DOCUMENT")
            all_embeddings.extend(embeddings)
            print(f"Processed chunks {i} to {i + len(batch)}")
            i += batch_size
            time.sleep(2)

        return all_embeddings

    def generate_sparse_embeddings(self, texts: List[str]) -> List[Dict]:
        print(f"Generating sparse embeddings for {len(texts)} texts...")
        return self.bm25.encode_documents(texts)

    def generate_query_embeddings(self, query: str):
        """Generate dense + sparse vectors for a query."""
        dense = self._embed_with_retry(query, "RETRIEVAL_QUERY")[0]
        sparse = self.bm25.encode_queries(query)
        return dense, sparse


class PineconeVectorStore:
    def __init__(
        self,
        api_key: str,
        index_name: str,
        dimension: int = EmbeddingManager.GEMINI_EMBEDDING_DIM,
        cloud: str = "aws",
        region: str = "us-east-1"
    ):
        self.api_key = api_key
        self.index_name = index_name
        self.dimension = dimension
        self.cloud = cloud
        self.region = region
        self.index = None
        self._initialize()

    def _initialize(self):
        try:
            print("Initializing Pinecone...")
            pc = Pinecone(api_key=self.api_key)
            if self.index_name not in pc.list_indexes().names():
                print(f"Creating index: {self.index_name}")
                pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="dotproduct",
                    spec=ServerlessSpec(cloud=self.cloud, region=self.region)
                )
            else:
                print(f"Index '{self.index_name}' already exists.")
            self.index = pc.Index(self.index_name)
            print("Pinecone index ready.")
        except Exception as e:
            print(f"Error initializing Pinecone: {e}")
            raise

    def add_documents(
        self,
        chunks,
        dense_embeddings: List[List[float]],
        sparse_embeddings: List[Dict],
        batch_size: int = 100
    ):
        try:
            print(f"Upserting {len(dense_embeddings)} vectors...")
            vectors = []
            for i, (chunk, dense) in enumerate(zip(chunks, dense_embeddings)):
                metadata = {"text": chunk.page_content, **chunk.metadata}
                vectors.append({
                    "id": hashlib.md5(chunk.page_content.encode()).hexdigest(),
                    "values": dense,
                    "sparse_values": sparse_embeddings[i],
                    "metadata": metadata
                })

            for i in range(0, len(vectors), batch_size):
                self.index.upsert(vectors[i: i + batch_size])
                print(f"Upserted batch {i // batch_size + 1}")

            print("Upsert successful.")
        except Exception as e:
            print(f"Error during upsert: {e}")
            raise

    def query(
        self,
        dense_vec: List[float],
        sparse_vec: Dict,
        top_k: int = 5,
        alpha: float = 0.5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict]:
        try:
            scaled_dense = [v * alpha for v in dense_vec]
            scaled_sparse = {
                "indices": sparse_vec["indices"],
                "values": [v * (1 - alpha) for v in sparse_vec["values"]]
            }
            results = self.index.query(
                vector=scaled_dense,
                sparse_vector=scaled_sparse,
                top_k=top_k,
                include_metadata=True,
                filter=filter
            )
            return [
                {
                    "text": match["metadata"].get("text"),
                    "score": match["score"],
                    "metadata": match["metadata"]
                }
                for match in results["matches"]
            ]
        except Exception as e:
            print(f"Error during query: {e}")
            raise

    def delete_index(self):
        """Delete the entire Pinecone index."""
        try:
            pc = Pinecone(api_key=self.api_key)
            pc.delete_index(self.index_name)
            self.index = None
            print(f"Index '{self.index_name}' deleted successfully.")
        except Exception as e:
            print(f"Error deleting index: {e}")
            raise
        
    def insert_query(
        self,
        query: str,
        dense: List[float],
        sparse: Dict,
    ):
        try:
            vector_id = hashlib.md5(query.encode()).hexdigest()
            meta = {"text": query, "type": "query"}
            self.index.upsert([{
                "id": vector_id,
                "values": dense,
                "sparse_values": sparse,
                "metadata": meta
            }])
            print(f"Query upserted with id: {vector_id}")
            return vector_id
        except Exception as e:
            print(f"Error inserting query: {e}")
            raise

    def get_queries(
        self,
        query: str,
        embedder: EmbeddingManager,
        top_k: int = 5
    ) -> List[Dict]:
        try:
            dense, sparse = embedder.generate_query_embeddings(query)
            results = self.query(
                dense_vec=dense,
                sparse_vec=sparse,
                top_k=top_k,
            )
            self.insert_query(query,dense,sparse)
            return results
        except Exception as e:
            print(f"Error fetching past queries: {e}")
            raise


def run_ingestion_pipeline():
    pdf_directory = "data/pdf"

    print("\n" + "=" * 50)
    print("STARTING GEMINI-ONLY INGESTION PIPELINE")
    print("=" * 50)

    if not os.path.exists(pdf_directory):
        os.makedirs(pdf_directory, exist_ok=True)
        print(f"Created directory '{pdf_directory}'. Add PDFs and run again.")
        return

    # Load PDFs
    dir_loader = DirectoryLoader(
        pdf_directory,
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader,
        show_progress=True
    )
    pdf_docs = dir_loader.load()
    if not pdf_docs:
        print(f"No PDFs found in '{pdf_directory}'.")
        return

    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(pdf_docs)
    chunk_texts = [chunk.page_content for chunk in chunks]
    print(f"Created {len(chunks)} text chunks.")

    # Embeddings
    embedder = EmbeddingManager()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    BM25_PATH = os.path.join(BASE_DIR, "datas", "bm25_encoder.json")
    if os.path.exists(BM25_PATH):
        embedder.load_bm25(BM25_PATH)
        print("Loaded existing BM25 vocab — new tokens will be merged in.")
    embedder.fit_bm25(chunk_texts)
    embedder.save_bm25(BM25_PATH)

    dense = embedder.generate_dense_embeddings(chunk_texts)
    sparse = embedder.generate_sparse_embeddings(chunk_texts)

    # Store in Pinecone
    db = PineconeVectorStore(
        api_key=os.getenv("PINECONE_API_KEY"),
        index_name="my-index"
    )
    db.add_documents(chunks, dense, sparse)

    print("\nGEMINI PIPELINE COMPLETED! Restart backend and ask questions.")


if __name__ == "__main__":
    run_ingestion_pipeline()