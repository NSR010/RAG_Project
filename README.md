# 🚀 MNNIT Study Mate AI

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit)
![Pinecone](https://img.shields.io/badge/Pinecone-Vector_DB-success)
![Groq](https://img.shields.io/badge/Groq-Llama_3.1-orange)

An advanced, Agentic AI-powered study platform designed specifically for engineering students at MNNIT. This project implements a cutting-edge **Hybrid RAG (Retrieval-Augmented Generation)** pipeline to provide highly accurate, context-aware answers, complete with auto-generated interactive visual mindmaps.

---

## ✨ Key Features

* 🔍 **Hybrid Search RAG:** Combines **Gemini (Dense)** embeddings and **BM25 (Sparse)** encoding for superior, context-perfect document retrieval.
* ⚡ **Lightning Fast Generation:** Powered by **Groq (Llama 3.1 - 8B)** for near-instant text generation and conceptual explanations.
* 🧠 **Auto-Generated Visuals:** Dynamically creates strict **Mermaid.js Mindmaps** at the end of explanations to help students visualize complex topics and hierarchies.
* 🛡️ **Robust Backend:** Built with **FastAPI**, featuring automated startup/shutdown lifespans and strict query-index cleanup protocols.
* 🔐 **Secure API Management:** 100% secure architecture using `.env` for environment variables, ensuring no API keys are ever leaked.

## 🏗️ Architecture Workflow

1. **Ingestion:** PDFs are chunked via LangChain -> Embedded using Gemini (Dense) & BM25 (Sparse) -> Upserted to Pinecone.
2. **Retrieval:** User queries are hybrid-searched against the Pinecone Vector DB to fetch the most relevant syllabus chunks.
3. **Generation:** Context is passed to Groq (Llama-3) to generate a structured text explanation AND a Mermaid mindmap.
4. **Delivery:** The Streamlit frontend renders the text and visualizes the mindmap natively.

## 🛠️ Tech Stack

* **Backend Framework:** FastAPI, Uvicorn
* **Frontend:** Streamlit (with native Mermaid.js rendering)
* **LLMs & Models:** Groq (Llama-3.1-8b-instant), Google Gemini (gemini-embedding-001)
* **Vector Database:** Pinecone (Serverless)
* **Sparse Encoder:** Rank-BM25
* **Document Parsing:** LangChain, PyMuPDF

## 📂 Project Structure

```text
RAG_Project/
├── AI_Engine/
│   ├── Ingestion_Pipeline2.py       # Handles document chunking, embedding, and upserting
│   ├── LLM_Rag_implementation2.py   # Core RAG logic, Groq integration, and Mermaid prompt builder
│   └── datas/                       # Stores the local bm25_encoder.json
├── Backend/
│   └── backend2.py                  # FastAPI server endpoints (/api/ask, /api/ingest)
├── Frontend/
│   └── app2.py                      # Streamlit user interface
├── data/                            # Raw PDF study materials (Ignored in Git)
├── .env                             # Secure API keys (Ignored in Git)
├── .gitignore                       # Git exclusion rules
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation

## ⚙️ Local Setup & Installation

Follow these steps to run the project locally:

**1. Clone the repository**
git clone [https://github.com/Apparition_010_/MNNIT-Study-Mate.git](https://github.com/Apparition_010_/MNNIT-Study-Mate.git)
cd MNNIT-Study-Mate
2. Create and activate a virtual environment
python -m venv myvenv
myvenv\Scripts\activate
3. Install dependencies
pip install -r requirements.txt
4. Setup Environment Variables
Create a .env file in the root directory and add your API keys securely:
GEMINI_API_KEY="your_gemini_api_key"
PINECONE_API_KEY="your_pinecone_api_key"
GROQ_API_KEY="your_groq_api_key"
5. Start the FastAPI Backend
python Backend/backend2.py
6. Start the Streamlit Frontend
streamlit run Frontend/app2.py
