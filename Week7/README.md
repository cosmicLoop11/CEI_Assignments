Week 7: Document Question Answering System (RAG)

An advanced, production-ready Retrieval-Augmented Generation (RAG) system built to answer questions accurately from custom PDF documents. Instead of relying solely on a language model's internal data, this system dynamically extracts knowledge from uploaded files, creates semantic vector embeddings, indexing them into a high-performance vector database, and generates context-grounded responses.

## Key Features & Pipeline Architecture

1. **Document Ingestion:** Automated raw text extraction from unstructured PDFs utilizing pre-compiled high-performance `PyMuPDF` (`fitz`) bindings.
2. **Text Chunking:** Implements a sentence-boundary text-splitting methodology bounded at a logical chunk profile of **800 characters** to preserve contextual semantic cohesion.
3. **Embedding Generation:** Maps unstructured text strings into deep numerical vector spaces using the state-of-the-art **`embed-english-v3.0`** model generating **1024-dimension** arrays.
4. **Vector Database Management:** Real-time indexing and storage within a **Pinecone Serverless Index** configured with Cosine Similarity metrics.
5. **Multi-Stage Advanced Retrieval Optimization:** 
   * **Initial Retrieval:** Queries the vector database to pull the `Top_K=10` most semantically similar text segments.
   * **Reranking Layer:** Leverages **`cohere.rerank-english-v3.0`** to score and filter those blocks down to the absolute `Top_N=3` highest-relevance contexts.
6. **Grounded Generation Engine:** Seamless context-injection into Cohere's active production **`command-r-08-2024`** model via a real-time streaming pipeline to guarantee factually accurate, hallucination-free answers.

---

## System Metrics Report

* **Text Extraction Engine:** PyMuPDF (v1.24.5)
* **Chunking Strategy:** 800-character sentence-boundary splits
* **Embedding Model:** `embed-english-v3.0` (1024 Dimensions)
* **Vector Store:** Pinecone Serverless (AWS, `us-east-1`, Cosine Metric)
* **Reranker Model:** `rerank-english-v3.0` (Top 10 filtered down to Top 3)
* **Language Model (LLM):** `command-r-08-2024` (Streaming enabled)

---

## Local Setup and Installation

### Prerequisites
Make sure you have Python 3.11+ installed on your machine.

Local Setup and Installation
Prerequisites
Make sure you have Python 3.11 or later installed on your machine.

1. Clone and Navigate to the Directory
git clone 
cd CEI_Assignments/Week\ 7

2. Set Up a Virtual Environment
python -m venv venv

Activate on Windows:
venv\Scripts\activate

Activate on Mac/Linux:
source venv/bin/activate

3. Install Required Dependencies
To bypass local C++ compilation errors and ensure a clean environment configuration setup, install the binary wheels explicitly:
pip install --only-binary :all: pymupdf
pip install -r requirements.txt

4. Environment Variables Configuration
Create a .env file in the Week 7 root directory to manage your access credentials safely (this file is pre-configured to be ignored by Git for security):
COHERE_API_KEY=your_actual_cohere_api_key_here
PINECONE_API_KEY=your_actual_pinecone_api_key_here

5. Run the Application
streamlit run src/app.py

