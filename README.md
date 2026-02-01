# PDF Semantic Search & RAG System (Endee-Compatible)

## 🔍 Project Overview

This project implements a **production-oriented Retrieval-Augmented Generation (RAG) system**
for querying PDF-based knowledge using an **Endee-compatible vector database abstraction**.

The system is designed with a strong focus on:
- Vector similarity search  
- Retrieval correctness  
- Hallucination prevention  

These are core challenges in **real-world AI infrastructure and semantic search systems**.

---

## 🧠 System Design & Workflow

The system follows a **vector-database-centric architecture**, inspired by Endee’s design principles:

1. PDF documents are ingested and converted into semantic vectors  
2. Vectors are stored in a **decoupled vector store layer**  
3. User queries are embedded using the same embedding model  
4. Similarity search retrieves the most relevant chunks  
5. Only **high-confidence results** are passed to the language model  
6. Guardrails ensure answers are grounded strictly in retrieved content  

This design mirrors how modern AI systems integrate vector databases into RAG pipelines.

---

## 🧩 Endee Compatibility & Design Alignment

This project is **explicitly designed around an Endee-compatible vector database abstraction**.

- The vector store exposes `upsert()` and `search()` interfaces that mirror Endee-style APIs  
- Similarity scoring, thresholding, and retrieval logic are implemented independently of the application layer  
- The storage backend is fully decoupled from the RAG pipeline logic  

As a result, the current in-memory vector store can be replaced with a **production Endee backend
(SDK or REST API)** with **no changes** to the core RAG logic.

This reflects real-world AI infrastructure patterns used in scalable semantic search and retrieval systems.

---

## 🚫 Hallucination Prevention Strategy

The system includes multiple safeguards to prevent hallucinated answers:

- Similarity score thresholding to reject weak or irrelevant retrievals  
- Keyword-to-context validation before invoking the LLM  
- Prompt-level constraints enforcing **PDF-only answers**  
- Hard fallback responses when context relevance is insufficient  

If the answer cannot be grounded in retrieved PDF content, the system safely responds:

> **"I could not find the answer in the provided PDF."**

---

## 🛠️ Technology Stack

| Component       | Technologies Used |
|----------------|-------------------|
| Language Model | GCP Vertex AI PaLM (text-bison) / HuggingFace models |
| Embeddings     | Sentence Transformers, Vertex AI (textembedding-gecko) |
| Vector Store   | Endee-Compatible In-Memory Vector Store |
| Frameworks     | LangChain, Chainlit, PyPDF |
| Language       | Python |

> **Note:** Local vector storage is used for evaluation.  
> The architecture is designed for direct replacement with Endee’s production vector database.

---

## 📂 Project Structure

```text
pdf-to-embedding-to-search/
├── app.py              # Core RAG pipeline logic
├── chatbot.py          # Chainlit-powered conversational UI
├── app.cfg             # Configuration for models and vector store
├── requirements.txt    # Project dependencies
├── input/              # Source PDF documents
├── output/             # Processed embeddings and metadata
└── test/
    └── bulktest.py     # Retrieval and accuracy evaluation script

```
⚙️ Setup & Execution Instructions
1. Clone & Initialize
git clone https://github.com/rachitar29/pdf-to-embedding-to-searching.git
cd pdf-to-embedding-to-searching
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
2. Configure Environment

Create required directories:

mkdir input output

Add PDF documents to the input/ directory.

(Optional) Authenticate GCP if using Vertex AI:

gcloud auth application-default login
3. Run the Application


Run the core pipeline:
```
python app.py
```
Run the interactive chatbot:
```
chainlit run chatbot.py
```
Run evaluation tests:
```
python test/bulktest.py
```
Testing & Evaluation

The project includes a dedicated evaluation script (test/bulktest.py) to assess:

Retrieval quality using similarity scores

LLM response accuracy against ground-truth answers

Impact of chunk size and embedding model selection

This enables iterative tuning of retrieval precision and recall.

🔮 Future Roadmap

Direct integration with Endee’s production vector database via REST or SDK

Hybrid retrieval combining keyword (BM25) and vector search

Latency and scalability benchmarking

Advanced UI for document ingestion and monitoring

📚 References

Retrieval-Augmented Generation (RAG) Research Paper

Endee Vector Database Documentation

Sentence Transformers (Hugging Face)
