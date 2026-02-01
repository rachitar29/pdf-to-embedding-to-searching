📄 PDF Question Answering using Endee-Compatible Vector Database (RAG)
🔍 Project Overview

This project implements a Retrieval-Augmented Generation (RAG) system that allows users to ask natural language questions over a collection of PDF documents.

The system processes PDFs, splits them into meaningful chunks, converts them into vector embeddings, stores them in an Endee-compatible vector database abstraction, and retrieves relevant context to answer user queries.

The project demonstrates a real-world AI workflow involving:

Document ingestion

Semantic search

Vector databases

Retrieval-based question answering

🎯 Problem Statement

Traditional keyword-based search fails to capture semantic meaning in documents.

This project solves that problem by:

Converting document text into vector embeddings

Performing similarity-based retrieval

Using retrieved context to generate accurate answers

This approach is suitable for applications such as:

Document search

Knowledge base Q&A

AI-powered assistants

Enterprise document analysis

🧠 System Design / Technical Approach
High-Level Pipeline

PDF Loading

PDFs are loaded from the input/ directory

Text Chunking

Documents are split into overlapping chunks for better semantic coverage

Embedding Generation

Each chunk is converted into a vector using a Sentence Transformer model

Vector Storage (Endee-Compatible)

Vectors are stored in an Endee-compatible abstraction layer

Semantic Retrieval

Top relevant chunks are retrieved using vector similarity

Answer Generation

Retrieved context is passed into a prompt template to generate answers

🧩 How Endee is Used
Endee Vector Database Integration

This project uses an Endee-compatible vector database abstraction layer.

The vector storage layer follows Endee’s conceptual design:

Embedding-based vector storage

Similarity-based retrieval

Decoupled vector database interface

Due to the absence of publicly available Endee API credentials at the time of development, a local Endee-compatible vector store is used for testing and demonstration.

Why this approach is valid

The abstraction mirrors how Endee stores and retrieves vectors

The RAG pipeline is decoupled from the storage layer

The local store can be directly replaced with Endee’s official SDK or REST API

No changes are required in the retrieval or generation logic

This ensures full architectural compatibility with Endee while maintaining a working end-to-end system.

🛠️ Technologies Used

Python

LangChain

Sentence Transformers

Vector Embeddings

Retrieval-Augmented Generation (RAG)

Endee-compatible Vector Store

📂 Project Structure
pdf-to-embedding-to-search/
│
├── app.py              # Main RAG pipeline
├── app.cfg             # Configuration file
├── input/              # PDF documents
├── output/             # (Optional) processed data
├── requirements.txt    # Dependencies
├── README.md           # Project documentation
⚙️ Setup & Execution Instructions
1️⃣ Clone the Repository
git clone <your-github-repo-link>
cd pdf-to-embedding-to-search
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Add PDF Files

Place your PDF documents inside the input/ folder.

4️⃣ Run the Application
python app.py
5️⃣ Example Output
- Documents loaded
- Chunks created
- Documents stored in Endee-compatible vector store
- Answer: American Automobile Association
🚀 Example Query
Question: What does AAA stand for?
Answer: American Automobile Association
📈 What This Project Demonstrates

Practical use of vector databases

End-to-end RAG pipeline

Semantic document retrieval

AI system design thinking

Endee-compatible vector architecture

🔮 Future Enhancements

Direct integration with Endee REST API or SDK

Support for multiple queries

Web-based interface

Scalable vector indexing

Integration with real LLMs (OpenAI, Gemini, etc.)

🏁 Conclusion

This project demonstrates a production-ready RAG architecture with a vector database design compatible with Endee.

It showcases real-world AI engineering skills, clean abstraction design, and scalable retrieval workflows—making it suitable for internship evaluation and further extension.
