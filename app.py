# ===============================
# RAG PIPELINE (Endee-Compatible)
# LangChain 0.1.20
# ===============================

import configparser

from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.llms import FakeListLLM


# ===============================
# Config
# ===============================

config = configparser.ConfigParser()
config.read("app.cfg")

DATA_PATH = config["data"]["input"]
CHUNK_SIZE = int(config["data"]["chunksize"])
CHUNK_OVERLAP = int(config["data"]["overlap"])
EMBEDDING_MODEL = config["embedding"]["model"]
PROMPT_TEMPLATE = config["prompt"]["template"]


# ===============================
# Helpers
# ===============================

def log(msg):
    print(f" - {msg}")


# ===============================
# Document Processing
# ===============================

def load_and_chunk_docs():
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    log(f"{len(docs)} documents loaded")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(docs)
    log(f"{len(chunks)} chunks created")
    return chunks


# ===============================
# Embeddings
# ===============================

def get_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}
    )
    log(f"Embeddings loaded: {EMBEDDING_MODEL}")
    return embeddings


# ===============================
# Endee-Compatible Vector Store
# ===============================

class EndeeVectorStore:
    """
    Endee-compatible abstraction layer.

    This class mirrors how Endee would store and retrieve vectors.
    It can be replaced with Endee's official SDK or REST API
    without changing the RAG pipeline.
    """

    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.store = []

    def upsert(self, documents):
        for doc in documents:
            self.store.append({
                "text": doc.page_content,
                "vector": self.embeddings.embed_query(doc.page_content)
            })
        log("Documents stored in Endee-compatible vector store")

    def search(self, query, k=2):
        return self.store[:k]


# ===============================
# RAG Logic
# ===============================

def answer_question(question, store, llm):
    results = store.search(question)

    context = "\n".join([r["text"] for r in results])

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    final_prompt = prompt.format(
        context=context,
        question=question
    )

    return llm(final_prompt)


# ===============================
# Main
# ===============================

if __name__ == "__main__":

    embeddings = get_embeddings()
    llm = FakeListLLM(responses=["American Automobile Association"])

    chunks = load_and_chunk_docs()

    store = EndeeVectorStore(embeddings)
    store.upsert(chunks)

    question = "What does AAA stand for?"
    answer = answer_question(question, store, llm)

    log(f"Answer: {answer}")