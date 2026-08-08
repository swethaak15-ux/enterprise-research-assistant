"""
rag.py

Module 4 - Document Research (RAG)

Handles ingestion of uploaded PDF and TXT documents into a persistent
Chroma vector database, and exposes a retriever used by the document-search
tool (tools.py) and the multi-source research pipeline (multi_source.py).
"""

import os
from typing import List

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

embedding_model = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)

_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)


def _load_document(file_path: str) -> List:
    """Load a PDF or TXT file into LangChain Document objects."""
    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".pdf":
        loader = PyPDFLoader(file_path)
    elif extension == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported document type: {extension}. Only .pdf and .txt are supported.")

    return loader.load()


def create_vector_database(file_path: str) -> Chroma:
    """
    Ingest a single PDF or TXT document, split it into chunks, embed it and
    add it to the persistent Chroma collection. Safe to call repeatedly for
    multiple documents - Chroma will append to the existing collection.
    """
    documents = _load_document(file_path)

    for doc in documents:
        doc.metadata["source_file"] = os.path.basename(file_path)

    chunks = _splitter.split_documents(documents)

    db = Chroma(
        persist_directory=config.CHROMA_DB_PATH,
        embedding_function=embedding_model,
    )
    db.add_documents(chunks)

    return db


def load_database() -> Chroma:
    return Chroma(
        persist_directory=config.CHROMA_DB_PATH,
        embedding_function=embedding_model,
    )


def get_retriever(k: int = 4):
    db = load_database()
    return db.as_retriever(search_kwargs={"k": k})


def has_documents() -> bool:
    """Whether any documents have been ingested yet (used to warn the user in the UI)."""
    if not os.path.isdir(config.CHROMA_DB_PATH):
        return False
    try:
        db = load_database()
        return db._collection.count() > 0
    except Exception:
        return False


def list_ingested_sources() -> List[str]:
    """Return the distinct source filenames currently in the knowledge base."""
    if not has_documents():
        return []
    db = load_database()
    data = db.get(include=["metadatas"])
    sources = {meta.get("source_file") for meta in data.get("metadatas", []) if meta.get("source_file")}
    return sorted(sources)
