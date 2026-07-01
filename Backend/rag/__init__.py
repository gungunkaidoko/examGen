"""
RAG Package
-----------
Retrieval-Augmented Generation pipeline for the CCC exam platform.

Public surface (import from here):

    from rag import ingest, get_retriever, retrieve_for_chapter

Modules
-------
  embedder       — Amazon Titan Text Embeddings V2 via Bedrock
  ingestion      — Load → chunk → embed → upsert to Pinecone
  pinecone_store — Pinecone index lifecycle management
  retriever      — Similarity + MMR search with metadata filtering
"""

from rag.ingestion import ingest
from rag.retriever import get_retriever, retrieve_for_chapter

__all__ = ["ingest", "get_retriever", "retrieve_for_chapter"]
