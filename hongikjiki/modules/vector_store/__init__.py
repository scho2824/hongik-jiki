"""
Vector Store Module for Hongik-Jiki Chatbot

This module provides vector database storage and retrieval functionality.
"""

from .base import VectorStoreBase
from .embeddings import EmbeddingsBase, HuggingFaceEmbeddings, OpenAIEmbeddings, get_embeddings
from .chroma_store import ChromaVectorStore
from .tag_index import TagIndex, TagAwareSearch
from .load import load_vector_store

# Add alias for backward compatibility
JungbubVectorStore = ChromaVectorStore

__all__ = [
    'VectorStoreBase',
    'EmbeddingsBase',
    'HuggingFaceEmbeddings',
    'OpenAIEmbeddings',
    'get_embeddings',
    'ChromaVectorStore',
    'JungbubVectorStore',
    'TagIndex',
    'TagAwareSearch',
    'load_vector_store'
]