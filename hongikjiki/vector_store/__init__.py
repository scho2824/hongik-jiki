"""
Vector Store Module for Hongik-Jiki Chatbot

This module provides vector database storage and retrieval functionality.
"""

from .base import VectorStoreBase
from .embeddings import EmbeddingsBase, HuggingFaceEmbeddings, OpenAIEmbeddings, get_embeddings
from .chroma_store import ChromaVectorStore
from .tag_index import TagIndex, TagAwareSearch

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
    'TagAwareSearch'
]