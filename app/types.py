"""Centralized type definitions for the application."""

from typing import TypeAlias

# Metadata type used across vector store, RAG, and memory
# Corresponds to a mutable dictionary, distinct from ChromaDB's immutable Mapping
Metadata: TypeAlias = dict[str, str | int | float | bool | None]
