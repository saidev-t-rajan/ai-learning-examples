import hashlib
from collections.abc import Sequence
from typing import cast
import chromadb
from chromadb.api import ClientAPI
from chromadb.api.types import Metadata as ChromaMetadata, QueryResult
from app.rag import embeddings
from app.types import Metadata


class ChromaVectorStore:
    """ChromaDB-based vector store for document embeddings.

    Note: This implementation converts ChromaDB's immutable metadata Mappings
    to mutable dicts on retrieval, which may not support all ChromaDB types
    (e.g. SparseVector).
    """

    client: ClientAPI
    collection_name: str

    def __init__(
        self,
        persist_directory: str,
        collection_name: str = "documents",
    ) -> None:
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection_name = collection_name

    def add_documents(
        self,
        texts: list[str],
        metadatas: Sequence[ChromaMetadata] | None = None,
    ) -> None:
        """
        Add documents to the store.
        Args:
            texts: List of string content.
            metadatas: Optional list of metadata mappings.
        """
        collection = self.client.get_or_create_collection(name=self.collection_name)
        doc_embeddings = embeddings.embed_documents(texts)

        ids = []
        for i, text in enumerate(texts):
            source = ""
            if metadatas and i < len(metadatas):
                # We rely on 'source' being a string or simpler type in metadata
                source = str(metadatas[i].get("source", ""))

            # Create a deterministic ID based on source and content
            # This prevents duplicate entries if the same file is ingested twice
            unique_str = f"{source}::{text}"
            ids.append(hashlib.md5(unique_str.encode("utf-8")).hexdigest())

        collection.upsert(
            documents=texts,
            embeddings=cast(list[Sequence[float]], doc_embeddings),
            metadatas=cast(list[ChromaMetadata], metadatas)
            if metadatas is not None
            else None,
            ids=ids,
        )

    def _process_search_results(
        self, results: QueryResult
    ) -> list[tuple[str, Metadata]]:
        """Extract documents and metadata from Chroma query results."""
        # Handle Optional types explicitly for type safety
        docs_list = results.get("documents")
        if not docs_list or not docs_list[0]:
            return []

        docs = docs_list[0]

        metas_list = results.get("metadatas")
        metas = metas_list[0] if metas_list and metas_list[0] else []

        # Chroma may return None for metas if not set
        if not metas:
            metas = [{}] * len(docs)  # type: ignore

        # Convert Mapping to dict
        return [(doc, cast(Metadata, dict(meta))) for doc, meta in zip(docs, metas)]

    def similarity_search(self, query: str, k: int = 5) -> list[tuple[str, Metadata]]:
        """
        Return top k documents similar to query with their metadata.
        Returns: List[(text, metadata)]
        """
        collection = self.client.get_or_create_collection(name=self.collection_name)
        query_vector = embeddings.embed_query(query)

        results = collection.query(
            query_embeddings=cast(list[Sequence[float]], [query_vector]),
            n_results=k,
        )

        return self._process_search_results(results)
