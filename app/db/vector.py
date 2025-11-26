import hashlib
from collections.abc import Sequence
from typing import cast
import chromadb
from chromadb.api import ClientAPI
from chromadb.api.types import Metadata as ChromaMetadata, QueryResult
from app.rag import embeddings
from app.types import Metadata


class ChromaVectorStore:
    client: ClientAPI
    collection_name: str

    def __init__(
        self,
        persist_directory: str | None = None,
        collection_name: str = "documents",
        host: str | None = None,
        port: int = 8000,
    ) -> None:
        self.client = self._create_client(persist_directory, host, port)
        self.collection_name = collection_name

    def _create_client(
        self,
        persist_directory: str | None,
        host: str | None,
        port: int,
    ) -> ClientAPI:
        if host:
            return chromadb.HttpClient(host=host, port=port)

        if not persist_directory:
            raise ValueError("persist_directory is required when host is not provided")

        return chromadb.PersistentClient(path=persist_directory)

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
    ) -> list[tuple[str, Metadata, float]]:
        """Extract documents, metadata, and distances from Chroma query results."""
        docs = (results.get("documents") or [[]])[0]
        metas = (results.get("metadatas") or [[]])[0]
        dists = (results.get("distances") or [[]])[0]

        if not docs:
            return []

        return [
            (doc, cast(Metadata, dict(meta or {})), dist)
            for doc, meta, dist in zip(docs, metas, dists)
        ]

    def similarity_search(
        self, query: str, k: int = 5
    ) -> list[tuple[str, Metadata, float]]:
        """
        Return top k documents similar to query with their metadata and distance score.
        Returns: List[(text, metadata, distance)]
        """
        collection = self.client.get_or_create_collection(name=self.collection_name)
        query_vector = embeddings.embed_query(query)

        results = collection.query(
            query_embeddings=cast(list[Sequence[float]], [query_vector]),
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        return self._process_search_results(results)
