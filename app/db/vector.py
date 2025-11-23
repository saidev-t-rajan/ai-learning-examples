import uuid
import os
from typing import List, Protocol, Optional
import chromadb


class EmbeddingService(Protocol):
    def embed_documents(self, texts: List[str]) -> List[List[float]]: ...

    def embed_query(self, text: str) -> List[float]: ...


class ChromaVectorStore:
    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: str = None,
    ):
        if persist_directory is None:
            # Default to <project_root>/data/chroma_db
            # This file is in app/db/vector.py -> ../../data/chroma_db
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            persist_directory = os.path.join(project_root, "data", "chroma_db")

        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection_name = collection_name

    def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        embedding_service: EmbeddingService = None,
    ):
        """
        Add documents to the store.
        Args:
            texts: List of string content.
            metadatas: Optional list of metadata dicts.
            embedding_service: Service to generate embeddings.
        """
        if not embedding_service:
            raise ValueError("Embedding service is required to add documents.")

        collection = self.client.get_or_create_collection(name=self.collection_name)

        embeddings = embedding_service.embed_documents(texts)

        # Generate IDs if not provided in metadata (or just random UUIDs for now)
        # Chroma requires 'ids' list.
        ids = [str(uuid.uuid4()) for _ in texts]

        # Chroma/SQLite has a limit on the number of parameters in a query.
        # We process in batches to avoid "Batch size is greater than max batch size" errors.
        batch_size = 5000
        total_docs = len(texts)

        for i in range(0, total_docs, batch_size):
            end_idx = min(i + batch_size, total_docs)

            batch_texts = texts[i:end_idx]
            batch_embeddings = embeddings[i:end_idx]
            batch_ids = ids[i:end_idx]
            batch_metadatas = metadatas[i:end_idx] if metadatas else None

            collection.add(
                documents=batch_texts,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
                ids=batch_ids,
            )

    def similarity_search(
        self, query: str, k: int = 5, embedding_service: EmbeddingService = None
    ) -> List[tuple[str, dict]]:
        """
        Return top k documents similar to query with their metadata.
        Returns: List[(text, metadata)]
        """
        if not embedding_service:
            raise ValueError("Embedding service is required for search.")

        # Use get_or_create_collection to avoid crashing if the collection doesn't exist yet
        collection = self.client.get_or_create_collection(name=self.collection_name)

        query_vector = embedding_service.embed_query(query)

        results = collection.query(
            query_embeddings=[query_vector],
            n_results=k,
        )

        # results['documents'] is List[List[str]]
        # results['metadatas'] is List[List[dict]]
        found_docs = []
        if results["documents"] and results["metadatas"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            # Handle case where metas might be None if not stored (though add_documents usually ensures it)
            # Chroma might return None in the list if no metadata?
            # Safest is to zip.
            for doc, meta in zip(docs, metas):
                found_docs.append((doc, meta if meta else {}))
        elif results["documents"]:
            # Fallback if no metadata found
            for doc in results["documents"][0]:
                found_docs.append((doc, {}))

        return found_docs
