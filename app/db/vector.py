import uuid
from typing import List, Protocol, Optional
import chromadb


class EmbeddingService(Protocol):
    def embed_documents(self, texts: List[str]) -> List[List[float]]: ...

    def embed_query(self, text: str) -> List[float]: ...


class ChromaVectorStore:
    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: str = "data/chroma_db",
    ):
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

        collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

    def similarity_search(
        self, query: str, k: int = 5, embedding_service: EmbeddingService = None
    ) -> List[str]:
        """
        Return top k documents similar to query.
        """
        if not embedding_service:
            raise ValueError("Embedding service is required for search.")

        collection = self.client.get_collection(name=self.collection_name)

        query_vector = embedding_service.embed_query(query)

        results = collection.query(
            query_embeddings=[query_vector],
            n_results=k,
        )

        # results['documents'] is List[List[str]]
        if results["documents"]:
            return results["documents"][0]
        return []
