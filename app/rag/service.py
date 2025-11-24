from typing import cast
from collections.abc import Generator
from chromadb.api.types import Metadata as ChromaMetadata
from app.types import Metadata
from app.rag.loader import load_document
from app.rag.splitter import split_text
from app.db.vector import ChromaVectorStore
from app.core.config import Settings
import os
import time


class RAGService:
    def __init__(
        self,
        vector_store: ChromaVectorStore | None = None,
        settings: Settings | None = None,
    ):
        self.settings = settings or Settings()
        # Initialize vector store (which handles persistence and embeddings)
        self.vector_store: ChromaVectorStore = (
            vector_store
            if vector_store
            else ChromaVectorStore(persist_directory=self.settings.CHROMA_DB_DIR)
        )

    def ingest_directory(
        self, directory_path: str
    ) -> Generator[tuple[str, int], None, None]:
        """
        Ingests all supported documents from a directory.
        Yields tuples of (filename, chunk_count).
        """
        if not os.path.exists(directory_path):
            raise ValueError(f"Directory not found: {directory_path}")

        files = sorted(
            [
                f
                for f in os.listdir(directory_path)
                if f.lower().endswith((".txt", ".pdf"))
            ]
        )

        for filename in files:
            filepath = os.path.join(directory_path, filename)
            try:
                chunks = self.ingest(filepath)
                yield (filename, chunks)
            except Exception as e:
                print(f"Failed to ingest {filename}: {e}")
                yield (filename, 0)

    def ingest(self, path: str) -> int:
        """
        Ingests a document from the given path.
        Returns the number of chunks stored.
        """
        text = load_document(path)

        raw_chunks = split_text(text, chunk_size=1500, chunk_overlap=300)

        if not raw_chunks:
            return 0

        # Store chunks without prefix to maximize embedding quality
        # Source information is preserved in metadata
        # We type hint as list[ChromaMetadata] to satisfy the VectorStore protocol.
        # Dicts satisfy the Mapping requirement of Metadata.
        metadatas: list[ChromaMetadata] = [
            cast(ChromaMetadata, {"source": path})
        ] * len(raw_chunks)

        self.vector_store.add_documents(
            texts=raw_chunks,
            metadatas=metadatas,
        )

        return len(raw_chunks)

    def retrieve(self, query: str) -> list[tuple[str, Metadata]]:
        """
        Retrieves context relevant to the query.
        """
        start_time = time.time()

        # 4. Search
        # We retrieve top 10 results for now
        results = self.vector_store.similarity_search(query=query, k=10)

        duration = time.time() - start_time
        if duration > 0.3:
            print(f"Warning: Retrieval took {duration:.2f}s")

        return results
