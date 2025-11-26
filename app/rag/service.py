from typing import cast
from collections.abc import Generator
from chromadb.api.types import Metadata as ChromaMetadata
from app.types import Metadata
from app.rag.loader import load_document
from app.rag.splitter import split_text
from app.db.vector import ChromaVectorStore
from app.core.config import Settings
from app.core.utils import validate_directory_path
from app.core.models import RetrievalResult
import time
import logging

logger = logging.getLogger(__name__)

RAG_CONTEXT_TEMPLATE = (
    "\n\nUse the following context to answer the question.\n"
    "Answer using ONLY the following context. Cite sources using the format [1], [2], etc.\n\n"
    "{context}"
)
RAG_DISTANCE_THRESHOLD = 1.0


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

    def retrieve_context(self, query: str) -> RetrievalResult:
        """
        Retrieve relevant context for the query, format it for the LLM,
        and calculate success metrics.
        """
        # 1. Retrieve raw results
        results = self.retrieve(query)

        # 2. Calculate Metrics
        distances = [score for _, _, score in results]
        avg_distance = sum(distances) / len(distances) if distances else None
        rag_success = avg_distance is not None and avg_distance < RAG_DISTANCE_THRESHOLD

        # 3. Format Context
        formatted_context = ""
        if results:
            formatted_chunks = "\n\n".join(
                f"[{i}] (Source: {meta.get('source', 'Unknown')})\n{text}"
                for i, (text, meta, _) in enumerate(results, 1)
            )
            formatted_context = RAG_CONTEXT_TEMPLATE.format(context=formatted_chunks)

        return RetrievalResult(
            formatted_context=formatted_context,
            avg_distance=avg_distance,
            is_success=rag_success,
        )

    def ingest_directory(
        self, directory_path: str
    ) -> Generator[tuple[str, int], None, None]:
        """
        Ingests all supported documents from a directory.
        Yields tuples of (filename, chunk_count).
        """
        valid_dir = validate_directory_path(directory_path)

        files = sorted(
            [
                f.name
                for f in valid_dir.iterdir()
                if f.is_file() and f.suffix.lower() in (".txt", ".pdf")
            ]
        )

        for filename in files:
            filepath = valid_dir / filename
            chunks = self.ingest(str(filepath))
            yield (filename, chunks)

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

    def retrieve(self, query: str) -> list[tuple[str, Metadata, float]]:
        """
        Retrieves context relevant to the query.
        Returns: List[(text, metadata, distance)]
        """
        start_time = time.time()

        # 4. Search
        # We retrieve top 10 results for now
        results = self.vector_store.similarity_search(query=query, k=10)

        duration = time.time() - start_time
        if duration > 0.3:
            logger.warning(f"Retrieval took {duration:.2f}s")

        return results
