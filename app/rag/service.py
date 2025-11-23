from app.rag.loader import DocumentLoader
from app.rag.splitter import RecursiveCharacterTextSplitter
from app.rag.embeddings import LocalEmbeddings
from app.db.vector import ChromaVectorStore
from app.core.utils import time_execution


class RAGService:
    def __init__(self, vector_store=None):
        self.loader = DocumentLoader()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
            separators=["\n\n", "\n", " ", ""],
        )
        self.embeddings = LocalEmbeddings()
        # Initialize vector store (which handles persistence)
        self.vector_store = vector_store if vector_store else ChromaVectorStore()

    def ingest(self, path: str) -> int:
        """
        Ingests a document from the given path.
        Returns the number of chunks stored.
        """
        # 1. Load
        text = self.loader.load(path)

        # 2. Split
        raw_chunks = self.splitter.split_text(text)

        if not raw_chunks:
            return 0

        # 3. Context Enrichment & Embed/Store
        enriched_chunks = []
        metadatas = []

        for chunk in raw_chunks:
            # Store chunks without prefix to maximize embedding quality
            # Source information is preserved in metadata
            enriched_chunks.append(chunk)
            metadatas.append({"source": path})

        self.vector_store.add_documents(
            texts=enriched_chunks,
            metadatas=metadatas,
            embedding_service=self.embeddings,
        )

        return len(enriched_chunks)

    @time_execution(threshold=0.3)
    def retrieve(self, query: str) -> str:
        """
        Retrieves context relevant to the query.
        """
        # 4. Search
        # We retrieve top 10 results for now
        results = self.vector_store.similarity_search(
            query=query, k=10, embedding_service=self.embeddings
        )

        formatted_results = []
        for i, item in enumerate(results, 1):
            # Handle both string (legacy) and tuple (new) formats
            if isinstance(item, tuple):
                text, metadata = item
                source = metadata.get("source", "Unknown") if metadata else "Unknown"
                formatted_results.append(f"[{i}] (Source: {source})\n{text}")
            else:
                formatted_results.append(item)

        # Join results with some separation
        return "\n\n".join(formatted_results)
