from typing import Protocol


class RAGService(Protocol):
    def ingest(self, path: str) -> int:
        """Ingests a document and returns the number of chunks created."""
        ...

    def retrieve(self, query: str) -> str:
        """Retrieves relevant context for a given query."""
        ...
