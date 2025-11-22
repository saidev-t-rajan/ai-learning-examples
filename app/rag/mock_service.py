class MockRAGService:
    def __init__(self):
        self.last_query = None

    def ingest(self, path: str) -> int:
        print(f"Simulating ingest for {path}...")
        return 5

    def retrieve(self, query: str) -> str:
        self.last_query = query
        return "Context: This is the retrieved context."
