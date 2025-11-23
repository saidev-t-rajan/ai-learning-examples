from typing import List
from sentence_transformers import SentenceTransformer
from app.core.config import Settings


class LocalEmbeddings:
    def __init__(self, settings: Settings = None):
        self.settings = settings or Settings()
        print("Notice: Using local 'all-MiniLM-L6-v2' embeddings.\n")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        embedding = self.model.encode(text)
        return embedding.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
