from app.rag.loader import PDFLoader
from app.rag.splitter import RecursiveCharacterTextSplitter
from app.rag.embeddings import OpenAIEmbeddings
from app.db.vector import ChromaVectorStore


class RAGService:
    def __init__(self):
        self.loader = PDFLoader()
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        self.embeddings = OpenAIEmbeddings()
        # Initialize vector store (which handles persistence)
        self.vector_store = ChromaVectorStore()

    def ingest(self, path: str) -> int:
        """
        Ingests a document from the given path.
        Returns the number of chunks stored.
        """
        # 1. Load
        text = self.loader.load(path)

        # 2. Split
        chunks = self.splitter.split_text(text)

        if not chunks:
            return 0

        # 3. Embed and Store
        # We attach metadata for potential future use (citations)
        metadatas = [{"source": path} for _ in chunks]

        self.vector_store.add_documents(
            texts=chunks, metadatas=metadatas, embedding_service=self.embeddings
        )

        return len(chunks)

    def retrieve(self, query: str) -> str:
        """
        Retrieves context relevant to the query.
        """
        # 4. Search
        # We retrieve top 3 results for now
        results = self.vector_store.similarity_search(
            query=query, k=3, embedding_service=self.embeddings
        )

        # Join results with some separation
        return "\n\n".join(results)
