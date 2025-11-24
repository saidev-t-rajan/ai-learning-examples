import os
import sys

# Ensure app is in path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))

from app.rag.service import RAGService
from app.main import ingest_directory_with_report

CORPUS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "corpus")


def main():
    print("Initializing RAG Service...")
    service = RAGService()

    ingest_directory_with_report(service, CORPUS_DIR)


if __name__ == "__main__":
    main()
