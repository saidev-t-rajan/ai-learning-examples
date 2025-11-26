import os
import sys
import argparse

# Ensure app is in path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))

from app.rag.service import RAGService
from app.cli import ingest_directory_with_report
from app.core.config import Settings


def main():
    parser = argparse.ArgumentParser(description="Ingest corpus documents.")
    parser.add_argument(
        "--large",
        action="store_true",
        help="Use the large corpus directory instead of the default.",
    )
    args = parser.parse_args()

    print("Initializing RAG Service...")
    settings = Settings()
    service = RAGService(settings=settings)

    target_dir = settings.CORPUS_LARGE_DIR if args.large else settings.CORPUS_DIR

    # Ensure we are using absolute path if running from script location logic was desired,
    # but Settings defaults are relative.
    # To be safe and DRY with previous logic (which used __file__), we can construct it
    # relative to the project root if needed, but usually running from root is expected.
    # However, the original script had:
    # CORPUS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "corpus")
    # If we rely on Settings default "data/corpus", it works if CWD is project root.
    # Let's assume CWD is project root or we might need to adjust if the user runs it from elsewhere.
    # Given the "dry" instruction, relying on Settings is better than hardcoding paths again.

    ingest_directory_with_report(service, target_dir)


if __name__ == "__main__":
    main()
