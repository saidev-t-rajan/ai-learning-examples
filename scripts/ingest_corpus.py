import os
import sys
import time

# Ensure app is in path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))

from app.rag.service import RAGService

CORPUS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "corpus")


def main():
    print("Initializing RAG Service...")
    service = RAGService()

    if not os.path.exists(CORPUS_DIR):
        print(f"Corpus directory not found: {CORPUS_DIR}")
        return

    files = [
        f for f in os.listdir(CORPUS_DIR) if f.endswith(".txt") or f.endswith(".pdf")
    ]
    total_files = len(files)
    print(f"Found {total_files} files to ingest.")

    start_time = time.time()
    total_chunks = 0

    for i, filename in enumerate(files, 1):
        filepath = os.path.join(CORPUS_DIR, filename)
        print(f"[{i}/{total_files}] Ingesting {filename}...", end="", flush=True)

        try:
            chunks = service.ingest(filepath)
            total_chunks += chunks
            print(f" Done ({chunks} chunks)")
        except Exception as e:
            print(f" Failed: {e}")

    end_time = time.time()
    duration = end_time - start_time

    print("-" * 30)
    print("Ingestion Complete.")
    print(f"Total Time:   {duration:.2f}s")
    print(f"Total Chunks: {total_chunks}")
    print(f"Avg Speed:    {total_chunks / duration:.1f} chunks/sec")


if __name__ == "__main__":
    main()
