import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from app.rag.service import RAGService


def debug_query():
    service = RAGService()
    query = "What is the name of the ship in Moby Dick?"
    print(f"Query: {query}")

    # We can access the vector store directly to get scores if we wanted,
    # but let's just use retrieve and print EVERYTHING.
    result = service.retrieve(query)
    print("--- Result ---")
    print(result)
    print("--------------")

    # Check for Pequod
    if "pequod" in result.lower():
        print("FOUND 'Pequod'")
    else:
        print("NOT FOUND 'Pequod'")


if __name__ == "__main__":
    debug_query()
