import pytest

# These imports will fail initially, which is expected
try:
    from app.rag.embeddings import LocalEmbeddings
    from app.db.vector import ChromaVectorStore
except ImportError:
    pass


@pytest.mark.integration
def test_local_embeddings_and_chroma_integration():
    # 1. Initialize Embeddings (Local)
    embeddings = LocalEmbeddings()

    # 2. Initialize Vector Store (Real Chroma)
    # We use a temporary collection for testing to avoid polluting data
    store = ChromaVectorStore(collection_name="test_integration_collection")

    # 3. Add Documents
    texts = ["Apple is a fruit", "Car is a vehicle"]
    metadatas = [{"category": "food"}, {"category": "machine"}]

    # This should internally call embeddings.embed_documents(texts)
    store.add_documents(texts, metadatas, embedding_service=embeddings)

    # 4. Search
    query = "fruit"
    # This should internally call embeddings.embed_query(query)
    results = store.similarity_search(query, k=1, embedding_service=embeddings)

    # 5. Assert
    assert len(results) == 1
    # Check that we get a tuple (text, metadata)
    assert isinstance(results[0], tuple)
    assert results[0][0] == "Apple is a fruit"
    # Optional: verify metadata too
    assert results[0][1] == {"category": "food"}

    # Cleanup (optional, but good for local usage)
    try:
        store.client.delete_collection("test_integration_collection")
    except Exception:
        pass
