import pytest
from app.core.chat_service import ChatService
from app.db.memory import ChatRepository
from app.rag.service import RAGService
from app.db.vector import ChromaVectorStore
from app.rag.embeddings import LocalEmbeddings


@pytest.mark.integration
def test_system_prompt_contains_citation_instructions_and_model_complies(
    tmp_path, settings
):
    # Setup Real Components
    embeddings = LocalEmbeddings()
    store = ChromaVectorStore(
        collection_name="test_citations_real",
        persist_directory=str(tmp_path / "chroma_db"),
    )

    # Ingest knowledge
    # We add a specific fact that the model is unlikely to guess exactly this way without context,
    # or just a standard fact but we want to see the citation.
    texts = ["The secret code for the vault is 1234-XYZ."]
    metadatas = [{"source": "secret_manual.pdf"}]
    store.add_documents(texts, metadatas, embedding_service=embeddings)

    rag_service = RAGService(vector_store=store)
    repo = ChatRepository(db_path=":memory:")

    chat_service = ChatService(repo=repo, rag_service=rag_service, settings=settings)

    # Act
    # Ask for the secret code. The RAG should retrieve it.
    gen = chat_service.get_response("What is the secret code?")

    # Collect response
    response = "".join([chunk for chunk in gen if isinstance(chunk, str)])

    # Assert
    # We verify the model found the info and cited it.
    assert "1234-XYZ" in response
    # The citation format in RAGService is "[1] (Source: ...)"
    # The system prompt instructs to use [1].
    # So we expect "[1]" in the output.
    assert "[1]" in response
