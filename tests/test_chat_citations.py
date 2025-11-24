import pytest
from app.core.chat_service import ChatService
from app.db.memory import ChatRepository
from app.rag.service import RAGService
from app.db.vector import ChromaVectorStore


@pytest.mark.integration
def test_system_prompt_contains_citation_instructions_and_model_complies(
    tmp_path, settings
):
    # Setup Real Components
    store = ChromaVectorStore(
        collection_name="test_citations_real",
        persist_directory=str(tmp_path / "chroma_db"),
    )

    # Ingest knowledge
    # We add a specific fact that the model is unlikely to guess exactly this way without context,
    # or just a standard fact but we want to see the citation.
    texts = ["The capital city of the hidden island of Lemuria is Zaramoth."]
    metadatas = [{"source": "geography_book.pdf"}]
    store.add_documents(texts, metadatas)

    rag_service = RAGService(vector_store=store)
    repo = ChatRepository(db_path=":memory:")

    chat_service = ChatService(repo=repo, rag_service=rag_service, settings=settings)

    # Act
    # Ask for the fact. The RAG should retrieve it.
    gen = chat_service.get_response("What is the capital of Lemuria?")

    # Collect response
    response = "".join([chunk for chunk in gen if isinstance(chunk, str)])

    # Assert
    # We verify the model found the info and cited it.
    assert "Zaramoth" in response
    # The system prompt instructs to use [1].
    # So we expect "[1]" in the output.
    assert "[1]" in response
