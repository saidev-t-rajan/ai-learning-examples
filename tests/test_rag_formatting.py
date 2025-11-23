from unittest.mock import MagicMock
from app.rag.service import RAGService
from app.db.vector import ChromaVectorStore


def test_retrieve_formats_citations():
    # Setup
    mock_store = MagicMock(spec=ChromaVectorStore)
    # Mock return: List of (text, metadata) tuples
    mock_store.similarity_search.return_value = [
        ("The sky is blue.", {"source": "nature.pdf"}),
        ("Roses are red.", {"source": "poetry.txt"}),
    ]

    service = RAGService(vector_store=mock_store)

    # Act
    result = service.retrieve("colors")

    # Assert
    expected_part_1 = "[1] (Source: nature.pdf)\nThe sky is blue."
    expected_part_2 = "[2] (Source: poetry.txt)\nRoses are red."

    assert expected_part_1 in result
    assert expected_part_2 in result
