from unittest.mock import patch, Mock
from app.main import _handle_command, INGEST_ALL
from app.core.config import Settings


def test_handle_command_ingest_all_default():
    rag_service = Mock()
    settings = Settings()

    with patch("app.main.ingest_directory_with_report") as mock_ingest:
        _handle_command(INGEST_ALL, rag_service, settings)
        mock_ingest.assert_called_once_with(rag_service, settings.CORPUS_DIR)


def test_handle_command_ingest_all_large():
    rag_service = Mock()
    settings = Settings()

    with patch("app.main.ingest_directory_with_report") as mock_ingest:
        _handle_command(f"{INGEST_ALL} --large", rag_service, settings)
        mock_ingest.assert_called_once_with(rag_service, settings.CORPUS_LARGE_DIR)
