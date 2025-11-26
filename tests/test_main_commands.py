from unittest.mock import patch, Mock
from app.cli import CLI, INGEST_ALL
from app.core.config import Settings


def test_handle_command_ingest_all_default():
    chat_service = Mock()
    rag_service = Mock()
    settings = Settings()

    cli = CLI(chat_service, rag_service, settings, planning_service=None)

    with patch("app.cli.ingest_directory_with_report") as mock_ingest:
        cli._handle_command(INGEST_ALL)
        mock_ingest.assert_called_once_with(rag_service, settings.CORPUS_DIR)


def test_handle_command_ingest_all_large():
    chat_service = Mock()
    rag_service = Mock()
    settings = Settings()

    cli = CLI(chat_service, rag_service, settings, planning_service=None)

    with patch("app.cli.ingest_directory_with_report") as mock_ingest:
        cli._handle_command(f"{INGEST_ALL} --large")
        mock_ingest.assert_called_once_with(rag_service, settings.CORPUS_LARGE_DIR)
