import sys
import logging
import argparse
from pathlib import Path
from app.core.config import Settings
from app.core.chat_service import ChatService
from app.db.memory import ChatRepository
from app.rag.service import RAGService
from app.cli import CLI
from app.agents.planning import PlanningService
from app.agents.healer import HealerService


def configure_logging(verbose: bool = False) -> None:
    """
    Configure logging to file and stderr.
    File: data/app.log (DEBUG)
    Stderr: INFO if verbose else WARNING
    """
    log_dir = Path("data")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "app.log"

    # Root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # File Handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console Handler (Stderr)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.INFO if verbose else logging.WARNING)
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Chat CLI")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    configure_logging(verbose=args.verbose)

    settings = Settings()
    repo = ChatRepository()
    rag_service = RAGService(settings=settings)
    chat_service = ChatService(repo=repo, rag_service=rag_service, settings=settings)
    planning_service = PlanningService(settings=settings)
    healer_service = HealerService(
        chat_service=chat_service, max_attempts=3, timeout_seconds=30
    )

    cli = CLI(
        chat_service=chat_service,
        rag_service=rag_service,
        settings=settings,
        planning_service=planning_service,
        healer_service=healer_service,
    )
    cli.run()


if __name__ == "__main__":
    main()
