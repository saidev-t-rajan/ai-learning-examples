import sqlite3
import json
from datetime import datetime, timezone
from contextlib import contextmanager
from collections.abc import Generator
from app.types import Metadata


class ChatRepository:
    _persistent_conn: sqlite3.Connection | None

    def __init__(self, db_path: str = "data/chat.db"):
        self.db_path = db_path
        self._persistent_conn = None

        # Keep in-memory DB open for the lifetime of the object
        if self.db_path == ":memory:":
            self._persistent_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._persistent_conn.row_factory = sqlite3.Row

        self._migrate()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        if self._persistent_conn:
            yield self._persistent_conn
            return

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _migrate(self) -> None:
        with self._get_connection() as conn:
            cur = conn.cursor()

            # Check current version
            cur.execute("PRAGMA user_version")
            version = cur.fetchone()[0]

            # Migration to Version 1
            if version < 1:
                self._migrate_v1(cur)
                cur.execute("PRAGMA user_version = 1")
                conn.commit()

    def _migrate_v1(self, cur: sqlite3.Cursor) -> None:
        """
        Version 1: Initialize the chat_history table.
        """
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                metadata TEXT
            )
        """)

    def add_message(
        self, role: str, content: str, metadata: Metadata | None = None
    ) -> int:
        """
        Persist a message to the database.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        metadata_json = json.dumps(metadata) if metadata else None

        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO chat_history (role, content, timestamp, metadata) VALUES (?, ?, ?, ?)",
                (role, content, timestamp, metadata_json),
            )
            conn.commit()
            assert cur.lastrowid is not None
            return cur.lastrowid

    def get_recent_messages(self, limit: int = 10) -> list[dict[str, str]]:
        """
        Retrieve the most recent messages for LLM context.
        Returns them in chronological order (oldest first).
        """
        with self._get_connection() as conn:
            cur = conn.cursor()
            # Fetch most recent messages (DESC)
            cur.execute(
                """
                SELECT role, content
                FROM chat_history
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cur.fetchall()

            # Reverse to get chronological order
            results = [{"role": row["role"], "content": row["content"]} for row in rows]
            return list(reversed(results))
