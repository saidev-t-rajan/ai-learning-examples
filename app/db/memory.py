import sqlite3
import json
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any
from contextlib import contextmanager


class ChatRepository:
    def __init__(self, db_path: str = "data/chat.db"):
        self.db_path = db_path
        self._persistent_conn = None

        # Keep in-memory DB open for the lifetime of the object
        if self.db_path == ":memory:":
            self._persistent_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._persistent_conn.row_factory = sqlite3.Row

        self._migrate()

    @contextmanager
    def _get_connection(self):
        if self._persistent_conn:
            yield self._persistent_conn
        else:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
            finally:
                conn.close()

    def _migrate(self):
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

    def _migrate_v1(self, cur):
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
        self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None
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
            return cur.lastrowid

    def get_recent_messages(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve the most recent messages for LLM context.
        Returns them in chronological order (oldest first).
        """
        with self._get_connection() as conn:
            cur = conn.cursor()
            # Get recent messages (newest first)
            cur.execute(
                "SELECT role, content, timestamp, metadata FROM chat_history ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            )
            rows = cur.fetchall()

            messages = []
            for row in rows:
                msg = {
                    "role": row["role"],
                    "content": row["content"],
                    # We can optionally include timestamp/metadata if needed by the LLM,
                    # but usually just role/content is sufficient for the messages list.
                    # The plan implies fetching context for LLM.
                }
                messages.append(msg)

            # Reverse to get chronological order (Oldest -> Newest)
            return messages[::-1]
