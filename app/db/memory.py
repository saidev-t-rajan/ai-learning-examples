import sqlite3
import json
from datetime import datetime, timezone
from contextlib import contextmanager
from collections.abc import Generator
from app.types import Metadata
from app.core.models import ChatMetrics


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

            # Migration to Version 2
            if version < 2:
                self._migrate_v2(cur)
                cur.execute("PRAGMA user_version = 2")
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

    def _migrate_v2(self, cur: sqlite3.Cursor) -> None:
        """
        Version 2: Add metrics columns for dashboard.
        """
        columns = [
            ("input_tokens", "INTEGER"),
            ("output_tokens", "INTEGER"),
            ("cost", "REAL"),
            ("total_latency", "REAL"),
            ("ttft", "REAL"),
            ("avg_retrieval_distance", "REAL"),
            ("rag_success", "INTEGER"),
            ("response_status", "TEXT"),
            ("feedback", "INTEGER"),
        ]
        for col_name, col_type in columns:
            try:
                cur.execute(
                    f"ALTER TABLE chat_history ADD COLUMN {col_name} {col_type}"
                )
            except sqlite3.OperationalError:
                # Column likely already exists if migration ran partially or manually
                pass

    def add_message(
        self,
        role: str,
        content: str,
        metadata: Metadata | None = None,
        metrics: ChatMetrics | None = None,
    ) -> int:
        """
        Persist a message to the database with optional metrics.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        metadata_json = json.dumps(metadata) if metadata else None

        # Extract metrics if provided
        input_tokens = metrics.input_tokens if metrics else 0
        output_tokens = metrics.output_tokens if metrics else 0
        cost = metrics.cost if metrics else 0.0
        total_latency = metrics.total_latency if metrics else 0.0
        ttft = metrics.ttft if metrics else 0.0
        avg_retrieval_distance = metrics.avg_retrieval_distance if metrics else None
        rag_success = metrics.rag_success if metrics else None
        response_status = metrics.response_status if metrics else None

        # Convert bool to int for SQLite
        rag_success_int = 1 if rag_success else 0 if rag_success is not None else None

        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO chat_history (
                    role, content, timestamp, metadata,
                    input_tokens, output_tokens, cost, total_latency, ttft,
                    avg_retrieval_distance, rag_success, response_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    role,
                    content,
                    timestamp,
                    metadata_json,
                    input_tokens,
                    output_tokens,
                    cost,
                    total_latency,
                    ttft,
                    avg_retrieval_distance,
                    rag_success_int,
                    response_status,
                ),
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

    def update_feedback(self, message_id: int, feedback: int) -> None:
        """
        Update feedback for a specific message.
        feedback: 1 (up), -1 (down), 0 (neutral/removed)
        """
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE chat_history SET feedback = ? WHERE id = ?",
                (feedback, message_id),
            )
            conn.commit()

    def get_assistant_metrics(self, limit: int = 100) -> list[dict]:
        """Get assistant message metrics for dashboard."""
        with self._get_connection() as conn:
            # Ensure row_factory is set for this connection if created fresh
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(
                """
                SELECT timestamp, total_latency, ttft, cost, input_tokens, output_tokens,
                       avg_retrieval_distance, rag_success, response_status, feedback
                FROM chat_history
                WHERE role = 'assistant'
                ORDER BY timestamp DESC
                LIMIT ?
            """,
                (limit,),
            )
            return [dict(row) for row in cur.fetchall()]

    def get_success_breakdown(self) -> dict[str, int]:
        """Get success/partial/error counts for pie chart."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT
                    SUM(CASE WHEN response_status = 'success' AND rag_success = 1 THEN 1 ELSE 0 END) as full_success,
                    SUM(CASE WHEN response_status = 'success' AND (rag_success = 0 OR rag_success IS NULL) THEN 1 ELSE 0 END) as partial,
                    SUM(CASE WHEN response_status LIKE 'error:%' THEN 1 ELSE 0 END) as error
                FROM chat_history
                WHERE role = 'assistant'
            """)
            row = cur.fetchone()
            # Handle case where table is empty (returns None)
            if not row:
                return {"full_success": 0, "partial": 0, "error": 0}
            return {
                "full_success": row[0] or 0,
                "partial": row[1] or 0,
                "error": row[2] or 0,
            }
