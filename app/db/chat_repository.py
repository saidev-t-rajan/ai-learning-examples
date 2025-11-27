import sqlite3
import json
from datetime import datetime, timezone
from contextlib import contextmanager
from collections.abc import Generator
from app.types import Metadata
from app.core.models import ChatMetrics, Feedback, ChatMessage, ChatLogEntry


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
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                metadata TEXT
            )
        """)

    def _get_existing_columns(self, cur: sqlite3.Cursor, table_name: str) -> set[str]:
        """Fetch existing column names from table."""
        cur.execute(f"PRAGMA table_info({table_name})")
        return {row[1] for row in cur.fetchall()}

    def _migrate_v2(self, cur: sqlite3.Cursor) -> None:
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

        existing_columns = self._get_existing_columns(cur, "chat_history")

        for col_name, col_type in columns:
            if col_name not in existing_columns:
                cur.execute(
                    f"ALTER TABLE chat_history ADD COLUMN {col_name} {col_type}"
                )

    def _extract_metric_values(self, metrics: ChatMetrics | None) -> tuple:
        """Extract metric values as tuple for SQL insert."""
        if not metrics:
            return (0, 0, 0.0, 0.0, 0.0, None, None, None)

        rag_success_int = (
            1 if metrics.rag_success else 0 if metrics.rag_success is not None else None
        )

        return (
            metrics.input_tokens,
            metrics.output_tokens,
            metrics.cost,
            metrics.total_latency,
            metrics.ttft,
            metrics.avg_retrieval_distance,
            rag_success_int,
            metrics.response_status,
        )

    def add_message(
        self,
        role: str,
        content: str,
        metadata: Metadata | None = None,
        metrics: ChatMetrics | None = None,
    ) -> int:
        """Add message with optional metrics to chat history."""
        timestamp = datetime.now(timezone.utc).isoformat()
        metadata_json = json.dumps(metadata) if metadata else None
        metric_values = self._extract_metric_values(metrics)

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
                (role, content, timestamp, metadata_json, *metric_values),
            )
            conn.commit()
            assert cur.lastrowid is not None
            return cur.lastrowid

    def get_recent_messages(self, limit: int = 10) -> list[ChatMessage]:
        """Retrieve last N messages in chronological order (oldest first)."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT role, content, timestamp
                FROM chat_history
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cur.fetchall()

            results = [
                ChatMessage(
                    role=row["role"], content=row["content"], timestamp=row["timestamp"]
                )
                for row in rows
            ]
            return list(reversed(results))

    def update_feedback(self, message_id: int, feedback: Feedback) -> None:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE chat_history SET feedback = ? WHERE id = ?",
                (int(feedback), message_id),
            )
            conn.commit()

    def get_assistant_metrics(self, limit: int = 100) -> list[ChatLogEntry]:
        with self._get_connection() as conn:
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

            results = []
            for row in cur.fetchall():
                metrics = ChatMetrics(
                    ttft=row["ttft"],
                    total_latency=row["total_latency"],
                    input_tokens=row["input_tokens"],
                    output_tokens=row["output_tokens"],
                    cost=row["cost"],
                    avg_retrieval_distance=row["avg_retrieval_distance"],
                    rag_success=bool(row["rag_success"])
                    if row["rag_success"] is not None
                    else False,
                    response_status=row["response_status"] or "success",
                )
                results.append(
                    ChatLogEntry(
                        timestamp=row["timestamp"],
                        metrics=metrics,
                        feedback=row["feedback"],
                    )
                )
            return results

    def get_success_breakdown(self) -> dict[str, int]:
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
            if not row:
                return {"full_success": 0, "partial": 0, "error": 0}
            return {
                "full_success": row[0] or 0,
                "partial": row[1] or 0,
                "error": row[2] or 0,
            }
