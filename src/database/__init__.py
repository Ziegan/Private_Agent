import sqlite3
import threading
import pathlib
from datetime import datetime, timedelta, timezone
from typing import Optional, List
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, BaseMessage
from ..config import (
    DEFAULT_DB_PATH,
    DEFAULT_EPISODIC_SUMMARIES,
    DEFAULT_HISTORY_MESSAGES,
    MAX_SUMMARY_CHARS,
)

class PersistentMemory:
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = str(pathlib.Path(db_path).expanduser()) if db_path != ":memory:" else db_path
        self._lock = threading.Lock()
        if self.db_path != ":memory:":
            pathlib.Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._closed = False
        self._create_table()

    def _create_table(self):
        with self._lock:
            with self.conn:
                self.conn.execute("""
                    CREATE TABLE IF NOT EXISTS chat_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT,
                        role TEXT,
                        content TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                self.conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_chat_history_session_id ON chat_history(session_id, id)"
                )

    def save_message(self, session_id: str, role: str, content: str):
        with self._lock:
            with self.conn:
                self.conn.execute(
                    "INSERT INTO chat_history (session_id, role, content) VALUES (?, ?, ?)",
                    (session_id, role, content)
                )

    def get_history(
        self,
        session_id: Optional[str] = None,
        limit: int = DEFAULT_HISTORY_MESSAGES,
    ) -> List[BaseMessage]:
        with self._lock:
            cursor = self.conn.cursor()
            if session_id:
                cursor.execute(
                    "SELECT session_id, role, content, timestamp FROM chat_history WHERE session_id = ? AND role != 'summary' ORDER BY id DESC LIMIT ?",
                    (session_id, limit)
                )
            else:
                cursor.execute(
                    "SELECT session_id, role, content, timestamp FROM chat_history WHERE role != 'summary' ORDER BY id DESC LIMIT ?",
                    (limit,)
                )
            rows = cursor.fetchall()
        
        messages = []
        for r in reversed(rows):
            role = r[1].lower()
            content = r[2]
            if role in ("human", "user"):
                messages.append(HumanMessage(content=content))
            elif role in ("ai", "assistant"):
                messages.append(AIMessage(content=content))
            elif role == "system":
                messages.append(SystemMessage(content=content))
            elif role == "tool":
                messages.append(ToolMessage(content=content, tool_call_id="unknown"))
            elif role == "summary":
                continue
            else:
                messages.append(HumanMessage(content=content))
        return messages

    def get_latest_session_id(self) -> Optional[str]:
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT session_id FROM chat_history WHERE role != 'summary' "
                "AND session_id IS NOT NULL ORDER BY id DESC LIMIT 1"
            )
            row = cursor.fetchone()
        return row[0] if row else None

    def list_sessions(self) -> List[str]:
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT session_id FROM chat_history WHERE role != 'summary' "
                "AND session_id IS NOT NULL GROUP BY session_id ORDER BY MAX(id) DESC"
            )
            rows = cursor.fetchall()
        return [row[0] for row in rows]

    def load_history(
        self,
        session_id: Optional[str] = None,
        limit: int = DEFAULT_HISTORY_MESSAGES,
    ) -> List[BaseMessage]:
        """Alias for get_history to support test expectations."""
        return self.get_history(session_id, limit)

    def clear_history(self, session_id: Optional[str] = None):
        with self._lock:
            with self.conn:
                if session_id:
                    self.conn.execute("DELETE FROM chat_history WHERE session_id = ?", (session_id,))
                else:
                    self.conn.execute("DELETE FROM chat_history")

    def save_summary(self, session_id: str, summary: str):
        """Save an episodic session summary to the database."""
        summary = str(summary).strip()[:MAX_SUMMARY_CHARS]
        if not summary:
            return
        with self._lock:
            with self.conn:
                self.conn.execute(
                    "INSERT INTO chat_history (session_id, role, content) VALUES (?, ?, ?)",
                    (session_id, "summary", summary)
                )

    def prune_history(self, retention_days: int) -> int:
        """Delete messages and derived summaries older than the retention window."""
        if retention_days <= 0:
            return 0
        cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).isoformat(
            sep=" ", timespec="seconds"
        )
        with self._lock:
            with self.conn:
                cursor = self.conn.execute(
                    "DELETE FROM chat_history WHERE timestamp < ?", (cutoff,)
                )
                return cursor.rowcount

    def get_all_episodic_summaries(
        self,
        limit: int = DEFAULT_EPISODIC_SUMMARIES,
        session_id: Optional[str] = None,
    ) -> List[str]:
        """Retrieve recent summaries, optionally restricted to one session."""
        with self._lock:
            cursor = self.conn.cursor()
            if session_id is None:
                cursor.execute(
                    "SELECT content FROM ("
                    "SELECT content, id FROM chat_history WHERE role = 'summary' "
                    "ORDER BY id DESC LIMIT ?"
                    ") ORDER BY id ASC",
                    (max(0, limit),),
                )
            else:
                cursor.execute(
                    "SELECT content FROM ("
                    "SELECT content, id FROM chat_history "
                    "WHERE role = 'summary' AND session_id = ? "
                    "ORDER BY id DESC LIMIT ?"
                    ") ORDER BY id ASC",
                    (session_id, max(0, limit)),
                )
            rows = cursor.fetchall()
        return [row[0] for row in rows]

    def close(self):
        with self._lock:
            if not self._closed:
                self.conn.close()
                self._closed = True
