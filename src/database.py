import sqlite3
import threading
from typing import Optional, List, Any
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, BaseMessage

class PersistentMemory:
    def __init__(self, db_path: str = "memory.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
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

    def save_message(self, session_id: str, role: str, content: str):
        with self._lock:
            with self.conn:
                self.conn.execute(
                    "INSERT INTO chat_history (session_id, role, content) VALUES (?, ?, ?)",
                    (session_id, role, content)
                )

    def get_history(self, session_id: Optional[str] = None, limit: int = 20) -> List[BaseMessage]:
        with self._lock:
            cursor = self.conn.cursor()
            if session_id:
                cursor.execute(
                    "SELECT session_id, role, content, timestamp FROM chat_history WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                    (session_id, limit)
                )
            else:
                cursor.execute(
                    "SELECT session_id, role, content, timestamp FROM chat_history ORDER BY id DESC LIMIT ?",
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
            else:
                messages.append(HumanMessage(content=content))
        return messages

    def load_history(self, session_id: Optional[str] = None, limit: int = 20) -> List[BaseMessage]:
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
        with self._lock:
            with self.conn:
                self.conn.execute(
                    "INSERT INTO chat_history (session_id, role, content) VALUES (?, ?, ?)",
                    (session_id, "summary", summary)
                )

    def get_all_episodic_summaries(self) -> List[str]:
        """Retrieve all stored session summaries."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("SELECT content FROM chat_history WHERE role = 'summary'")
            rows = cursor.fetchall()
        return [row[0] for row in rows]
