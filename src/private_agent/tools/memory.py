"""Tools for reading and deleting SQLite conversation history."""

import sqlite3

from langchain_core.tools import tool

from ..config import DEFAULT_HISTORY_READ_LIMIT
from . import _sqlite_lock, _sqlite_state
from .schemas import DeleteSqliteHistoryInput, ReadSqliteHistoryInput


@tool(args_schema=ReadSqliteHistoryInput)
def read_chat_history_from_sqlite(limit: int = DEFAULT_HISTORY_READ_LIMIT) -> str:
    """Read recent conversation entries from persistent SQLite memory."""
    try:
        with _sqlite_lock:
            with sqlite3.connect(
                _sqlite_state.ACTIVE_DB_PATH,
                check_same_thread=False,
            ) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT session_id, role, content, timestamp "
                    "FROM chat_history ORDER BY id DESC LIMIT ?",
                    (limit,),
                )
                rows = cursor.fetchall()
        if not rows:
            return "SQLite memory chat history is empty."
        return "\n".join(
            f"[{timestamp}] Session: {session_id} | {role}: {content}"
            for session_id, role, content, timestamp in reversed(rows)
        )
    except Exception as exc:
        return f"Error reading SQLite history: {exc}"


@tool(args_schema=DeleteSqliteHistoryInput)
def delete_chat_history_from_sqlite(session_id: str | None = None) -> str:
    """Delete stored SQLite chat history for one session or all sessions."""
    try:
        with _sqlite_lock:
            with sqlite3.connect(
                _sqlite_state.ACTIVE_DB_PATH,
                check_same_thread=False,
            ) as conn:
                if session_id:
                    cursor = conn.cursor()
                    cursor.execute(
                        "DELETE FROM chat_history WHERE session_id = ?",
                        (session_id,),
                    )
                    rows_affected = cursor.rowcount
                    return (
                        f"Success: Deleted {rows_affected} messages for session "
                        f"'{session_id}' from SQLite memory."
                    )
                conn.execute("DELETE FROM chat_history")
                return "Success: Wiped all chat history rows from SQLite persistent database."
    except Exception as exc:
        return f"Error deleting SQLite history: {exc}"
