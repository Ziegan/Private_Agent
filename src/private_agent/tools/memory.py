"""Tools for reading and deleting SQLite conversation history."""

import sqlite3

from langchain_core.tools import tool

from ..config import DEFAULT_HISTORY_READ_LIMIT
from . import _sqlite_lock, _sqlite_state
from .schemas import (
    DeleteSqliteHistoryEntryInput,
    DeleteSqliteHistoryInput,
    ReadSqliteHistoryInput,
    SearchSqliteHistoryInput,
)


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


@tool(args_schema=SearchSqliteHistoryInput)
def search_chat_history_in_sqlite(
    query: str,
    session_id: str | None = None,
    limit: int = 10,
) -> str:
    """Search bounded local conversation history and summaries; results include reviewable entry IDs."""
    try:
        with _sqlite_lock:
            with sqlite3.connect(
                _sqlite_state.ACTIVE_DB_PATH,
                check_same_thread=False,
            ) as conn:
                parameters: list[object] = [query.casefold()]
                session_clause = ""
                if session_id:
                    session_clause = " AND session_id = ?"
                    parameters.append(session_id)
                parameters.append(min(max(limit, 1), 50))
                rows = conn.execute(
                    "SELECT id, session_id, role, timestamp, content "
                    "FROM chat_history WHERE instr(lower(content), ? ) > 0"
                    + session_clause
                    + " ORDER BY id DESC LIMIT ?",
                    parameters,
                ).fetchall()
        if not rows:
            return "No matching SQLite conversation entries."
        return "\n\n".join(
            f"Entry ID: {entry_id} | Session: {row_session_id} | "
            f"Role: {role} | Timestamp: {timestamp}\n"
            f"{str(content)[:1200]}"
            for entry_id, row_session_id, role, timestamp, content in rows
        )
    except Exception as exc:
        return f"Error searching SQLite history: {exc}"


@tool(args_schema=DeleteSqliteHistoryEntryInput)
def delete_chat_history_entry_from_sqlite(entry_id: int, session_id: str) -> str:
    """Delete one locally stored conversation entry identified by search results."""
    try:
        with _sqlite_lock:
            with sqlite3.connect(
                _sqlite_state.ACTIVE_DB_PATH,
                check_same_thread=False,
            ) as conn:
                cursor = conn.execute(
                    "DELETE FROM chat_history WHERE id = ? AND session_id = ?",
                    (entry_id, session_id),
                )
                deleted = cursor.rowcount
        if deleted:
            return f"Success: Deleted SQLite history entry {entry_id} from session '{session_id}'."
        return "No matching SQLite history entry was deleted; verify its ID and session."
    except Exception as exc:
        return f"Error deleting SQLite history entry: {exc}"
