"""Shared active SQLite path and lock for history tools."""

import os
import threading

from ..config import DEFAULT_DB_PATH

ACTIVE_DB_PATH = DEFAULT_DB_PATH
SQLITE_LOCK = threading.Lock()


def set_active_db_path(path: str) -> None:
    global ACTIVE_DB_PATH
    ACTIVE_DB_PATH = os.path.abspath(path)
