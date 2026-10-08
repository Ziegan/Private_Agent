"""SQLite-backed conversation history and episodic memory."""

import sqlite3
import threading
import pathlib
import os
import hashlib
import json
import logging
import re
import tempfile
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Optional, List
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, BaseMessage
from ..config import (
    DEFAULT_DB_PATH,
    DEFAULT_EPISODIC_SUMMARIES,
    DEFAULT_HISTORY_MESSAGES,
    MAX_SUMMARY_CHARS,
    MAX_LEARNED_ITEM_CHARS,
    MAX_RESUME_STATE_CHARS,
    LEARNED_ITEM_EXPIRY_DAYS,
    LEARNED_ITEM_REVIEW_DAYS,
)

_LOGGER = logging.getLogger(__name__)
_SCHEMA_VERSION = 1


class PersistentMemory:
    SCHEMA_TABLE_ALLOWLIST = {
        "chat_history",
        "agent_tasks",
        "agent_plan_revisions",
        "agent_todo_steps",
        "agent_task_events",
        "agent_task_leases",
        "agent_task_actions",
        "agent_learned_items",
        "agent_learning_settings",
    }

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = str(pathlib.Path(db_path).expanduser()) if db_path != ":memory:" else db_path
        self._lock = threading.Lock()
        if self.db_path != ":memory:":
            pathlib.Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            timeout=2.0,
        )
        self._closed = False
        self._episode_fts_enabled = False
        try:
            self.conn.execute("PRAGMA foreign_keys = ON")
            self.conn.execute("PRAGMA busy_timeout = 2000")
            self._create_table()
        except BaseException:
            self.conn.close()
            self._closed = True
            raise

    @contextmanager
    def _write_transaction(self):
        """Acquire the SQLite write lock up front; caller holds the instance lock."""
        try:
            self.conn.execute("BEGIN IMMEDIATE")
            yield
            self.conn.commit()
        except BaseException:
            self.conn.rollback()
            raise

    @classmethod
    def describe_table_schema(cls, connection: sqlite3.Connection, table_name: str) -> dict:
        """Describe an allowlisted application table without exposing SQL execution."""
        if table_name not in cls.SCHEMA_TABLE_ALLOWLIST:
            raise ValueError("Schema inspection is limited to approved agent tables.")
        table = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table_name,),
        ).fetchone()
        if table is None:
            raise ValueError(f"Approved table {table_name!r} does not exist.")
        columns = connection.execute(
            f'PRAGMA table_info("{table_name}")'
        ).fetchall()
        foreign_keys = connection.execute(
            f'PRAGMA foreign_key_list("{table_name}")'
        ).fetchall()
        indexes = connection.execute(
            f'PRAGMA index_list("{table_name}")'
        ).fetchall()
        index_details = []
        for index in indexes:
            index_name = str(index[1]).replace('"', '""')
            index_columns = connection.execute(
                f'PRAGMA index_info("{index_name}")'
            ).fetchall()
            index_details.append(
                {
                    "name": index[1],
                    "unique": bool(index[2]),
                    "origin": index[3],
                    "columns": [item[2] for item in index_columns],
                }
            )
        create_sql = table[0] or ""
        checks = re.findall(
            r"\bCHECK\s*\(([^()]*(?:\([^()]*\)[^()]*)*)\)",
            create_sql,
            flags=re.IGNORECASE,
        )
        return {
            "table": table_name,
            "columns": [
                {
                    "name": row[1],
                    "type": row[2] or "unspecified",
                    "not_null": bool(row[3]),
                    "default": row[4],
                    "primary_key_position": row[5],
                }
                for row in columns
            ],
            "foreign_keys": [
                {
                    "referenced_table": row[2],
                    "from_column": row[3],
                    "to_column": row[4],
                    "on_update": row[5],
                    "on_delete": row[6],
                }
                for row in foreign_keys
            ],
            "indexes": index_details,
            "declared_check_constraints": checks,
            "create_statement": create_sql,
            "length_limit_note": (
                "SQLite declared types do not imply a maximum string length. "
                "No explicit database maximum applies unless a CHECK constraint "
                "states one; application API limits are reported separately."
            ),
        }

    def get_table_schema(self, table_name: str) -> dict:
        """Describe an allowlisted application table using this connection."""
        with self._lock:
            return self.describe_table_schema(self.conn, table_name)

    def _configure_episode_fts(self) -> None:
        """Install a private FTS index when the SQLite build provides FTS5."""
        try:
            self.conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS agent_episode_fts USING "
                "fts5(session_id UNINDEXED, timestamp UNINDEXED, content, "
                "topic, description, task_summary)"
            )
            for trigger_sql in (
                """
                CREATE TRIGGER IF NOT EXISTS chat_history_episode_fts_insert
                AFTER INSERT ON chat_history
                WHEN NEW.role = 'summary'
                BEGIN
                    INSERT INTO agent_episode_fts
                    (rowid, session_id, timestamp, content, topic, description, task_summary)
                    VALUES
                    (NEW.id, NEW.session_id, NEW.timestamp, NEW.content,
                     NEW.episode_topic, NEW.episode_description, NEW.episode_task_summary);
                END
                """,
                """
                CREATE TRIGGER IF NOT EXISTS chat_history_episode_fts_delete
                AFTER DELETE ON chat_history
                WHEN OLD.role = 'summary'
                BEGIN
                    DELETE FROM agent_episode_fts WHERE rowid = OLD.id;
                END
                """,
                """
                CREATE TRIGGER IF NOT EXISTS chat_history_episode_fts_update
                AFTER UPDATE ON chat_history
                BEGIN
                    DELETE FROM agent_episode_fts
                    WHERE rowid = OLD.id AND OLD.role = 'summary';
                    INSERT INTO agent_episode_fts
                    (rowid, session_id, timestamp, content, topic, description, task_summary)
                    SELECT NEW.id, NEW.session_id, NEW.timestamp, NEW.content,
                           NEW.episode_topic, NEW.episode_description,
                           NEW.episode_task_summary
                    WHERE NEW.role = 'summary';
                END
                """,
            ):
                self.conn.execute(trigger_sql)
            self.conn.execute(
                "INSERT INTO agent_episode_fts "
                "(rowid, session_id, timestamp, content, topic, description, task_summary) "
                "SELECT id, session_id, timestamp, content, episode_topic, "
                "episode_description, episode_task_summary FROM chat_history "
                "WHERE role = 'summary' AND id NOT IN "
                "(SELECT rowid FROM agent_episode_fts)"
            )
            self._episode_fts_enabled = True
        except sqlite3.OperationalError as exc:
            self._episode_fts_enabled = False
            _LOGGER.warning(
                "Episode FTS5 unavailable; using bounded lexical retrieval: %s",
                str(exc)[:200],
            )

    def _create_table(self):
        with self._lock:
            with self.conn:
                version = self.conn.execute("PRAGMA user_version").fetchone()[0]
                if version > _SCHEMA_VERSION:
                    raise sqlite3.DatabaseError(
                        f"Database schema version {version} is newer than the "
                        f"supported version {_SCHEMA_VERSION}."
                    )
                self.conn.execute("""
                    CREATE TABLE IF NOT EXISTS chat_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT,
                        role TEXT,
                        content TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        is_skill_file BOOLEAN NOT NULL DEFAULT 0,
                        skill_file_path TEXT
                    )
                """)
                columns = {
                    row[1]
                    for row in self.conn.execute(
                        "PRAGMA table_info(chat_history)"
                    ).fetchall()
                }
                if "is_skill_file" not in columns:
                    self.conn.execute(
                        "ALTER TABLE chat_history ADD COLUMN "
                        "is_skill_file BOOLEAN NOT NULL DEFAULT 0"
                    )
                if "skill_file_path" not in columns:
                    self.conn.execute(
                        "ALTER TABLE chat_history ADD COLUMN skill_file_path TEXT"
                    )
                for column, column_type in (
                    ("episode_topic", "TEXT"),
                    ("episode_description", "TEXT"),
                    ("episode_task_summary", "TEXT"),
                    ("episode_plan_json", "TEXT"),
                    ("episode_outcome", "TEXT"),
                    ("episode_task_id", "TEXT"),
                    ("skill_task_id", "TEXT"),
                ):
                    if column not in columns:
                        self.conn.execute(
                            f"ALTER TABLE chat_history ADD COLUMN {column} "
                            f"{column_type}"
                        )
                self.conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_chat_history_session_id ON chat_history(session_id, id)"
                )
                self.conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_chat_history_episode_task "
                    "ON chat_history(episode_task_id)"
                )
                self.conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_chat_history_skill_task "
                    "ON chat_history(skill_task_id)"
                )
                self._configure_episode_fts()
                self.conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS agent_tasks (
                        task_id TEXT PRIMARY KEY,
                        originating_session_id TEXT NOT NULL,
                        goal TEXT NOT NULL,
                        task_type TEXT NOT NULL,
                        status TEXT NOT NULL,
                        current_revision INTEGER NOT NULL,
                        approved_revision INTEGER,
                        approved_digest TEXT,
                        approved_scope TEXT,
                        user_rating INTEGER,
                        resume_state_json TEXT,
                        validation_revision INTEGER,
                        validation_evidence TEXT,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        checkpoint_at TEXT
                    )
                    """
                )
                task_columns = {
                    row[1]
                    for row in self.conn.execute(
                        "PRAGMA table_info(agent_tasks)"
                    ).fetchall()
                }
                if "user_rating" not in task_columns:
                    self.conn.execute(
                        "ALTER TABLE agent_tasks ADD COLUMN user_rating INTEGER"
                    )
                if "resume_state_json" not in task_columns:
                    self.conn.execute(
                        "ALTER TABLE agent_tasks ADD COLUMN resume_state_json TEXT"
                    )
                if "validation_revision" not in task_columns:
                    self.conn.execute(
                        "ALTER TABLE agent_tasks ADD COLUMN validation_revision INTEGER"
                    )
                if "validation_evidence" not in task_columns:
                    self.conn.execute(
                        "ALTER TABLE agent_tasks ADD COLUMN validation_evidence TEXT"
                    )
                self.conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS agent_plan_revisions (
                        task_id TEXT NOT NULL REFERENCES agent_tasks(task_id)
                            ON DELETE CASCADE,
                        revision INTEGER NOT NULL,
                        digest TEXT NOT NULL,
                        plan_json TEXT NOT NULL,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (task_id, revision)
                    )
                    """
                )
                self.conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS agent_todo_steps (
                        task_id TEXT NOT NULL,
                        revision INTEGER NOT NULL,
                        step_id TEXT NOT NULL,
                        position INTEGER NOT NULL,
                        description TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'pending',
                        evidence TEXT NOT NULL DEFAULT '',
                        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (task_id, revision, step_id),
                        FOREIGN KEY (task_id, revision)
                            REFERENCES agent_plan_revisions(task_id, revision)
                            ON DELETE CASCADE
                    )
                    """
                )
                self.conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS agent_task_events (
                        event_id TEXT PRIMARY KEY,
                        task_id TEXT NOT NULL REFERENCES agent_tasks(task_id)
                            ON DELETE CASCADE,
                        revision INTEGER NOT NULL,
                        event_type TEXT NOT NULL,
                        details_json TEXT NOT NULL,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                self.conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS agent_task_leases (
                        task_id TEXT PRIMARY KEY REFERENCES agent_tasks(task_id)
                            ON DELETE CASCADE,
                        owner_id TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        expires_at REAL NOT NULL,
                        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                self.conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS agent_task_actions (
                        action_id TEXT PRIMARY KEY,
                        task_id TEXT NOT NULL REFERENCES agent_tasks(task_id)
                            ON DELETE CASCADE,
                        revision INTEGER NOT NULL,
                        tool_call_id TEXT NOT NULL,
                        tool_name TEXT NOT NULL,
                        arguments_digest TEXT NOT NULL,
                        status TEXT NOT NULL,
                        started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        outcome_digest TEXT,
                        UNIQUE(task_id, tool_call_id)
                    )
                    """
                )
                self.conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_agent_task_actions_status "
                    "ON agent_task_actions(task_id, status, started_at)"
                )
                self.conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS agent_learned_items (
                        item_id TEXT PRIMARY KEY,
                        statement TEXT NOT NULL,
                        scope TEXT NOT NULL,
                        provenance TEXT NOT NULL,
                        source_task_id TEXT,
                        status TEXT NOT NULL DEFAULT 'active',
                        expires_at TEXT,
                        review_at TEXT,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                learned_columns = {
                    row[1]
                    for row in self.conn.execute(
                        "PRAGMA table_info(agent_learned_items)"
                    ).fetchall()
                }
                for column in ("expires_at", "review_at"):
                    if column not in learned_columns:
                        self.conn.execute(
                            f"ALTER TABLE agent_learned_items ADD COLUMN {column} TEXT"
                        )
                self.conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_agent_learned_status "
                    "ON agent_learned_items(status, updated_at DESC)"
                )
                self.conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS agent_learning_settings (
                        setting_key TEXT PRIMARY KEY,
                        setting_value TEXT NOT NULL
                    )
                    """
                )
                self.conn.execute(
                    "INSERT OR IGNORE INTO agent_learning_settings "
                    "(setting_key, setting_value) VALUES ('enabled', '1')"
                )
                self.conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_agent_tasks_status_updated "
                    "ON agent_tasks(status, updated_at DESC)"
                )
                self.conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_agent_task_events_task "
                    "ON agent_task_events(task_id, created_at)"
                )
                self.conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")

    @staticmethod
    def _plan_digest(plan: dict) -> tuple[str, str]:
        plan_json = json.dumps(
            plan,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return plan_json, hashlib.sha256(plan_json.encode("utf-8")).hexdigest()

    @staticmethod
    def _sanitize_episode_text(value: str) -> str:
        value = re.sub(
            r"(?i)\b(api[_-]?key|access[_-]?token|password|secret)\b"
            r"(\s*[:=]\s*)[^\s,;]+",
            r"\1\2[redacted]",
            value,
        )
        value = re.sub(
            r"(?<![\w])/(?:home|Users|tmp|var|opt|mnt)/[^\s,;)\]]+",
            "[local path]",
            value,
        )
        return value.strip()

    @staticmethod
    def _validate_learned_statement(statement: str) -> str:
        statement = str(statement).strip()
        if not statement or len(statement) > MAX_LEARNED_ITEM_CHARS:
            raise ValueError(
                "Learned statement must contain 1 to "
                f"{MAX_LEARNED_ITEM_CHARS} characters."
            )
        sensitive_patterns = (
            r"(?i)\b(?:api[_-]?key|access[_-]?token|password|secret|"
            r"private[_-]?key)\b\s*[:=]\s*\S+",
            r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
            r"\bAKIA[0-9A-Z]{16}\b",
            r"\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,})\b",
            r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
            r"\b\d{3}-\d{2}-\d{4}\b",
            r"(?<!\w)(?:\+\d{1,3}[ -]?)?(?:\(\d{3}\)|\d{3})"
            r"[ .-]\d{3}[ .-]\d{4}(?!\w)",
            r"(?<![\w])/(?:home|Users)/[^/\s]+(?:/[^\s]*)?",
            r"(?i)\b[A-Z]:\\Users\\[^\\\s]+(?:\\[^\s]*)?",
        )
        if any(re.search(pattern, statement) for pattern in sensitive_patterns):
            raise ValueError(
                "Learned statements cannot contain credentials, contact details, "
                "government identifiers, or user-specific filesystem paths."
            )
        return statement

    @staticmethod
    def _decode_episode_plan(
        plan_json: Optional[str],
        episode_id: int,
    ) -> Optional[dict]:
        if not plan_json:
            return None
        try:
            plan = json.loads(plan_json)
        except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
            _LOGGER.warning("Skipping malformed structured episode plan id=%s", episode_id)
            return None
        if (
            not isinstance(plan, dict)
            or not isinstance(plan.get("steps"), list)
            or any(not isinstance(step, dict) for step in plan["steps"])
        ):
            _LOGGER.warning("Skipping invalid structured episode plan id=%s", episode_id)
            return None
        return plan

    @staticmethod
    def _decode_resume_capsule(
        capsule_json: Optional[str],
        task_id: str,
    ) -> Optional[dict]:
        if not capsule_json:
            return None
        try:
            capsule = json.loads(capsule_json)
        except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
            _LOGGER.warning("Skipping malformed task resume capsule id=%s", task_id)
            return None
        if (
            not isinstance(capsule, dict)
            or not isinstance(capsule.get("todo"), list)
            or not isinstance(capsule.get("unresolved_actions"), list)
        ):
            _LOGGER.warning("Skipping invalid task resume capsule id=%s", task_id)
            return None
        return capsule

    @staticmethod
    def _decode_task_plan(plan_json: str, task_id: str, revision: int) -> dict:
        try:
            plan = json.loads(plan_json)
        except (json.JSONDecodeError, TypeError, UnicodeDecodeError) as exc:
            _LOGGER.warning(
                "Malformed saved task plan id=%s revision=%s error=%s",
                task_id,
                revision,
                type(exc).__name__,
            )
            raise ValueError(
                f"Saved task plan {task_id} revision {revision} is malformed."
            ) from exc
        if (
            not isinstance(plan, dict)
            or not isinstance(plan.get("steps"), list)
            or any(not isinstance(step, dict) for step in plan["steps"])
        ):
            _LOGGER.warning(
                "Invalid saved task plan id=%s revision=%s",
                task_id,
                revision,
            )
            raise ValueError(
                f"Saved task plan {task_id} revision {revision} is invalid."
            )
        return plan

    def _record_task_event(
        self,
        task_id: str,
        revision: int,
        event_type: str,
        details: dict,
    ) -> None:
        details_json = json.dumps(details, ensure_ascii=False, sort_keys=True)
        event_id = hashlib.sha256(
            f"{task_id}:{revision}:{event_type}:{details_json}".encode("utf-8")
        ).hexdigest()
        self.conn.execute(
            "INSERT OR IGNORE INTO agent_task_events "
            "(event_id, task_id, revision, event_type, details_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                event_id,
                task_id,
                revision,
                event_type,
                details_json,
            ),
        )

    def _refresh_task_resume_capsule(self, task_id: str) -> None:
        task = self.conn.execute(
            "SELECT goal, current_revision, status, checkpoint_at "
            "FROM agent_tasks WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        if task is None:
            return
        revision = int(task[1])
        plan_row = self.conn.execute(
            "SELECT plan_json FROM agent_plan_revisions "
            "WHERE task_id = ? AND revision = ?",
            (task_id, revision),
        ).fetchone()
        step_rows = self.conn.execute(
            "SELECT step_id, status, evidence FROM agent_todo_steps "
            "WHERE task_id = ? AND revision = ? ORDER BY position",
            (task_id, revision),
        ).fetchall()
        action_rows = self.conn.execute(
            "SELECT action_id, tool_name, status FROM agent_task_actions "
            "WHERE task_id = ? AND status IN ('in_flight', 'outcome_unknown') "
            "ORDER BY started_at",
            (task_id,),
        ).fetchall()
        if plan_row is None:
            _LOGGER.warning(
                "Missing saved task plan id=%s revision=%s",
                task_id,
                revision,
            )
            raise ValueError(
                f"Saved task plan {task_id} revision {revision} is missing."
            )
        plan = self._decode_task_plan(plan_row[0], task_id, revision)
        unresolved = [
            {
                "action_id": row[0],
                "tool_name": row[1],
                "status": row[2],
            }
            for row in action_rows
        ]
        if unresolved:
            next_safe_action = "Reconcile unresolved tool actions before continuing."
        elif task[2] != "active":
            next_safe_action = "Review the saved task and obtain approval before execution."
        else:
            next_step = next(
                (
                    row for row in step_rows
                    if row[1] in {"in_progress", "pending", "blocked"}
                ),
                None,
            )
            next_safe_action = (
                f"Continue step {next_step[0]}."
                if next_step
                else "Review verification evidence and complete the task."
            )
        capsule = {
            "goal": task[0],
            "assumptions": plan.get("assumptions", []),
            "constraints": plan.get("constraints", []),
            "unresolved_questions": plan.get("research_questions", []),
            "revision": revision,
            "todo": [
                {
                    "step_id": row[0],
                    "status": row[1],
                    "evidence_ref": (
                        f"task:{task_id}:revision:{revision}:step:{row[0]}"
                        if row[2]
                        else None
                    ),
                }
                for row in step_rows
            ],
            "checkpoint_at": task[3],
            "unresolved_actions": unresolved,
            "next_safe_action": (
                "Task is completed; no execution action remains."
                if task[2] == "completed"
                else "Task was cancelled; review its history before starting new work."
                if task[2] in {"cancelled", "abandoned"}
                else next_safe_action
            ),
        }
        capsule_json = json.dumps(
            capsule,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if len(capsule_json) > MAX_RESUME_STATE_CHARS:
            raise ValueError(
                "Task resume capsule exceeds the configured "
                f"{MAX_RESUME_STATE_CHARS}-character limit. Reduce plan "
                "constraints or steps before saving progress."
            )
        self.conn.execute(
            "UPDATE agent_tasks SET resume_state_json = ? WHERE task_id = ?",
            (capsule_json, task_id),
        )

    @staticmethod
    def _normalize_text_items(value: object, label: str) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise TypeError(f"{label} must be a list of text items.")
        if len(value) > 10:
            raise ValueError(f"{label} must contain at most 10 text items.")
        normalized = []
        for index, item in enumerate(value, start=1):
            if not isinstance(item, str):
                raise TypeError(f"{label} item {index} must be text.")
            text = item.strip()
            if not text or len(text) > 500:
                raise ValueError(
                    f"{label} item {index} must contain 1 to 500 characters."
                )
            normalized.append(text)
        return normalized

    @staticmethod
    def _normalize_research_references(value: object) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise TypeError("Research references must be a list of text citations.")
        if len(value) > 10:
            raise ValueError("Research references must contain at most 10 citations.")
        references = []
        total_chars = 0
        for index, item in enumerate(value, start=1):
            if not isinstance(item, str):
                raise TypeError(f"Research reference {index} must be text.")
            reference = item.strip()
            if not reference or len(reference) > 1500:
                raise ValueError(
                    f"Research reference {index} must contain 1 to 1500 characters."
                )
            total_chars += len(reference)
            references.append(reference)
        if total_chars > 10000:
            raise ValueError(
                "Research citations exceed the 10000-character plan budget."
            )
        return references

    @classmethod
    def _normalize_plan_steps(cls, steps: list[dict]) -> list[dict]:
        if not steps or len(steps) > 20:
            raise ValueError("A plan must contain between 1 and 20 steps.")
        normalized = []
        for index, step in enumerate(steps, start=1):
            step_data = (
                step
                if isinstance(step, dict)
                else step.model_dump()
                if callable(getattr(step, "model_dump", None))
                else {}
            )
            description = str(step_data.get("description", "")).strip()
            if not description or len(description) > 1000:
                raise ValueError(
                    f"Plan step {index} must contain 1 to 1000 characters."
                )
            step_id = str(step_data.get("step_id") or f"step-{index}").strip()
            if len(step_id) > 64 or not step_id:
                raise ValueError(f"Plan step {index} has an invalid ID.")
            dependencies = step_data.get("dependencies", [])
            if not isinstance(dependencies, list) or len(dependencies) > 20:
                raise ValueError(
                    f"Plan step {index} dependencies must be a list of step IDs."
                )
            if any(not isinstance(item, str) for item in dependencies):
                raise ValueError(f"Plan step {index} dependency IDs must be text.")
            dependency_ids = [item.strip() for item in dependencies]
            if any(not item or len(item) > 64 for item in dependency_ids):
                raise ValueError(f"Plan step {index} has an invalid dependency ID.")
            if len(set(dependency_ids)) != len(dependency_ids):
                raise ValueError(f"Plan step {index} has duplicate dependencies.")
            normalized.append(
                {
                    "step_id": step_id,
                    "description": description,
                    "dependencies": dependency_ids,
                    "validation": cls._normalize_text_items(
                        step_data.get("validation", []),
                        f"Plan step {index} validation",
                    ),
                    "proof": cls._normalize_text_items(
                        step_data.get("proof", []),
                        f"Plan step {index} proof",
                    ),
                    "risks": cls._normalize_text_items(
                        step_data.get("risks", []),
                        f"Plan step {index} risks",
                    ),
                    "edge_cases": cls._normalize_text_items(
                        step_data.get("edge_cases", []),
                        f"Plan step {index} edge cases",
                    ),
                }
            )
        if len({step["step_id"] for step in normalized}) != len(normalized):
            raise ValueError("Plan step IDs must be unique.")
        step_ids = {step["step_id"] for step in normalized}
        dependencies = {
            step["step_id"]: step["dependencies"] for step in normalized
        }
        for step in normalized:
            if step["step_id"] in step["dependencies"]:
                raise ValueError(f"Plan step {step['step_id']} cannot depend on itself.")
            missing = set(step["dependencies"]) - step_ids
            if missing:
                raise ValueError(
                    f"Plan step {step['step_id']} references unknown dependencies: "
                    + ", ".join(sorted(missing))
                )

        visited: set[str] = set()
        visiting: set[str] = set()

        def visit(step_id: str) -> None:
            if step_id in visiting:
                raise ValueError("Plan step dependencies must not contain a cycle.")
            if step_id in visited:
                return
            visiting.add(step_id)
            for dependency in dependencies[step_id]:
                visit(dependency)
            visiting.remove(step_id)
            visited.add(step_id)

        for step_id in dependencies:
            visit(step_id)
        return normalized

    def create_task_plan(
        self,
        session_id: str,
        goal: str,
        task_type: str,
        steps: list[dict],
        *,
        assumptions: Optional[list[str]] = None,
        constraints: Optional[list[str]] = None,
        research_questions: Optional[list[str]] = None,
        research_references: Optional[list[str]] = None,
    ) -> dict:
        """Persist a new task and its first immutable plan revision."""
        goal = str(goal).strip()
        task_type = str(task_type).strip().lower()
        if not session_id or not goal or len(goal) > 2000:
            raise ValueError("Task goal must contain 1 to 2000 characters.")
        if task_type not in {"research", "coding", "other"}:
            raise ValueError("Task type must be research, coding, or other.")
        normalized_steps = self._normalize_plan_steps(steps)
        plan = {
            "goal": goal,
            "task_type": task_type,
            "steps": normalized_steps,
            "assumptions": self._normalize_text_items(assumptions, "Assumptions"),
            "constraints": self._normalize_text_items(constraints, "Constraints"),
            "research_questions": self._normalize_text_items(
                research_questions,
                "Research questions",
            ),
            "research_references": self._normalize_research_references(
                research_references
            ),
        }
        plan_json, digest = self._plan_digest(plan)
        task_id = uuid.uuid4().hex
        with self._lock:
            with self._write_transaction():
                self.conn.execute(
                    "INSERT INTO agent_tasks "
                    "(task_id, originating_session_id, goal, task_type, status, "
                    "current_revision) VALUES (?, ?, ?, ?, 'awaiting_approval', 1)",
                    (task_id, session_id, goal, task_type),
                )
                self.conn.execute(
                    "INSERT INTO agent_plan_revisions "
                    "(task_id, revision, digest, plan_json) VALUES (?, 1, ?, ?)",
                    (task_id, digest, plan_json),
                )
                self.conn.executemany(
                    "INSERT INTO agent_todo_steps "
                    "(task_id, revision, step_id, position, description) "
                    "VALUES (?, 1, ?, ?, ?)",
                    [
                        (task_id, step["step_id"], index, step["description"])
                        for index, step in enumerate(normalized_steps, start=1)
                    ],
                )
                self._record_task_event(
                    task_id,
                    1,
                    "created",
                    {"goal": goal, "task_type": task_type},
                )
                self._refresh_task_resume_capsule(task_id)
        return self.get_task(task_id)

    def revise_task_plan(
        self,
        task_id: str,
        goal: str,
        task_type: str,
        steps: list[dict],
        *,
        expected_revision: int,
        assumptions: Optional[list[str]] = None,
        constraints: Optional[list[str]] = None,
        research_questions: Optional[list[str]] = None,
        research_references: Optional[list[str]] = None,
    ) -> dict:
        """Append an immutable revision and invalidate any previous approval."""
        goal = str(goal).strip()
        task_type = str(task_type).strip().lower()
        if not goal or len(goal) > 2000:
            raise ValueError("Task goal must contain 1 to 2000 characters.")
        if task_type not in {"research", "coding", "other"}:
            raise ValueError("Task type must be research, coding, or other.")
        normalized_steps = self._normalize_plan_steps(steps)
        plan = {
            "goal": goal,
            "task_type": task_type,
            "steps": normalized_steps,
            "assumptions": self._normalize_text_items(assumptions, "Assumptions"),
            "constraints": self._normalize_text_items(constraints, "Constraints"),
            "research_questions": self._normalize_text_items(
                research_questions,
                "Research questions",
            ),
            "research_references": self._normalize_research_references(
                research_references
            ),
        }
        plan_json, digest = self._plan_digest(plan)
        with self._lock:
            with self._write_transaction():
                task = self.conn.execute(
                    "SELECT current_revision FROM agent_tasks WHERE task_id = ?",
                    (task_id,),
                ).fetchone()
                if task is None:
                    raise ValueError("Task not found.")
                if task[0] != expected_revision:
                    raise ValueError(
                        "Task plan changed since it was read; refresh before revising."
                    )
                revision = expected_revision + 1
                self.conn.execute(
                    "INSERT INTO agent_plan_revisions "
                    "(task_id, revision, digest, plan_json) VALUES (?, ?, ?, ?)",
                    (task_id, revision, digest, plan_json),
                )
                self.conn.executemany(
                    "INSERT INTO agent_todo_steps "
                    "(task_id, revision, step_id, position, description) "
                    "VALUES (?, ?, ?, ?, ?)",
                    [
                        (
                            task_id,
                            revision,
                            step["step_id"],
                            index,
                            step["description"],
                        )
                        for index, step in enumerate(normalized_steps, start=1)
                    ],
                )
                self.conn.execute(
                    "UPDATE agent_tasks SET goal = ?, task_type = ?, status = "
                    "'awaiting_approval', current_revision = ?, "
                    "approved_revision = NULL, approved_digest = NULL, "
                    "approved_scope = NULL, validation_revision = NULL, "
                    "validation_evidence = NULL, updated_at = CURRENT_TIMESTAMP, "
                    "checkpoint_at = CURRENT_TIMESTAMP WHERE task_id = ?",
                    (goal, task_type, revision, task_id),
                )
                self._record_task_event(
                    task_id,
                    revision,
                    "revised",
                    {"previous_revision": expected_revision},
                )
                self._refresh_task_resume_capsule(task_id)
        return self.get_task(task_id)

    def list_tasks(self, limit: int = 50) -> list[dict]:
        limit = max(1, min(int(limit), 100))
        with self._lock:
            rows = self.conn.execute(
                "SELECT task_id FROM agent_tasks ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        tasks = []
        for row in rows:
            try:
                task = self.get_task(row[0])
            except ValueError as exc:
                _LOGGER.warning(
                    "Skipping malformed task from listing id=%s error=%s",
                    row[0],
                    str(exc)[:160],
                )
                continue
            if task is not None:
                tasks.append(task)
        return tasks

    def count_task_plans(self) -> tuple[int, int]:
        """Return total task plans and plans with current approval."""
        with self._lock:
            row = self.conn.execute(
                "SELECT COUNT(*), "
                "COALESCE(SUM(CASE WHEN task.approved_revision = task.current_revision "
                "AND task.approved_digest = revision.digest THEN 1 ELSE 0 END), 0) "
                "FROM agent_tasks AS task "
                "LEFT JOIN agent_plan_revisions AS revision "
                "ON revision.task_id = task.task_id "
                "AND revision.revision = task.current_revision"
            ).fetchone()
        return int(row[0]), int(row[1])

    def mark_stale_tasks(self, stale_after_days: int) -> list[str]:
        """Expire old task approvals without deleting the task or its recovery state."""
        stale_after_days = max(0, int(stale_after_days))
        if not stale_after_days:
            return []
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=stale_after_days)
        ).strftime("%Y-%m-%d %H:%M:%S")
        now = time.time()
        with self._lock:
            with self._write_transaction():
                rows = self.conn.execute(
                    "SELECT task.task_id, task.current_revision, task.status "
                    "FROM agent_tasks AS task WHERE task.status IN "
                    "('draft', 'awaiting_approval', 'active', 'paused', 'interrupted') "
                    "AND COALESCE(task.checkpoint_at, task.updated_at) < ? "
                    "AND NOT EXISTS (SELECT 1 FROM agent_task_leases AS lease "
                    "WHERE lease.task_id = task.task_id AND lease.expires_at > ?)",
                    (cutoff, now),
                ).fetchall()
                for task_id, revision, previous_status in rows:
                    self.conn.execute(
                        "UPDATE agent_tasks SET status = 'stale', "
                        "approved_revision = NULL, approved_digest = NULL, "
                        "approved_scope = NULL, updated_at = CURRENT_TIMESTAMP, "
                        "checkpoint_at = CURRENT_TIMESTAMP WHERE task_id = ?",
                        (task_id,),
                    )
                    self.conn.execute(
                        "DELETE FROM agent_task_leases WHERE task_id = ?",
                        (task_id,),
                    )
                    self._record_task_event(
                        task_id,
                        revision,
                        "staled",
                        {"previous_status": previous_status},
                    )
                    self._refresh_task_resume_capsule(task_id)
        return [row[0] for row in rows]

    def get_task(self, task_id: str) -> Optional[dict]:
        with self._lock:
            task = self.conn.execute(
                "SELECT task_id, originating_session_id, goal, task_type, status, "
                "current_revision, approved_revision, approved_digest, "
                "approved_scope, created_at, updated_at, checkpoint_at, "
                "user_rating, resume_state_json, validation_revision, "
                "validation_evidence "
                "FROM agent_tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if task is None:
                return None
            revision = task[5]
            plan = self.conn.execute(
                "SELECT digest, plan_json, created_at FROM agent_plan_revisions "
                "WHERE task_id = ? AND revision = ?",
                (task_id, revision),
            ).fetchone()
            steps = self.conn.execute(
                "SELECT step_id, position, description, status, evidence, updated_at "
                "FROM agent_todo_steps WHERE task_id = ? AND revision = ? "
                "ORDER BY position",
                (task_id, revision),
            ).fetchall()
        if plan is None:
            _LOGGER.warning(
                "Missing saved task plan id=%s revision=%s",
                task_id,
                revision,
            )
            raise ValueError(
                f"Saved task plan {task_id} revision {revision} is missing."
            )
        plan_data = self._decode_task_plan(plan[1], task_id, revision)
        step_details = {
            item["step_id"]: item for item in plan_data.get("steps", [])
        }
        return {
            "task_id": task[0],
            "session_id": task[1],
            "goal": task[2],
            "task_type": task[3],
            "status": task[4],
            "revision": revision,
            "approved_revision": task[6],
            "approved_digest": task[7],
            "approved_scope": task[8],
            "created_at": task[9],
            "updated_at": task[10],
            "checkpoint_at": task[11],
            "user_rating": task[12],
            "resume_capsule": self._decode_resume_capsule(task[13], task_id),
            "validation_revision": task[14],
            "validation_evidence": task[15],
            "digest": plan[0],
            "plan": plan_data,
            "plan_created_at": plan[2],
            "steps": [
                {
                    "step_id": step[0],
                    "position": step[1],
                    "description": step[2],
                    "status": step[3],
                    "evidence": step[4],
                    "updated_at": step[5],
                    **{
                        key: value
                        for key, value in step_details.get(step[0], {}).items()
                        if key not in {"step_id", "description"}
                    },
                }
                for step in steps
            ],
        }

    def approve_task_plan(
        self,
        task_id: str,
        revision: int,
        digest: str,
        scope: str,
        *,
        owner_id: Optional[str] = None,
    ) -> dict:
        """Record explicit runtime approval for one immutable plan and scope."""
        if not scope or len(scope) > 1000:
            raise ValueError("Approval scope must contain 1 to 1000 characters.")
        with self._lock:
            with self._write_transaction():
                task = self.conn.execute(
                    "SELECT current_revision, status FROM agent_tasks "
                    "WHERE task_id = ?",
                    (task_id,),
                ).fetchone()
                plan = self.conn.execute(
                    "SELECT digest FROM agent_plan_revisions "
                    "WHERE task_id = ? AND revision = ?",
                    (task_id, revision),
                ).fetchone()
                if task is None or plan is None:
                    raise ValueError("Task or plan revision not found.")
                lease = self.conn.execute(
                    "SELECT owner_id FROM agent_task_leases WHERE task_id = ?",
                    (task_id,),
                ).fetchone()
                if lease and lease[0] != owner_id:
                    raise ValueError(
                        "Task approval blocked: another process owns the task lease."
                    )
                if task[0] != revision or plan[0] != digest:
                    raise ValueError("Approval does not match the current plan.")
                if task[1] not in {"awaiting_approval", "paused", "interrupted"}:
                    raise ValueError(
                        f"Task cannot be approved while status is '{task[1]}'."
                    )
                self.conn.execute(
                    "UPDATE agent_tasks SET status = 'active', "
                    "approved_revision = ?, approved_digest = ?, "
                    "approved_scope = ?, updated_at = CURRENT_TIMESTAMP, "
                    "checkpoint_at = CURRENT_TIMESTAMP WHERE task_id = ?",
                    (revision, digest, scope, task_id),
                )
                self._record_task_event(
                    task_id,
                    revision,
                    "approved",
                    {"scope": scope, "digest": digest},
                )
                self._refresh_task_resume_capsule(task_id)
        return self.get_task(task_id)

    def acquire_task_lease(
        self,
        task_id: str,
        owner_id: str,
        session_id: str,
        *,
        lease_seconds: int = 300,
        allow_takeover: bool = False,
    ) -> bool:
        """Acquire or renew exclusive task execution ownership."""
        if not owner_id or not session_id:
            raise ValueError("Task lease requires an owner and session ID.")
        if not 30 <= lease_seconds <= 3600:
            raise ValueError("Task lease duration must be between 30 and 3600 seconds.")
        now = time.time()
        expires_at = now + lease_seconds
        with self._lock:
            with self._write_transaction():
                task = self.conn.execute(
                    "SELECT current_revision, status FROM agent_tasks "
                    "WHERE task_id = ?",
                    (task_id,),
                ).fetchone()
                if task is None:
                    raise ValueError("Task not found.")
                lease = self.conn.execute(
                    "SELECT owner_id, expires_at FROM agent_task_leases "
                    "WHERE task_id = ?",
                    (task_id,),
                ).fetchone()
                if (
                    lease
                    and lease[0] != owner_id
                    and (lease[1] > now or not allow_takeover)
                ):
                    return False
                self.conn.execute(
                    "INSERT INTO agent_task_leases "
                    "(task_id, owner_id, session_id, expires_at, updated_at) "
                    "VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP) "
                    "ON CONFLICT(task_id) DO UPDATE SET "
                    "owner_id = excluded.owner_id, session_id = excluded.session_id, "
                    "expires_at = excluded.expires_at, updated_at = CURRENT_TIMESTAMP",
                    (task_id, owner_id, session_id, expires_at),
                )
                self._record_task_event(
                    task_id,
                    task[0],
                    "lease_acquired",
                    {"session_id": session_id, "expires_at": expires_at},
                )
        return True

    def release_task_lease(self, task_id: str, owner_id: str) -> bool:
        with self._lock:
            with self._write_transaction():
                cursor = self.conn.execute(
                    "DELETE FROM agent_task_leases WHERE task_id = ? AND owner_id = ?",
                    (task_id, owner_id),
                )
                if cursor.rowcount:
                    self._record_task_event(
                        task_id,
                        0,
                        "lease_released",
                        {"owner_id": owner_id},
                    )
                return cursor.rowcount == 1

    def get_task_lease(self, task_id: str) -> Optional[dict]:
        with self._lock:
            row = self.conn.execute(
                "SELECT owner_id, session_id, expires_at FROM agent_task_leases "
                "WHERE task_id = ?",
                (task_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "owner_id": row[0],
            "session_id": row[1],
            "expires_at": row[2],
            "expired": row[2] <= time.time(),
        }

    def begin_task_action(
        self,
        task_id: str,
        owner_id: str,
        tool_call_id: str,
        tool_name: str,
        arguments: dict,
        *,
        lease_seconds: int = 180,
    ) -> str:
        """Durably record action intent before execution; never persist arguments."""
        if not tool_call_id or not tool_name:
            raise ValueError("Task action requires a tool call ID and name.")
        if not 30 <= lease_seconds <= 3600:
            raise ValueError("Task lease duration must be between 30 and 3600 seconds.")
        args_json = json.dumps(
            arguments,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        args_digest = hashlib.sha256(args_json.encode("utf-8")).hexdigest()
        action_id = uuid.uuid4().hex
        with self._lock:
            with self._write_transaction():
                task = self.conn.execute(
                    "SELECT current_revision, approved_revision, approved_digest, status "
                    "FROM agent_tasks WHERE task_id = ?",
                    (task_id,),
                ).fetchone()
                lease = self.conn.execute(
                    "SELECT owner_id, expires_at FROM agent_task_leases "
                    "WHERE task_id = ?",
                    (task_id,),
                ).fetchone()
                if not (
                    task
                    and task[3] == "active"
                    and task[0] == task[1]
                    and lease
                    and lease[0] == owner_id
                    and lease[1] > time.time()
                ):
                    raise ValueError(
                        "Action blocked: this process does not hold the active "
                        "approved task lease."
                    )
                unresolved = self.conn.execute(
                    "SELECT tool_name, status FROM agent_task_actions "
                    "WHERE task_id = ? AND status IN "
                    "('in_flight', 'outcome_unknown') LIMIT 1",
                    (task_id,),
                ).fetchone()
                if unresolved:
                    raise ValueError(
                        "Action blocked: unresolved action "
                        f"'{unresolved[0]}' has status '{unresolved[1]}'. "
                        "A user must reconcile it before another task action."
                    )
                plan = self.conn.execute(
                    "SELECT digest FROM agent_plan_revisions "
                    "WHERE task_id = ? AND revision = ?",
                    (task_id, task[0]),
                ).fetchone()
                if not plan or plan[0] != task[2]:
                    raise ValueError("Action blocked: active plan approval is stale.")
                existing = self.conn.execute(
                    "SELECT action_id, status FROM agent_task_actions "
                    "WHERE task_id = ? AND tool_call_id = ?",
                    (task_id, tool_call_id),
                ).fetchone()
                if existing:
                    raise ValueError(
                        f"Tool call {tool_call_id} already has durable action state "
                        f"'{existing[1]}'; reconcile before retrying."
                    )
                self.conn.execute(
                    "INSERT INTO agent_task_actions "
                    "(action_id, task_id, revision, tool_call_id, tool_name, "
                    "arguments_digest, status) "
                    "VALUES (?, ?, ?, ?, ?, ?, 'in_flight')",
                    (
                        action_id,
                        task_id,
                        task[0],
                        tool_call_id,
                        tool_name,
                        args_digest,
                    ),
                )
                self.conn.execute(
                    "UPDATE agent_task_leases SET expires_at = ?, "
                    "updated_at = CURRENT_TIMESTAMP WHERE task_id = ? AND owner_id = ?",
                    (time.time() + lease_seconds, task_id, owner_id),
                )
                self._record_task_event(
                    task_id,
                    task[0],
                    "action_intent",
                    {
                        "action_id": action_id,
                        "tool_name": tool_name,
                        "arguments_digest": args_digest,
                        "status": "in_flight",
                    },
                )
                self._refresh_task_resume_capsule(task_id)
        return action_id

    def finish_task_action(
        self,
        action_id: str,
        status: str,
        outcome_digest: Optional[str] = None,
    ) -> dict:
        if status not in {"outcome_observed", "outcome_unknown"}:
            raise ValueError("Task action outcome status is invalid.")
        with self._lock:
            with self._write_transaction():
                action = self.conn.execute(
                    "SELECT task_id, revision, status FROM agent_task_actions "
                    "WHERE action_id = ?",
                    (action_id,),
                ).fetchone()
                if action is None:
                    raise ValueError("Task action record not found.")
                if action[2] != "in_flight":
                    raise ValueError("Task action is no longer in flight.")
                self.conn.execute(
                    "UPDATE agent_task_actions SET status = ?, outcome_digest = ?, "
                    "updated_at = CURRENT_TIMESTAMP WHERE action_id = ?",
                    (status, outcome_digest, action_id),
                )
                self._record_task_event(
                    action[0],
                    action[1],
                    "action_outcome",
                    {"action_id": action_id, "status": status},
                )
                self._refresh_task_resume_capsule(action[0])
        return {
            "action_id": action_id,
            "task_id": action[0],
            "revision": action[1],
            "status": status,
        }

    def list_task_actions(
        self,
        task_id: str,
        *,
        unresolved_only: bool = False,
    ) -> list[dict]:
        query = (
            "SELECT action_id, revision, tool_call_id, tool_name, "
            "arguments_digest, status, started_at, updated_at "
            "FROM agent_task_actions WHERE task_id = ?"
        )
        if unresolved_only:
            query += " AND status IN ('in_flight', 'outcome_unknown')"
        query += " ORDER BY started_at"
        with self._lock:
            rows = self.conn.execute(query, (task_id,)).fetchall()
        return [
            {
                "action_id": row[0],
                "revision": row[1],
                "tool_call_id": row[2],
                "tool_name": row[3],
                "arguments_digest": row[4],
                "status": row[5],
                "started_at": row[6],
                "updated_at": row[7],
            }
            for row in rows
        ]

    def resolve_task_action(
        self,
        action_id: str,
        resolution: str,
        *,
        owner_id: Optional[str] = None,
    ) -> dict:
        """Record a user's explicit reconciliation of an uncertain action."""
        if resolution not in {"verified", "not_executed"}:
            raise ValueError("Resolution must be 'verified' or 'not_executed'.")
        with self._lock:
            with self._write_transaction():
                action = self.conn.execute(
                    "SELECT task_id, revision, status FROM agent_task_actions "
                    "WHERE action_id = ?",
                    (action_id,),
                ).fetchone()
                if action is None:
                    raise ValueError("Task action record not found.")
                if action[2] not in {"in_flight", "outcome_unknown"}:
                    raise ValueError("Only unresolved task actions can be reconciled.")
                lease = self.conn.execute(
                    "SELECT owner_id, expires_at FROM agent_task_leases "
                    "WHERE task_id = ?",
                    (action[0],),
                ).fetchone()
                if (
                    lease
                    and lease[1] > time.time()
                    and lease[0] != owner_id
                ):
                    raise ValueError(
                        "Action reconciliation blocked: another process still "
                        "holds the active task lease."
                    )
                self.conn.execute(
                    "UPDATE agent_task_actions SET status = ?, "
                    "updated_at = CURRENT_TIMESTAMP WHERE action_id = ?",
                    (resolution, action_id),
                )
                self._record_task_event(
                    action[0],
                    action[1],
                    "action_reconciled",
                    {"action_id": action_id, "resolution": resolution},
                )
                self._refresh_task_resume_capsule(action[0])
        return {
            "action_id": action_id,
            "task_id": action[0],
            "revision": action[1],
            "status": resolution,
        }

    def reconcile_task_actions_after_restart(self) -> int:
        """Mark orphaned in-flight actions unknown; never guess they completed."""
        with self._lock:
            with self._write_transaction():
                now = time.time()
                rows = self.conn.execute(
                    "SELECT action.action_id, action.task_id, action.revision "
                    "FROM agent_task_actions AS action "
                    "LEFT JOIN agent_task_leases AS lease "
                    "ON lease.task_id = action.task_id "
                    "WHERE action.status = 'in_flight' "
                    "AND (lease.task_id IS NULL OR lease.expires_at <= ?)",
                    (now,),
                ).fetchall()
                for action_id, task_id, revision in rows:
                    self.conn.execute(
                        "UPDATE agent_task_actions SET status = 'outcome_unknown', "
                        "updated_at = CURRENT_TIMESTAMP WHERE action_id = ?",
                        (action_id,),
                    )
                    self._record_task_event(
                        task_id,
                        revision,
                        "action_reconciliation_required",
                        {"action_id": action_id, "status": "outcome_unknown"},
                    )
                    self._refresh_task_resume_capsule(task_id)
                self.conn.execute(
                    "DELETE FROM agent_task_leases WHERE expires_at <= ?",
                    (now,),
                )
        return len(rows)

    def update_task_step(
        self,
        task_id: str,
        revision: int,
        step_id: str,
        status: str,
        evidence: str = "",
        owner_id: Optional[str] = None,
    ) -> dict:
        """Persist a model-reported todo update for the approved active plan."""
        allowed_statuses = {
            "pending",
            "in_progress",
            "blocked",
            "reported_done",
        }
        evidence = str(evidence).strip()
        if len(evidence) > 2000:
            raise ValueError(
                "Progress evidence exceeds the 2000-character application limit; "
                "prepare a concise, meaning-preserving summary."
            )
        if status not in allowed_statuses:
            raise ValueError(
                "Step status must be pending, in_progress, blocked, "
                "reported_done, or skipped."
            )
        if status in {"reported_done", "blocked"} and not evidence:
            raise ValueError("A reported completion or blocked step needs evidence.")
        with self._lock:
            with self._write_transaction():
                task = self.conn.execute(
                    "SELECT current_revision, approved_revision, approved_digest, "
                    "status FROM agent_tasks WHERE task_id = ?",
                    (task_id,),
                ).fetchone()
                plan = self.conn.execute(
                    "SELECT digest, plan_json FROM agent_plan_revisions "
                    "WHERE task_id = ? AND revision = ?",
                    (task_id, revision),
                ).fetchone()
                if (
                    task is None
                    or plan is None
                    or task[0] != revision
                    or task[1] != revision
                    or task[2] != plan[0]
                    or task[3] != "active"
                ):
                    raise ValueError("No active approval for this plan revision.")
                if owner_id is not None:
                    lease = self.conn.execute(
                        "SELECT owner_id, expires_at FROM agent_task_leases "
                        "WHERE task_id = ?",
                        (task_id,),
                    ).fetchone()
                    if (
                        not lease
                        or lease[0] != owner_id
                        or lease[1] <= time.time()
                    ):
                        raise ValueError(
                            "Progress update blocked: this process does not hold "
                            "the active task lease."
                        )
                step = self.conn.execute(
                    "SELECT position FROM agent_todo_steps WHERE task_id = ? "
                    "AND revision = ? AND step_id = ?",
                    (task_id, revision, step_id),
                ).fetchone()
                if step is None:
                    raise ValueError("Todo step not found in the active plan.")
                if status in {"in_progress", "reported_done"}:
                    plan_steps = json.loads(plan[1]).get("steps", [])
                    dependencies = next(
                        (
                            item.get("dependencies", [])
                            for item in plan_steps
                            if item.get("step_id") == step_id
                        ),
                        [],
                    )
                    dependency_states = {
                        row[0]: row[1]
                        for row in self.conn.execute(
                            "SELECT step_id, status FROM agent_todo_steps "
                            "WHERE task_id = ? AND revision = ?",
                            (task_id, revision),
                        ).fetchall()
                    }
                    unmet_dependencies = [
                        dependency
                        for dependency in dependencies
                        if dependency_states.get(dependency)
                        not in {"reported_done", "verified", "skipped"}
                    ]
                    if unmet_dependencies:
                        names = ", ".join(unmet_dependencies)
                        raise ValueError(
                            f"Step dependencies must be reported complete first: {names}"
                        )
                cursor = self.conn.execute(
                    "UPDATE agent_todo_steps SET status = ?, evidence = ?, "
                    "updated_at = CURRENT_TIMESTAMP WHERE task_id = ? "
                    "AND revision = ? AND step_id = ? AND status != 'verified'",
                    (status, evidence, task_id, revision, step_id),
                )
                if cursor.rowcount != 1:
                    raise ValueError("Todo step not found in the active plan.")
                self.conn.execute(
                    "UPDATE agent_tasks SET updated_at = CURRENT_TIMESTAMP, "
                    "checkpoint_at = CURRENT_TIMESTAMP WHERE task_id = ?",
                    (task_id,),
                )
                self._record_task_event(
                    task_id,
                    revision,
                    "step_updated",
                    {"step_id": step_id, "status": status, "evidence": evidence},
                )
                self._refresh_task_resume_capsule(task_id)
        return self.get_task(task_id)

    def verify_task_step(
        self,
        task_id: str,
        revision: int,
        step_id: str,
        evidence: str,
        *,
        owner_id: Optional[str] = None,
    ) -> dict:
        """Mark a model-reported step verified through the user/runtime flow."""
        evidence = str(evidence).strip()
        if not evidence:
            raise ValueError("Verification requires concrete evidence.")
        if len(evidence) > 2000:
            raise ValueError(
                "Verification evidence exceeds the 2000-character application "
                "limit; prepare a concise, meaning-preserving summary."
            )
        with self._lock:
            with self._write_transaction():
                task = self.conn.execute(
                    "SELECT current_revision, status FROM agent_tasks "
                    "WHERE task_id = ?",
                    (task_id,),
                ).fetchone()
                if task is None or task[0] != revision or task[1] != "active":
                    raise ValueError("Only steps in the active revision can be verified.")
                lease = self.conn.execute(
                    "SELECT owner_id FROM agent_task_leases WHERE task_id = ?",
                    (task_id,),
                ).fetchone()
                if lease and lease[0] != owner_id:
                    raise ValueError(
                        "Step verification blocked: another process owns the task lease."
                    )
                cursor = self.conn.execute(
                    "UPDATE agent_todo_steps SET status = 'verified', evidence = ?, "
                    "updated_at = CURRENT_TIMESTAMP WHERE task_id = ? "
                    "AND revision = ? AND step_id = ? "
                    "AND status = 'reported_done'",
                    (evidence, task_id, revision, step_id),
                )
                if cursor.rowcount != 1:
                    raise ValueError("Step is not reported complete or does not exist.")
                self.conn.execute(
                    "UPDATE agent_tasks SET updated_at = CURRENT_TIMESTAMP, "
                    "checkpoint_at = CURRENT_TIMESTAMP WHERE task_id = ?",
                    (task_id,),
                )
                self._record_task_event(
                    task_id,
                    revision,
                    "step_verified",
                    {"step_id": step_id, "evidence": evidence},
                )
                self._refresh_task_resume_capsule(task_id)
        return self.get_task(task_id)

    def record_task_validation(
        self,
        task_id: str,
        revision: int,
        evidence: str,
        *,
        owner_id: Optional[str] = None,
    ) -> dict:
        """Persist verified runtime evidence for the current task revision."""
        evidence = str(evidence).strip()
        if not evidence:
            raise ValueError("Task validation requires concrete evidence.")
        if len(evidence) > 2000:
            raise ValueError(
                "Task validation evidence exceeds the 2000-character application "
                "limit; prepare a concise, meaning-preserving summary."
            )
        with self._lock:
            with self._write_transaction():
                task = self.conn.execute(
                    "SELECT current_revision, status FROM agent_tasks "
                    "WHERE task_id = ?",
                    (task_id,),
                ).fetchone()
                if task is None or task[0] != revision or task[1] != "active":
                    raise ValueError(
                        "Validation evidence must match the active task revision."
                    )
                lease = self.conn.execute(
                    "SELECT owner_id FROM agent_task_leases WHERE task_id = ?",
                    (task_id,),
                ).fetchone()
                if lease and lease[0] != owner_id:
                    raise ValueError(
                        "Task validation blocked: another process owns the task lease."
                    )
                self.conn.execute(
                    "UPDATE agent_tasks SET validation_revision = ?, "
                    "validation_evidence = ?, updated_at = CURRENT_TIMESTAMP, "
                    "checkpoint_at = CURRENT_TIMESTAMP WHERE task_id = ?",
                    (revision, evidence, task_id),
                )
                self._record_task_event(
                    task_id,
                    revision,
                    "task_validated",
                    {"evidence": evidence},
                )
                self._refresh_task_resume_capsule(task_id)
        return self.get_task(task_id)

    def set_task_status(
        self,
        task_id: str,
        status: str,
        *,
        owner_id: Optional[str] = None,
    ) -> dict:
        """Apply user/runtime task lifecycle transitions; never model approval."""
        transitions = {
            "paused": {"active", "awaiting_approval"},
            "interrupted": {"active"},
            "awaiting_approval": {"active", "paused", "interrupted", "stale"},
            "stale": {"draft", "awaiting_approval", "active", "paused", "interrupted"},
            "cancelled": {
                "draft", "awaiting_approval", "active", "paused",
                "interrupted", "stale",
            },
            "abandoned": {
                "draft", "awaiting_approval", "active", "paused",
                "interrupted", "cancelled", "stale",
            },
        }
        if status not in transitions:
            raise ValueError("Unsupported task lifecycle transition.")
        with self._lock:
            with self._write_transaction():
                task = self.conn.execute(
                    "SELECT current_revision, status FROM agent_tasks "
                    "WHERE task_id = ?",
                    (task_id,),
                ).fetchone()
                if task is None:
                    raise ValueError("Task not found.")
                if task[1] not in transitions[status]:
                    raise ValueError(
                        f"Cannot change task from '{task[1]}' to '{status}'."
                    )
                lease = self.conn.execute(
                    "SELECT owner_id, expires_at FROM agent_task_leases "
                    "WHERE task_id = ?",
                    (task_id,),
                ).fetchone()
                if (
                    lease
                    and lease[1] > time.time()
                    and lease[0] != owner_id
                ):
                    raise ValueError(
                        "Task transition blocked: another process still holds "
                        "the active task lease."
                    )
                if status in {
                    "paused",
                    "interrupted",
                    "awaiting_approval",
                    "cancelled",
                    "abandoned",
                    "stale",
                }:
                    self.conn.execute(
                        "UPDATE agent_tasks SET status = ?, approved_revision = NULL, "
                        "approved_digest = NULL, approved_scope = NULL, "
                        "updated_at = CURRENT_TIMESTAMP, "
                        "checkpoint_at = CURRENT_TIMESTAMP WHERE task_id = ?",
                        (status, task_id),
                    )
                    self.conn.execute(
                        "DELETE FROM agent_task_leases WHERE task_id = ?",
                        (task_id,),
                    )
                self._record_task_event(
                    task_id, task[0], status, {"previous_status": task[1]}
                )
                self._refresh_task_resume_capsule(task_id)
        return self.get_task(task_id)

    def complete_task(
        self,
        task_id: str,
        *,
        owner_id: Optional[str] = None,
    ) -> dict:
        """Complete only when every current-revision step is explicitly verified."""
        with self._lock:
            with self._write_transaction():
                task = self.conn.execute(
                    "SELECT current_revision, status FROM agent_tasks "
                    "WHERE task_id = ?",
                    (task_id,),
                ).fetchone()
                if task is None:
                    raise ValueError("Task not found.")
                if task[1] != "active":
                    raise ValueError("Only an active task can be completed.")
                unresolved_actions = self.conn.execute(
                    "SELECT COUNT(*) FROM agent_task_actions "
                    "WHERE task_id = ? AND status IN "
                    "('in_flight', 'outcome_unknown')",
                    (task_id,),
                ).fetchone()[0]
                if unresolved_actions:
                    raise ValueError(
                        "Task cannot be completed while tool actions have "
                        "unresolved outcomes."
                    )
                lease = self.conn.execute(
                    "SELECT owner_id FROM agent_task_leases WHERE task_id = ?",
                    (task_id,),
                ).fetchone()
                if lease and lease[0] != owner_id:
                    raise ValueError(
                        "Task completion blocked: another process owns the task lease."
                    )
                remaining = self.conn.execute(
                    "SELECT COUNT(*) FROM agent_todo_steps WHERE task_id = ? "
                    "AND revision = ? AND status NOT IN ('verified', 'skipped')",
                    (task_id, task[0]),
                ).fetchone()[0]
                if remaining:
                    raise ValueError(
                        "Every current plan step must be verified or explicitly skipped."
                    )
                details = self.conn.execute(
                    "SELECT task.originating_session_id, task.goal, task.task_type, "
                    "revision.plan_json FROM agent_tasks AS task "
                    "JOIN agent_plan_revisions AS revision "
                    "ON revision.task_id = task.task_id "
                    "AND revision.revision = task.current_revision "
                    "WHERE task.task_id = ?",
                    (task_id,),
                ).fetchone()
                verified_steps = self.conn.execute(
                    "SELECT step_id, description, status, evidence "
                    "FROM agent_todo_steps WHERE task_id = ? AND revision = ? "
                    "ORDER BY position",
                    (task_id, task[0]),
                ).fetchall()
                plan = json.loads(details[3])
                plan_steps = {item["step_id"]: item for item in plan.get("steps", [])}
                episode_topic = self._sanitize_episode_text(
                    details[1].splitlines()[0]
                )
                episode_description = (
                    f"Completed {details[2]} task: "
                    f"{sum(row[2] == 'verified' for row in verified_steps)} verified, "
                    f"{sum(row[2] == 'skipped' for row in verified_steps)} skipped."
                )
                episode_task_summary = self._sanitize_episode_text(details[1])
                episode_plan = {
                    "assumptions": [
                        self._sanitize_episode_text(value)
                        for value in plan.get("assumptions", [])
                    ],
                    "constraints": [
                        self._sanitize_episode_text(value)
                        for value in plan.get("constraints", [])
                    ],
                    "research_questions": [
                        self._sanitize_episode_text(value)
                        for value in plan.get("research_questions", [])
                    ],
                    "steps": [
                        {
                            "step_id": row[0],
                            "description": self._sanitize_episode_text(row[1]),
                            "status": row[2],
                            "dependencies": plan_steps.get(row[0], {}).get(
                                "dependencies", []
                            ),
                            "validation": [
                                self._sanitize_episode_text(value)
                                for value in plan_steps.get(row[0], {}).get(
                                    "validation", []
                                )
                            ],
                            "proof": [
                                self._sanitize_episode_text(value)
                                for value in plan_steps.get(row[0], {}).get("proof", [])
                            ],
                            "risks": [
                                self._sanitize_episode_text(value)
                                for value in plan_steps.get(row[0], {}).get("risks", [])
                            ],
                            "edge_cases": [
                                self._sanitize_episode_text(value)
                                for value in plan_steps.get(row[0], {}).get(
                                    "edge_cases", []
                                )
                            ],
                        }
                        for row in verified_steps
                    ],
                }
                episode_outcome = (
                    "Completed after all steps were verified or explicitly skipped."
                )
                episode_content = (
                    f"Topic: {episode_topic}\n"
                    f"Description: {episode_description}\n"
                    f"{episode_description}\n{episode_outcome}"
                )
                if len(episode_content) > MAX_SUMMARY_CHARS:
                    raise ValueError(
                        "Structured episode summary exceeds the configured "
                        f"{MAX_SUMMARY_CHARS}-character budget."
                    )
                self.conn.execute(
                    "UPDATE agent_tasks SET status = 'completed', "
                    "updated_at = CURRENT_TIMESTAMP, "
                    "checkpoint_at = CURRENT_TIMESTAMP WHERE task_id = ?",
                    (task_id,),
                )
                self.conn.execute(
                    "DELETE FROM agent_task_leases WHERE task_id = ?",
                    (task_id,),
                )
                episode_cursor = self.conn.execute(
                    "INSERT INTO chat_history "
                    "(session_id, role, content, episode_topic, "
                    "episode_description, episode_task_summary, "
                    "episode_plan_json, episode_outcome, episode_task_id) "
                    "VALUES (?, 'summary', ?, ?, ?, ?, ?, ?, ?)",
                    (
                        details[0],
                        episode_content,
                        episode_topic,
                        episode_description,
                        episode_task_summary,
                        json.dumps(episode_plan, ensure_ascii=False),
                        episode_outcome,
                        task_id,
                    ),
                )
                self._record_task_event(
                    task_id, task[0], "completed", {"verified_steps_only": True}
                )
                self._refresh_task_resume_capsule(task_id)
        completed = self.get_task(task_id)
        completed["episode_id"] = episode_cursor.lastrowid
        return completed

    def rate_completed_task(self, task_id: str, rating: int) -> dict:
        """Persist subjective user feedback for a successfully completed task."""
        if not isinstance(rating, int) or isinstance(rating, bool) or not 0 <= rating <= 5:
            raise ValueError("Task completeness rating must be an integer from 0 to 5.")
        with self._lock:
            with self._write_transaction():
                task = self.conn.execute(
                    "SELECT current_revision, status FROM agent_tasks "
                    "WHERE task_id = ?",
                    (task_id,),
                ).fetchone()
                if task is None or task[1] != "completed":
                    raise ValueError("Only a completed task can receive a rating.")
                self.conn.execute(
                    "UPDATE agent_tasks SET user_rating = ?, "
                    "updated_at = CURRENT_TIMESTAMP WHERE task_id = ?",
                    (rating, task_id),
                )
                self._record_task_event(
                    task_id,
                    task[0],
                    "user_rated",
                    {"rating": rating, "rating_is_subjective": True},
                )
        return self.get_task(task_id)

    def learning_is_enabled(self) -> bool:
        with self._lock:
            row = self.conn.execute(
                "SELECT setting_value FROM agent_learning_settings "
                "WHERE setting_key = 'enabled'"
            ).fetchone()
        return bool(row and row[0] == "1")

    def set_learning_enabled(self, enabled: bool) -> None:
        with self._lock:
            with self._write_transaction():
                self.conn.execute(
                    "INSERT INTO agent_learning_settings "
                    "(setting_key, setting_value) VALUES ('enabled', ?) "
                    "ON CONFLICT(setting_key) DO UPDATE SET "
                    "setting_value = excluded.setting_value",
                    ("1" if enabled else "0",),
                )

    def add_learned_item(
        self,
        statement: str,
        scope: str,
        *,
        provenance: str = "user-confirmed",
        source_task_id: Optional[str] = None,
    ) -> dict:
        statement = self._validate_learned_statement(statement)
        scope = str(scope).strip().lower()
        provenance = str(provenance).strip()
        if scope not in {"global", "coding", "research", "other"}:
            raise ValueError("Learning scope must be global, coding, research, or other.")
        if not provenance or len(provenance) > 100:
            raise ValueError("Learning provenance must contain 1 to 100 characters.")
        item_id = uuid.uuid4().hex
        with self._lock:
            with self._write_transaction():
                enabled = self.conn.execute(
                    "SELECT setting_value FROM agent_learning_settings "
                    "WHERE setting_key = 'enabled'"
                ).fetchone()
                if enabled is None or enabled[0] != "1":
                    raise ValueError("Adaptive learning is disabled.")
                if source_task_id:
                    source = self.conn.execute(
                        "SELECT status FROM agent_tasks WHERE task_id = ?",
                        (source_task_id,),
                    ).fetchone()
                    if source is None or source[0] != "completed":
                        raise ValueError(
                            "Learned task procedures must reference a completed task."
                        )
                self.conn.execute(
                    "INSERT INTO agent_learned_items "
                    "(item_id, statement, scope, provenance, source_task_id, "
                    "expires_at, review_at) "
                    "VALUES (?, ?, ?, ?, ?, "
                    "datetime('now', ?), datetime('now', ?))",
                    (
                        item_id,
                        statement,
                        scope,
                        provenance,
                        source_task_id,
                        f"+{LEARNED_ITEM_EXPIRY_DAYS} days",
                        f"+{LEARNED_ITEM_REVIEW_DAYS} days",
                    ),
                )
        return self.get_learned_item(item_id)

    def list_learned_items(self, *, include_disabled: bool = True) -> list[dict]:
        query = "SELECT item_id FROM agent_learned_items"
        if not include_disabled:
            query += " WHERE status = 'active'"
        query += " ORDER BY updated_at DESC LIMIT 200"
        with self._lock:
            rows = self.conn.execute(query).fetchall()
        return [
            item for row in rows
            if (item := self.get_learned_item(row[0])) is not None
        ]

    def get_learned_item(self, item_id: str) -> Optional[dict]:
        with self._lock:
            row = self.conn.execute(
                "SELECT item_id, statement, scope, provenance, source_task_id, "
                "status, created_at, updated_at, expires_at, review_at "
                "FROM agent_learned_items "
                "WHERE item_id = ?",
                (item_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "item_id": row[0],
            "statement": row[1],
            "scope": row[2],
            "provenance": row[3],
            "source_task_id": row[4],
            "status": row[5],
            "created_at": row[6],
            "updated_at": row[7],
            "expires_at": row[8],
            "review_at": row[9],
            "expired": bool(row[8] and row[8] <= datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )),
            "review_due": bool(row[9] and row[9] <= datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )),
        }

    def update_learned_item(
        self,
        item_id: str,
        statement: str,
        scope: str,
    ) -> dict:
        statement = self._validate_learned_statement(statement)
        scope = str(scope).strip().lower()
        if scope not in {"global", "coding", "research", "other"}:
            raise ValueError("Learning scope must be global, coding, research, or other.")
        with self._lock:
            with self._write_transaction():
                cursor = self.conn.execute(
                    "UPDATE agent_learned_items SET statement = ?, scope = ?, "
                    "provenance = 'user-corrected', status = 'active', "
                    "expires_at = datetime('now', ?), "
                    "review_at = datetime('now', ?), "
                    "updated_at = CURRENT_TIMESTAMP WHERE item_id = ?",
                    (
                        statement,
                        scope,
                        f"+{LEARNED_ITEM_EXPIRY_DAYS} days",
                        f"+{LEARNED_ITEM_REVIEW_DAYS} days",
                        item_id,
                    ),
                )
                if cursor.rowcount != 1:
                    raise ValueError("Learned item was not found.")
        return self.get_learned_item(item_id)

    def review_learned_item(self, item_id: str) -> dict:
        """Renew an item only after the user explicitly confirms it remains useful."""
        with self._lock:
            with self._write_transaction():
                cursor = self.conn.execute(
                    "UPDATE agent_learned_items SET status = 'active', "
                    "expires_at = datetime('now', ?), "
                    "review_at = datetime('now', ?), "
                    "updated_at = CURRENT_TIMESTAMP WHERE item_id = ?",
                    (
                        f"+{LEARNED_ITEM_EXPIRY_DAYS} days",
                        f"+{LEARNED_ITEM_REVIEW_DAYS} days",
                        item_id,
                    ),
                )
                if cursor.rowcount != 1:
                    raise ValueError("Learned item was not found.")
        return self.get_learned_item(item_id)

    def set_learned_item_status(self, item_id: str, status: str) -> dict:
        if status not in {"active", "disabled"}:
            raise ValueError("Learned item status must be active or disabled.")
        with self._lock:
            with self._write_transaction():
                cursor = self.conn.execute(
                    "UPDATE agent_learned_items SET status = ?, "
                    "updated_at = CURRENT_TIMESTAMP WHERE item_id = ?",
                    (status, item_id),
                )
                if cursor.rowcount != 1:
                    raise ValueError("Learned item was not found.")
        return self.get_learned_item(item_id)

    def delete_learned_item(self, item_id: str) -> bool:
        with self._lock:
            with self._write_transaction():
                cursor = self.conn.execute(
                    "DELETE FROM agent_learned_items WHERE item_id = ?",
                    (item_id,),
                )
                return cursor.rowcount == 1

    def get_relevant_learned_items(
        self,
        task_type: str,
        limit: int = 5,
    ) -> list[dict]:
        if task_type not in {"coding", "research", "other"}:
            task_type = "other"
        limit = max(1, min(int(limit), 10))
        with self._lock:
            enabled = self.conn.execute(
                "SELECT setting_value FROM agent_learning_settings "
                "WHERE setting_key = 'enabled'"
            ).fetchone()
            if enabled is None or enabled[0] != "1":
                return []
            rows = self.conn.execute(
                "SELECT item_id FROM agent_learned_items "
                "WHERE status = 'active' "
                "AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP) "
                "AND (review_at IS NULL OR review_at > CURRENT_TIMESTAMP) "
                "AND scope IN ('global', ?) "
                "ORDER BY updated_at DESC LIMIT ?",
                (task_type, limit),
            ).fetchall()
        return [
            item for row in rows
            if (item := self.get_learned_item(row[0])) is not None
        ]

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
                "SELECT session_id FROM chat_history "
                "WHERE role NOT IN ('summary', 'skill') "
                "AND session_id IS NOT NULL GROUP BY session_id ORDER BY MAX(id) DESC"
            )
            rows = cursor.fetchall()
        return [row[0] for row in rows]

    def count_conversation_messages(self) -> int:
        """Count saved chat/tool messages, excluding summaries and skill links."""
        with self._lock:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM chat_history "
                "WHERE role NOT IN ('summary', 'skill')"
            ).fetchone()
        return int(row[0])

    def count_conversation_sessions(self) -> int:
        """Count sessions containing saved chat/tool messages."""
        with self._lock:
            row = self.conn.execute(
                "SELECT COUNT(DISTINCT session_id) FROM chat_history "
                "WHERE role NOT IN ('summary', 'skill') AND session_id IS NOT NULL"
            ).fetchone()
        return int(row[0])

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
                    self.conn.execute(
                        "DELETE FROM agent_learned_items WHERE source_task_id IN "
                        "(SELECT task_id FROM agent_tasks "
                        "WHERE originating_session_id = ?)",
                        (session_id,),
                    )
                    self.conn.execute(
                        "DELETE FROM agent_tasks WHERE originating_session_id = ?",
                        (session_id,),
                    )
                    self.conn.execute("DELETE FROM chat_history WHERE session_id = ?", (session_id,))
                else:
                    self.conn.execute("DELETE FROM agent_learned_items")
                    self.conn.execute("DELETE FROM agent_tasks")
                    self.conn.execute("DELETE FROM chat_history")

    def save_summary(self, session_id: str, summary: str):
        """Save an episodic session summary to the database."""
        summary = str(summary).strip()
        if not summary:
            return
        if len(summary) > MAX_SUMMARY_CHARS:
            raise ValueError(
                f"Episode summary exceeds the configured {MAX_SUMMARY_CHARS}-"
                "character limit; provide a schema-informed concise summary."
            )
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

    def prune_retained_records(
        self,
        *,
        conversation_days: int = 0,
        episode_days: int = 0,
        skill_days: int = 0,
    ) -> dict[str, object]:
        """Prune configured memory classes while retaining recovery-linked records."""
        days_by_class = {
            "conversation": max(0, int(conversation_days)),
            "episode": max(0, int(episode_days)),
            "skill": max(0, int(skill_days)),
        }
        deleted = {"conversation": 0, "episodes": 0, "skills": 0}
        skill_paths: list[str] = []
        resumable_statuses = (
            "draft", "awaiting_approval", "active", "paused", "interrupted", "stale"
        )
        with self._lock:
            with self._write_transaction():
                if days_by_class["conversation"]:
                    cutoff = (
                        datetime.now(timezone.utc)
                        - timedelta(days=days_by_class["conversation"])
                    ).strftime("%Y-%m-%d %H:%M:%S")
                    cursor = self.conn.execute(
                        "DELETE FROM chat_history WHERE timestamp < ? "
                        "AND role NOT IN ('summary', 'skill') "
                        "AND is_skill_file = 0",
                        (cutoff,),
                    )
                    deleted["conversation"] = cursor.rowcount
                if days_by_class["episode"]:
                    cutoff = (
                        datetime.now(timezone.utc)
                        - timedelta(days=days_by_class["episode"])
                    ).strftime("%Y-%m-%d %H:%M:%S")
                    cursor = self.conn.execute(
                        "DELETE FROM chat_history WHERE role = 'summary' "
                        "AND timestamp < ? AND (episode_task_id IS NULL OR "
                        "episode_task_id NOT IN (SELECT task_id FROM agent_tasks "
                        "WHERE status IN (?, ?, ?, ?, ?, ?)))",
                        (cutoff, *resumable_statuses),
                    )
                    deleted["episodes"] = cursor.rowcount
                if days_by_class["skill"]:
                    cutoff = (
                        datetime.now(timezone.utc)
                        - timedelta(days=days_by_class["skill"])
                    ).strftime("%Y-%m-%d %H:%M:%S")
                    skill_paths = [
                        str(row[0])
                        for row in self.conn.execute(
                            "SELECT skill_file_path FROM chat_history "
                            "WHERE is_skill_file = 1 AND timestamp < ? "
                            "AND (skill_task_id IS NULL OR skill_task_id NOT IN "
                            "(SELECT task_id FROM agent_tasks "
                            "WHERE status IN (?, ?, ?, ?, ?, ?))) "
                            "AND skill_file_path IS NOT NULL",
                            (cutoff, *resumable_statuses),
                        ).fetchall()
                    ]
                    cursor = self.conn.execute(
                        "DELETE FROM chat_history WHERE is_skill_file = 1 "
                        "AND timestamp < ? AND (skill_task_id IS NULL OR "
                        "skill_task_id NOT IN (SELECT task_id FROM agent_tasks "
                        "WHERE status IN (?, ?, ?, ?, ?, ?)))",
                        (cutoff, *resumable_statuses),
                    )
                    deleted["skills"] = cursor.rowcount
        return {**deleted, "skill_paths": skill_paths}

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

    def list_episodic_memories(self) -> list[dict[str, object]]:
        """Return all saved session summaries for user review."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT id, session_id, timestamp, content FROM chat_history "
                "WHERE role = 'summary' ORDER BY id"
            ).fetchall()
        return [
            {
                "id": row[0],
                "session_id": row[1],
                "timestamp": row[2],
                "content": row[3],
            }
            for row in rows
        ]

    def count_episodic_memories(self) -> int:
        with self._lock:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM chat_history WHERE role = 'summary'"
            ).fetchone()
        return int(row[0])

    def get_recent_episodic_memories(
        self,
        limit: int = DEFAULT_EPISODIC_SUMMARIES,
    ) -> list[dict[str, object]]:
        """Return a bounded chronological window of prior episode summaries."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT id, session_id, timestamp, content, episode_topic, "
                "episode_description, episode_task_summary, episode_plan_json, "
                "episode_outcome FROM chat_history "
                "WHERE role = 'summary' ORDER BY id DESC LIMIT ?",
                (max(0, limit),),
            ).fetchall()
        return [
            {
                "id": row[0],
                "session_id": row[1],
                "timestamp": row[2],
                "content": row[3],
                "topic": row[4],
                "description": row[5],
                "task_summary": row[6],
                "plan": self._decode_episode_plan(row[7], row[0]),
                "outcome": row[8],
            }
            for row in reversed(rows)
        ]

    def search_relevant_episodic_memories(
        self,
        query: str,
        limit: int = 3,
    ) -> list[dict[str, object]]:
        """Return a small deterministic lexical match set from recent episodes."""
        limit = max(1, min(int(limit), 100))
        stop_words = {
            "about", "after", "also", "and", "are", "for", "from", "have",
            "into", "its", "make", "need", "that", "the", "this", "with",
            "what", "when", "where", "which", "will", "would", "topic",
            "task", "plan", "steps", "verified",
        }
        query_terms = {
            term for term in re.findall(r"[a-z0-9]+", query.casefold())
            if len(term) > 2 and term not in stop_words
        }
        if not query_terms:
            return []
        with self._lock:
            rows = None
            if self._episode_fts_enabled:
                expression = " OR ".join(
                    f'"{term}"' for term in sorted(query_terms)
                )
                try:
                    rows = self.conn.execute(
                        "SELECT history.id, history.session_id, history.timestamp, "
                        "history.content, history.episode_topic, "
                        "history.episode_description, history.episode_task_summary, "
                        "history.episode_plan_json, history.episode_outcome "
                        "FROM agent_episode_fts "
                        "JOIN chat_history AS history "
                        "ON history.id = agent_episode_fts.rowid "
                        "WHERE agent_episode_fts MATCH ? "
                        "AND history.role = 'summary' "
                        "ORDER BY bm25(agent_episode_fts), history.id DESC LIMIT 500",
                        (expression,),
                    ).fetchall()
                except sqlite3.OperationalError as exc:
                    self._episode_fts_enabled = False
                    _LOGGER.warning(
                        "Episode FTS5 query failed; using lexical fallback: %s",
                        str(exc)[:200],
                    )
            if rows is None:
                rows = self.conn.execute(
                    "SELECT id, session_id, timestamp, content, episode_topic, "
                    "episode_description, episode_task_summary, episode_plan_json, "
                    "episode_outcome FROM chat_history WHERE role = 'summary' "
                    "ORDER BY id DESC LIMIT 500"
                ).fetchall()
        matches = []
        minimum_overlap = (
            1 if len(query_terms) == 1 else max(2, (len(query_terms) + 3) // 4)
        )
        for row in rows:
            searchable = " ".join(
                str(value or "")
                for value in (row[3], row[4], row[5], row[6])
            ).casefold()
            episode_terms = set(re.findall(r"[a-z0-9]+", searchable))
            overlap = query_terms & episode_terms
            if len(overlap) < minimum_overlap:
                continue
            matches.append(
                (
                    len(overlap) / len(query_terms),
                    len(overlap),
                    row[0],
                    {
                        "id": row[0],
                        "session_id": row[1],
                        "timestamp": row[2],
                        "content": row[3],
                        "topic": row[4],
                        "description": row[5],
                        "task_summary": row[6],
                        "plan": self._decode_episode_plan(row[7], row[0]),
                        "outcome": row[8],
                    },
                )
            )
        matches.sort(key=lambda item: (-item[0], -item[1], -item[2]))
        return [item[3] for item in matches[:limit]]

    def get_episodic_memory(self, entry_id: int) -> Optional[dict[str, object]]:
        """Return one saved session summary by its SQLite row ID."""
        with self._lock:
            row = self.conn.execute(
                "SELECT id, session_id, timestamp, content, episode_topic, "
                "episode_description, episode_task_summary, episode_plan_json, "
                "episode_outcome FROM chat_history "
                "WHERE id = ? AND role = 'summary'",
                (entry_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "session_id": row[1],
            "timestamp": row[2],
            "content": row[3],
            "topic": row[4],
            "description": row[5],
            "task_summary": row[6],
            "plan": self._decode_episode_plan(row[7], row[0]),
            "outcome": row[8],
        }

    def delete_episodic_memory(self, entry_id: int) -> bool:
        """Delete one summary row without deleting the associated conversation."""
        with self._lock:
            with self.conn:
                cursor = self.conn.execute(
                    "DELETE FROM chat_history WHERE id = ? AND role = 'summary'",
                    (entry_id,),
                )
                return cursor.rowcount == 1

    def register_skill_file(
        self,
        session_id: str,
        skill_file_path: str,
        description: str,
        *,
        source_task_id: Optional[str] = None,
    ) -> int:
        """Associate a successfully created skill artifact with its originating session."""
        with self._lock:
            with self.conn:
                cursor = self.conn.execute(
                    "INSERT INTO chat_history "
                    "(session_id, role, content, is_skill_file, skill_file_path, "
                    "skill_task_id) VALUES (?, 'skill', ?, ?, ?, ?)",
                    (
                        session_id,
                        description,
                        True,
                        skill_file_path,
                        source_task_id,
                    ),
                )
                return int(cursor.lastrowid)

    def delete_task_records(self, task_id: str) -> dict:
        """Delete a non-active task and its linked learning, episode, and skill metadata."""
        with self._lock:
            with self._write_transaction():
                task = self.conn.execute(
                    "SELECT status FROM agent_tasks WHERE task_id = ?",
                    (task_id,),
                ).fetchone()
                if task is None:
                    raise ValueError("Task not found.")
                if task[0] == "active":
                    raise ValueError(
                        "Pause or cancel an active task before deleting its records."
                    )
                skill_rows = self.conn.execute(
                    "SELECT id, skill_file_path FROM chat_history "
                    "WHERE is_skill_file = 1 AND skill_task_id = ?",
                    (task_id,),
                ).fetchall()
                episode_cursor = self.conn.execute(
                    "DELETE FROM chat_history WHERE role = 'summary' "
                    "AND episode_task_id = ?",
                    (task_id,),
                )
                self.conn.execute(
                    "DELETE FROM chat_history WHERE is_skill_file = 1 "
                    "AND skill_task_id = ?",
                    (task_id,),
                )
                self.conn.execute(
                    "DELETE FROM agent_learned_items WHERE source_task_id = ?",
                    (task_id,),
                )
                self.conn.execute(
                    "DELETE FROM agent_tasks WHERE task_id = ?",
                    (task_id,),
                )
        return {
            "task_id": task_id,
            "episodes_deleted": episode_cursor.rowcount,
            "skills_deleted": len(skill_rows),
            "skill_paths": [row[1] for row in skill_rows],
        }

    def get_task_linked_skill_paths(self, task_id: str) -> list[str]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT skill_file_path FROM chat_history "
                "WHERE is_skill_file = 1 AND skill_task_id = ? "
                "ORDER BY id",
                (task_id,),
            ).fetchall()
        return [str(row[0]) for row in rows if row[0]]

    def get_recent_skill_files(
        self,
        limit: int = DEFAULT_EPISODIC_SUMMARIES,
    ) -> list[dict[str, object]]:
        """Return recently registered skill artifacts and their source session."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT id, session_id, timestamp, content, skill_file_path "
                "FROM chat_history WHERE is_skill_file = 1 "
                "ORDER BY id DESC LIMIT ?",
                (max(0, limit),),
            ).fetchall()
        return [
            {
                "id": row[0],
                "session_id": row[1],
                "timestamp": row[2],
                "description": row[3],
                "path": row[4],
            }
            for row in rows
        ]

    def count_registered_skill_files(self) -> int:
        with self._lock:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM chat_history WHERE is_skill_file = 1"
            ).fetchone()
        return int(row[0])

    def maintain_database(self) -> dict[str, int | str]:
        """Check integrity, optimize query planning, and reclaim free SQLite pages."""
        with self._lock:
            self.conn.commit()
            result = self.conn.execute("PRAGMA integrity_check").fetchone()
            integrity = str(result[0]) if result else "no result"
            if integrity != "ok":
                raise sqlite3.DatabaseError(
                    f"SQLite integrity check failed: {integrity}"
                )
            pages_before = int(
                self.conn.execute("PRAGMA page_count").fetchone()[0]
            )
            self.conn.execute("PRAGMA optimize")
            self.conn.execute("VACUUM")
            pages_after = int(
                self.conn.execute("PRAGMA page_count").fetchone()[0]
            )
        return {
            "integrity": integrity,
            "pages_before": pages_before,
            "pages_after": pages_after,
        }

    def backup_to(self, destination: str) -> str:
        """Write a consistent private backup without overwriting an existing file."""
        target = pathlib.Path(destination).expanduser().absolute()
        if self.db_path != ":memory:" and target.resolve() == pathlib.Path(
            self.db_path
        ).resolve():
            raise ValueError("Backup destination must differ from the active database.")
        if target.exists():
            raise FileExistsError(f"Backup destination already exists: {target}")
        if not target.parent.is_dir():
            raise FileNotFoundError(
                f"Backup parent directory does not exist: {target.parent}"
            )
        descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
        )
        os.close(descriptor)
        staged = pathlib.Path(temp_name)
        try:
            with self._lock:
                destination_connection = sqlite3.connect(str(staged))
                try:
                    self.conn.backup(destination_connection)
                    result = destination_connection.execute(
                        "PRAGMA integrity_check"
                    ).fetchone()
                    if not result or result[0] != "ok":
                        raise sqlite3.DatabaseError(
                            "Backup integrity check failed."
                        )
                finally:
                    destination_connection.close()
            os.chmod(staged, 0o600)
            with staged.open("rb") as backup_file:
                os.fsync(backup_file.fileno())
            os.link(staged, target)
            return str(target)
        finally:
            staged.unlink(missing_ok=True)

    @classmethod
    def restore_backup(cls, source: str, destination: str) -> str:
        """Restore a valid SQLite backup into a new path, never replacing a database."""
        source_path = pathlib.Path(source).expanduser().resolve(strict=True)
        target = pathlib.Path(destination).expanduser().absolute()
        if not source_path.is_file():
            raise ValueError("Backup source must be a regular file.")
        if source_path == target.resolve():
            raise ValueError("Restore destination must differ from the backup source.")
        if target.exists():
            raise FileExistsError(f"Restore destination already exists: {target}")
        if not target.parent.is_dir():
            raise FileNotFoundError(
                f"Restore parent directory does not exist: {target.parent}"
            )
        descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
        )
        os.close(descriptor)
        staged = pathlib.Path(temp_name)
        source_connection = None
        destination_connection = None
        try:
            source_connection = sqlite3.connect(
                source_path.as_uri() + "?mode=ro",
                uri=True,
            )
            integrity = source_connection.execute("PRAGMA integrity_check").fetchone()
            if not integrity or integrity[0] != "ok":
                raise sqlite3.DatabaseError(
                    "Backup source failed SQLite integrity validation."
                )
            tables = {
                row[0]
                for row in source_connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            if "chat_history" not in tables:
                raise sqlite3.DatabaseError(
                    "Backup source is not a Private Agent SQLite database."
                )
            destination_connection = sqlite3.connect(str(staged))
            source_connection.backup(destination_connection)
            restored_integrity = destination_connection.execute(
                "PRAGMA integrity_check"
            ).fetchone()
            if not restored_integrity or restored_integrity[0] != "ok":
                raise sqlite3.DatabaseError(
                    "Restored database failed SQLite integrity validation."
                )
            destination_connection.close()
            destination_connection = None
            source_connection.close()
            source_connection = None
            os.chmod(staged, 0o600)
            with staged.open("rb") as restored_file:
                os.fsync(restored_file.fileno())
            os.link(staged, target)
            return str(target)
        finally:
            if destination_connection is not None:
                destination_connection.close()
            if source_connection is not None:
                source_connection.close()
            staged.unlink(missing_ok=True)

    def close(self):
        with self._lock:
            if not self._closed:
                self.conn.close()
                self._closed = True
