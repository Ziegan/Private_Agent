import asyncio
import json
import os
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from private_agent.database import PersistentMemory
from private_agent.tools import (
    delete_chat_history_entry_from_sqlite,
    delete_chat_history_from_sqlite,
    read_chat_history_from_sqlite,
    search_chat_history_in_sqlite,
)


def test_persistent_memory(temp_db):
    memory = PersistentMemory(db_path=temp_db)
    session_id = "test_session_1"

    memory.save_message(session_id, "human", "Hello agent")
    memory.save_message(session_id, "ai", "Hello human")

    history = memory.load_history(session_id)
    assert len(history) == 2
    assert history[0].content == "Hello agent"
    assert history[1].content == "Hello human"

    memory.save_summary(session_id, "Test summary of session.")
    summaries = memory.get_all_episodic_summaries()
    assert len(summaries) == 1
    assert summaries[0] == "Test summary of session."
    assert len(memory.load_history(session_id)) == 2
    assert memory.get_latest_session_id() == session_id
    assert memory.list_sessions() == [session_id]


def test_conversation_counts_exclude_episodic_summaries_and_skill_links(temp_db):
    memory = PersistentMemory(db_path=temp_db)
    memory.save_message("session-a", "human", "First message")
    memory.save_message("session-a", "ai", "First reply")
    memory.save_summary("session-a", "A saved episode")
    memory.register_skill_file("session-b", "/skills/example.md", "Skill description")

    assert memory.count_conversation_messages() == 2
    assert memory.count_conversation_sessions() == 1
    assert memory.count_episodic_memories() == 1
    assert memory.list_sessions() == ["session-a"]
    memory.close()


def test_task_plan_revision_approval_and_progress_survive_database_reopen(
    tmp_path,
):
    db_path = str(tmp_path / "task-state.sqlite")
    memory = PersistentMemory(db_path)
    task = memory.create_task_plan(
        "session-one",
        "Implement a CSV parser and tests",
        "coding",
        [{"step_id": "inspect", "description": "Inspect project conventions"}],
    )
    assert task["status"] == "awaiting_approval"
    assert task["steps"][0]["status"] == "pending"
    assert task["resume_capsule"]["goal"] == task["goal"]
    assert task["resume_capsule"]["next_safe_action"].startswith("Review")
    memory.close()
    memory = PersistentMemory(db_path)
    restored = memory.get_task(task["task_id"])
    assert restored is not None
    assert restored["goal"] == "Implement a CSV parser and tests"
    assert restored["digest"] == task["digest"]
    assert restored["steps"][0]["step_id"] == "inspect"
    assert restored["resume_capsule"]["revision"] == restored["revision"]

    active = memory.approve_task_plan(
        task["task_id"],
        task["revision"],
        task["digest"],
        "coding: approved output scope",
    )
    assert active["status"] == "active"
    reported = memory.update_task_step(
        task["task_id"],
        task["revision"],
        "inspect",
        "reported_done",
        "Reviewed existing parser and test layout.",
    )
    assert reported["steps"][0]["status"] == "reported_done"
    capsule = reported["resume_capsule"]
    assert capsule["todo"][0]["evidence_ref"]
    assert "Reviewed existing parser" not in repr(capsule)
    verified = memory.verify_task_step(
        task["task_id"],
        task["revision"],
        "inspect",
        "Confirmed via inspected project files.",
    )
    assert verified["steps"][0]["status"] == "verified"
    with pytest.raises(ValueError, match="requires concrete evidence"):
        memory.record_task_validation(task["task_id"], task["revision"], " ")
    with pytest.raises(ValueError, match="active task revision"):
        memory.record_task_validation(
            task["task_id"], task["revision"] + 1, "Tests passed."
        )
    validated = memory.record_task_validation(
        task["task_id"],
        task["revision"],
        "Automated unit tests passed; final checkpoint verified.",
    )
    assert validated["validation_revision"] == task["revision"]
    assert "Automated unit tests passed" in validated["validation_evidence"]
    assert memory.complete_task(task["task_id"])["status"] == "completed"
    assert memory.get_task(task["task_id"])["resume_capsule"][
        "next_safe_action"
    ].startswith("Task is completed")
    memory.close()


def test_count_task_plans_counts_only_current_revision_approvals(tmp_path):
    memory = PersistentMemory(str(tmp_path / "task-counts.sqlite"))
    pending = memory.create_task_plan(
        "session",
        "Pending plan",
        "research",
        [{"step_id": "inspect", "description": "Inspect sources"}],
    )
    approved = memory.create_task_plan(
        "session",
        "Approved plan",
        "research",
        [{"step_id": "search", "description": "Search sources"}],
    )
    memory.approve_task_plan(
        approved["task_id"],
        approved["revision"],
        approved["digest"],
        "research: goal=Approved plan",
    )

    assert memory.count_task_plans() == (2, 1)

    memory.revise_task_plan(
        approved["task_id"],
        "Revised approved plan",
        "research",
        [{"step_id": "search", "description": "Search updated sources"}],
        expected_revision=approved["revision"],
    )
    assert memory.count_task_plans() == (2, 0)
    assert memory.get_task(pending["task_id"])["status"] == "awaiting_approval"
    memory.close()


def test_task_pause_between_steps_survives_reopen_without_replaying_verified_work(
    tmp_path,
):
    db_path = str(tmp_path / "paused-between-steps.sqlite")
    memory = PersistentMemory(db_path)
    task = memory.create_task_plan(
        "session-before-restart",
        "Complete the staged implementation",
        "research",
        [
            {"step_id": "inspect", "description": "Inspect the existing behavior"},
            {"step_id": "compare", "description": "Compare candidate approaches"},
            {"step_id": "validate", "description": "Validate the selected approach"},
        ],
    )
    memory.approve_task_plan(
        task["task_id"], task["revision"], task["digest"], "research: approved"
    )
    memory.update_task_step(
        task["task_id"], task["revision"], "inspect", "reported_done",
        "Inspected the persisted project state.",
    )
    memory.verify_task_step(
        task["task_id"], task["revision"], "inspect",
        "Confirmed against the project files.",
    )
    memory.set_task_status(task["task_id"], "paused")
    memory.close()

    child_env = os.environ.copy()
    src_path = str(Path(__file__).resolve().parents[2] / "src")
    child_env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (src_path, child_env.get("PYTHONPATH", "")))
    )
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, sys; "
            "from private_agent.database import PersistentMemory; "
            "memory = PersistentMemory(sys.argv[1]); "
            "task = memory.get_task(sys.argv[2]); "
            "print(json.dumps({'status': task['status'], "
            "'steps': [step['status'] for step in task['steps']], "
            "'approved_revision': task['approved_revision']})); "
            "memory.close()",
            db_path,
            task["task_id"],
        ],
        check=True,
        capture_output=True,
        text=True,
        env=child_env,
    )
    cross_process_state = json.loads(child.stdout)
    assert cross_process_state == {
        "status": "paused",
        "steps": ["verified", "pending", "pending"],
        "approved_revision": None,
    }

    reopened = PersistentMemory(db_path)
    restored = reopened.get_task(task["task_id"])
    assert restored["status"] == "paused"
    assert [step["status"] for step in restored["steps"]] == [
        "verified",
        "pending",
        "pending",
    ]
    assert restored["approved_revision"] is None
    assert restored["approved_digest"] is None

    reviewed = reopened.set_task_status(task["task_id"], "awaiting_approval")
    assert reviewed["status"] == "awaiting_approval"
    assert [step["status"] for step in reviewed["steps"]] == [
        "verified",
        "pending",
        "pending",
    ]
    reapproved = reopened.approve_task_plan(
        task["task_id"],
        task["revision"],
        task["digest"],
        "research: reapproved after review",
    )
    assert reapproved["status"] == "active"
    assert reapproved["steps"][0]["status"] == "verified"
    reopened.close()


def test_task_plan_research_citations_persist_and_obey_plan_limits(temp_db):
    memory = PersistentMemory(db_path=temp_db)
    reference = (
        "Untrusted planning search citation: Local manual; "
        "URL: https://example.org/manual"
    )
    task = memory.create_task_plan(
        "session",
        "Research the local manual",
        "research",
        [{"step_id": "inspect", "description": "Review cited documentation"}],
        research_questions=["Which options are current?"],
        research_references=[reference],
    )

    assert memory.get_task(task["task_id"])["plan"]["research_references"] == [
        reference
    ]
    memory.approve_task_plan(
        task["task_id"],
        task["revision"],
        task["digest"],
        "research: exact current revision",
    )
    revised = memory.revise_task_plan(
        task["task_id"],
        task["goal"],
        task["task_type"],
        task["plan"]["steps"],
        expected_revision=task["revision"],
        research_questions=["Which deployment option is current?"],
        research_references=[reference],
    )
    assert revised["revision"] == task["revision"] + 1
    assert revised["approved_revision"] is None
    assert revised["plan"]["research_references"] == [reference]
    with pytest.raises(ValueError, match="1500 characters"):
        memory.create_task_plan(
            "session",
            "Research another manual",
            "research",
            [{"step_id": "inspect", "description": "Review cited documentation"}],
            research_references=["x" * 1501],
        )
    memory.close()


def test_task_validation_columns_migrate_existing_database(tmp_path):
    db_path = str(tmp_path / "legacy-task-validation.sqlite")
    memory = PersistentMemory(db_path)
    memory.close()

    connection = sqlite3.connect(db_path)
    connection.execute("ALTER TABLE agent_tasks DROP COLUMN validation_evidence")
    connection.execute("ALTER TABLE agent_tasks DROP COLUMN validation_revision")
    connection.commit()
    connection.close()

    memory = PersistentMemory(db_path)
    columns = {
        row[1]
        for row in memory.conn.execute("PRAGMA table_info(agent_tasks)").fetchall()
    }
    assert {"validation_revision", "validation_evidence"} <= columns
    memory.close()


def test_stale_task_expiry_clears_approval_but_preserves_recovery_state(temp_db):
    memory = PersistentMemory(db_path=temp_db)
    stale = memory.create_task_plan(
        "old-session",
        "Continue a previously paused task",
        "research",
        [{"step_id": "inspect", "description": "Inspect saved context"}],
    )
    memory.set_task_status(stale["task_id"], "paused")
    live = memory.create_task_plan(
        "live-session",
        "Continue a task with a live owner",
        "research",
        [{"step_id": "inspect", "description": "Inspect current work"}],
    )
    assert memory.acquire_task_lease(
        live["task_id"], "current-owner", "live-session", lease_seconds=300
    )
    memory.approve_task_plan(
        live["task_id"],
        live["revision"],
        live["digest"],
        "research: current plan",
        owner_id="current-owner",
    )
    old_timestamp = "2000-01-01 00:00:00"
    memory.conn.execute(
        "UPDATE agent_tasks SET updated_at = ?, checkpoint_at = ? "
        "WHERE task_id IN (?, ?)",
        (old_timestamp, old_timestamp, stale["task_id"], live["task_id"]),
    )
    memory.conn.commit()
    memory.close()

    child_env = os.environ.copy()
    src_path = str(Path(__file__).resolve().parents[2] / "src")
    child_env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (src_path, child_env.get("PYTHONPATH", "")))
    )
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, sys; "
            "from private_agent.database import PersistentMemory; "
            "memory = PersistentMemory(sys.argv[1]); "
            "print(json.dumps(memory.mark_stale_tasks(30))); "
            "memory.close()",
            temp_db,
        ],
        check=True,
        capture_output=True,
        text=True,
        env=child_env,
    )
    expired_ids = json.loads(child.stdout)
    memory = PersistentMemory(db_path=temp_db)

    assert expired_ids == [stale["task_id"]]
    saved_stale = memory.get_task(stale["task_id"])
    assert saved_stale["status"] == "stale"
    assert saved_stale["approved_revision"] is None
    assert saved_stale["resume_capsule"]["goal"] == stale["goal"]
    assert saved_stale["resume_capsule"]["revision"] == stale["revision"]
    assert memory.get_task(live["task_id"])["status"] == "active"
    reviewed = memory.set_task_status(stale["task_id"], "awaiting_approval")
    assert reviewed["status"] == "awaiting_approval"
    assert reviewed["approved_revision"] is None
    memory.close()


def test_failed_schema_migration_preserves_existing_database(tmp_path, monkeypatch):
    import private_agent.database.memory as memory_module

    db_path = str(tmp_path / "migration-failure.sqlite")
    connection = sqlite3.connect(db_path)
    connection.execute(
        "CREATE TABLE chat_history (id INTEGER PRIMARY KEY, session_id TEXT, "
        "role TEXT, content TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)"
    )
    connection.execute(
        "INSERT INTO chat_history (session_id, role, content) "
        "VALUES ('legacy-session', 'human', 'keep this source row')"
    )
    connection.commit()
    connection.close()

    original_connect = sqlite3.connect

    def deny_alter(*args, **kwargs):
        connection = original_connect(*args, **kwargs)
        connection.set_authorizer(
            lambda action, *_details: (
                sqlite3.SQLITE_DENY
                if action == sqlite3.SQLITE_ALTER_TABLE
                else sqlite3.SQLITE_OK
            )
        )
        return connection

    monkeypatch.setattr(memory_module.sqlite3, "connect", deny_alter)
    with pytest.raises(sqlite3.DatabaseError):
        PersistentMemory(db_path)
    monkeypatch.setattr(memory_module.sqlite3, "connect", original_connect)

    connection = sqlite3.connect(db_path)
    row = connection.execute(
        "SELECT session_id, role, content FROM chat_history"
    ).fetchone()
    columns = {
        item[1]
        for item in connection.execute("PRAGMA table_info(chat_history)")
    }
    connection.close()
    assert row == ("legacy-session", "human", "keep this source row")
    assert "is_skill_file" not in columns


def test_legacy_summary_only_row_survives_schema_migration(tmp_path):
    db_path = str(tmp_path / "legacy-summary.sqlite")
    connection = sqlite3.connect(db_path)
    connection.execute(
        "CREATE TABLE chat_history (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "session_id TEXT, role TEXT, content TEXT, "
        "timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, "
        "is_skill_file BOOLEAN NOT NULL DEFAULT 0, skill_file_path TEXT)"
    )
    connection.execute(
        "INSERT INTO chat_history (session_id, role, content) "
        "VALUES ('legacy-session', 'summary', 'summary with no structured fields')"
    )
    connection.commit()
    connection.close()

    memory = PersistentMemory(db_path)
    episode = memory.get_recent_episodic_memories(1)[0]

    assert episode["content"] == "summary with no structured fields"
    assert episode["topic"] is None
    assert episode["description"] is None
    assert episode["task_summary"] is None
    assert episode["plan"] is None
    memory.close()


def test_task_resume_capsule_enforces_configured_size_limit(tmp_path, monkeypatch):
    import private_agent.database.memory as memory_module

    monkeypatch.setattr(memory_module, "MAX_RESUME_STATE_CHARS", 256)
    memory = PersistentMemory(str(tmp_path / "resume-capsule-limit.sqlite"))
    with pytest.raises(ValueError, match="resume capsule exceeds"):
        memory.create_task_plan(
            "session",
            "Bounded task",
            "research",
            [{"step_id": "inspect", "description": "Inspect"}],
            constraints=["constraint " * 40],
        )
    assert memory.list_tasks() == []
    memory.close()


def test_malformed_task_records_log_bounded_diagnostics_and_do_not_load(
    tmp_path,
    caplog,
):
    memory = PersistentMemory(str(tmp_path / "malformed-task.sqlite"))
    task = memory.create_task_plan(
        "session",
        "Recover malformed plan",
        "research",
        [{"step_id": "inspect", "description": "Inspect the source"}],
    )
    memory.conn.execute(
        "UPDATE agent_tasks SET resume_state_json = ? WHERE task_id = ?",
        ("private malformed resume content", task["task_id"]),
    )
    memory.conn.commit()

    loaded = memory.get_task(task["task_id"])
    assert loaded["resume_capsule"] is None
    assert "Skipping malformed task resume capsule" in caplog.text
    memory.conn.execute(
        "UPDATE agent_plan_revisions SET plan_json = ? WHERE task_id = ?",
        ("private corrupted plan contents", task["task_id"]),
    )
    memory.conn.commit()
    with pytest.raises(ValueError, match="is malformed"):
        memory.get_task(task["task_id"])
    assert memory.list_tasks() == []
    assert "Malformed saved task plan" in caplog.text
    assert "private corrupted plan contents" not in caplog.text
    memory.close()


def test_task_lease_and_action_journal_require_exclusive_owner(tmp_path):
    db_path = str(tmp_path / "task-lease.sqlite")
    memory = PersistentMemory(db_path)
    task = memory.create_task_plan(
        "session",
        "Apply a reviewed change",
        "coding",
        [{"step_id": "change", "description": "Apply the change"}],
    )
    assert memory.acquire_task_lease(
        task["task_id"],
        "owner-one",
        "session-one",
        lease_seconds=30,
    )
    memory.approve_task_plan(
        task["task_id"],
        task["revision"],
        task["digest"],
        "coding: approved",
        owner_id="owner-one",
    )
    assert not memory.acquire_task_lease(
        task["task_id"],
        "owner-two",
        "session-two",
        lease_seconds=30,
        allow_takeover=True,
    )
    with pytest.raises(ValueError, match="another process"):
        memory.set_task_status(
            task["task_id"],
            "paused",
            owner_id="owner-two",
        )

    memory.begin_task_action(
        task["task_id"],
        "owner-one",
        "tool-call-1",
        "write_file",
        {"path": "/private/source.py", "content": "private content"},
    )
    action = memory.list_task_actions(task["task_id"])[0]
    assert action["status"] == "in_flight"
    assert "private content" not in str(action)
    memory.close()
    memory = PersistentMemory(db_path)
    assert memory.list_task_actions(task["task_id"])[0]["status"] == "in_flight"
    assert memory.reconcile_task_actions_after_restart() == 0

    memory.conn.execute(
        "UPDATE agent_task_leases SET expires_at = 0 WHERE task_id = ?",
        (task["task_id"],),
    )
    memory.conn.commit()
    assert memory.reconcile_task_actions_after_restart() == 1
    assert memory.list_task_actions(
        task["task_id"],
        unresolved_only=True,
    )[0]["status"] == "outcome_unknown"
    assert memory.acquire_task_lease(
        task["task_id"],
        "owner-two",
        "session-two",
        lease_seconds=30,
        allow_takeover=True,
    )
    with pytest.raises(ValueError, match="unresolved action"):
        memory.begin_task_action(
            task["task_id"],
            "owner-two",
            "tool-call-2",
            "write_file",
            {},
        )
    memory.resolve_task_action(
        action["action_id"],
        "not_executed",
        owner_id="owner-two",
    )
    second_action = memory.begin_task_action(
        task["task_id"],
        "owner-two",
        "tool-call-3",
        "read_file",
        {"path": "source.py"},
    )
    assert memory.finish_task_action(
        second_action,
        "outcome_observed",
        "digest",
    )["status"] == "outcome_observed"
    memory.close()


def test_task_record_deletion_removes_linked_episode_skills_and_learning(tmp_path):
    memory = PersistentMemory(str(tmp_path / "task-linked-delete.sqlite"))
    task = memory.create_task_plan(
        "session",
        "Complete a linked task",
        "research",
        [{"step_id": "inspect", "description": "Inspect local guidance"}],
    )
    memory.approve_task_plan(
        task["task_id"],
        task["revision"],
        task["digest"],
        "research: approved",
    )
    memory.update_task_step(
        task["task_id"],
        task["revision"],
        "inspect",
        "reported_done",
        "Read and reviewed the local guidance.",
    )
    memory.verify_task_step(
        task["task_id"],
        task["revision"],
        "inspect",
        "Confirmed the local source was inspected.",
    )
    completed = memory.complete_task(task["task_id"])
    memory.set_learning_enabled(True)
    memory.add_learned_item(
        "Prefer relevant local guidance.",
        "research",
        source_task_id=task["task_id"],
    )
    skill_path = str(tmp_path / "skills" / "research.md")
    memory.register_skill_file(
        "session",
        skill_path,
        "Research helper",
        source_task_id=task["task_id"],
    )
    memory.save_summary("unrelated-session", "Unrelated user episode")
    assert memory.count_episodic_memories() == 2
    assert memory.count_registered_skill_files() == 1

    with pytest.raises(ValueError, match="not found"):
        memory.delete_task_records("missing-task")
    removed = memory.delete_task_records(task["task_id"])

    assert removed["episodes_deleted"] == 1
    assert removed["skills_deleted"] == 1
    assert removed["skill_paths"] == [skill_path]
    assert memory.get_task(task["task_id"]) is None
    assert memory.count_episodic_memories() == 1
    assert memory.count_registered_skill_files() == 0
    assert memory.list_learned_items() == []
    assert memory.get_episodic_memory(completed["episode_id"]) is None
    memory.close()


@pytest.mark.asyncio
async def test_task_tool_call_is_not_run_when_action_journaling_fails(
    tmp_path,
    monkeypatch,
):
    import private_agent.agent.runtime as runtime

    memory = PersistentMemory(str(tmp_path / "task-journal.sqlite"))
    task = memory.create_task_plan(
        "session",
        "Run a task tool",
        "research",
        [{"step_id": "inspect", "description": "Inspect the source"}],
    )
    executed = False

    async def fake_execute(*_args, **_kwargs):
        nonlocal executed
        executed = True
        return runtime.ToolMessage(content="done", tool_call_id="call-1")

    monkeypatch.setattr(runtime, "execute_tool_call", fake_execute)
    blocked = await runtime._execute_task_tool_call(
        memory,
        task["task_id"],
        "missing-owner",
        {"id": "call-1", "name": "write_file", "args": {"content": "secret"}},
    )
    assert "journaling failed" in blocked.content
    assert not executed
    memory.close()


@pytest.mark.asyncio
async def test_uncertain_task_action_blocks_retries_until_user_reconciles(
    tmp_path,
    monkeypatch,
):
    import private_agent.agent.runtime as runtime

    memory = PersistentMemory(str(tmp_path / "task-action-recovery.sqlite"))
    task = memory.create_task_plan(
        "session",
        "Run an uncertain task tool",
        "research",
        [{"step_id": "inspect", "description": "Inspect the source"}],
    )
    owner_id = "task-owner"
    assert memory.acquire_task_lease(
        task["task_id"],
        owner_id,
        "session",
    )
    memory.approve_task_plan(
        task["task_id"],
        task["revision"],
        task["digest"],
        "research: approved",
        owner_id=owner_id,
    )
    calls = 0

    async def uncertain_execute(tool_call, **_kwargs):
        nonlocal calls
        calls += 1
        return runtime.ToolMessage(
            content="Error executing tool: connection closed after request",
            tool_call_id=tool_call["id"],
        )

    monkeypatch.setattr(runtime, "execute_tool_call", uncertain_execute)
    first = await runtime._execute_task_tool_call(
        memory,
        task["task_id"],
        owner_id,
        {"id": "uncertain-call", "name": "web_search", "args": {"query": "secret"}},
    )
    assert "Error executing tool" in first.content
    assert memory.list_task_actions(
        task["task_id"],
        unresolved_only=True,
    )[0]["status"] == "outcome_unknown"
    memory.update_task_step(
        task["task_id"],
        task["revision"],
        "inspect",
        "reported_done",
        "The plan step was reported complete.",
    )
    memory.verify_task_step(
        task["task_id"],
        task["revision"],
        "inspect",
        "Verified the plan step.",
        owner_id=owner_id,
    )
    with pytest.raises(ValueError, match="unresolved outcomes"):
        memory.complete_task(task["task_id"])

    retry = await runtime._execute_task_tool_call(
        memory,
        task["task_id"],
        owner_id,
        {"id": "retry-call", "name": "web_search", "args": {"query": "secret"}},
    )
    assert "unresolved action" in retry.content
    assert calls == 1

    action_id = memory.list_task_actions(task["task_id"])[0]["action_id"]
    memory.resolve_task_action(
        action_id,
        "verified",
        owner_id=owner_id,
    )

    async def resolved_execute(tool_call, **_kwargs):
        return runtime.ToolMessage(
            content="Search result observed",
            tool_call_id=tool_call["id"],
        )

    monkeypatch.setattr(runtime, "execute_tool_call", resolved_execute)
    resumed = await runtime._execute_task_tool_call(
        memory,
        task["task_id"],
        owner_id,
        {"id": "resolved-call", "name": "web_search", "args": {"query": "secret"}},
    )
    assert resumed.content == "Search result observed"
    resolved_action = next(
        action
        for action in memory.list_task_actions(task["task_id"])
        if action["tool_call_id"] == "resolved-call"
    )
    assert resolved_action["status"] == "outcome_observed"
    memory.close()


@pytest.mark.asyncio
async def test_cancelled_task_tool_is_journaled_unknown_and_blocks_next_action(
    tmp_path,
    monkeypatch,
):
    import private_agent.agent.runtime as runtime

    memory = PersistentMemory(str(tmp_path / "cancelled-action.sqlite"))
    task = memory.create_task_plan(
        "session",
        "Run a cancellable task action",
        "research",
        [{"step_id": "inspect", "description": "Inspect source material"}],
    )
    owner_id = "cancel-owner"
    assert memory.acquire_task_lease(task["task_id"], owner_id, "session")
    memory.approve_task_plan(
        task["task_id"],
        task["revision"],
        task["digest"],
        "research: approved",
        owner_id=owner_id,
    )

    async def cancelled_execute(*_args, **_kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(runtime, "execute_tool_call", cancelled_execute)
    with pytest.raises(asyncio.CancelledError):
        await runtime._execute_task_tool_call(
            memory,
            task["task_id"],
            owner_id,
            {"id": "cancelled-call", "name": "web_search", "args": {}},
        )
    assert memory.list_task_actions(
        task["task_id"],
        unresolved_only=True,
    )[0]["status"] == "outcome_unknown"
    memory.close()

    child_env = os.environ.copy()
    src_path = str(Path(__file__).resolve().parents[2] / "src")
    child_env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (src_path, child_env.get("PYTHONPATH", "")))
    )
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, sys; "
            "from private_agent.database import PersistentMemory; "
            "memory = PersistentMemory(sys.argv[1]); "
            "actions = memory.list_task_actions(sys.argv[2], unresolved_only=True); "
            "print(json.dumps([action['status'] for action in actions])); "
            "memory.close()",
            str(tmp_path / "cancelled-action.sqlite"),
            task["task_id"],
        ],
        check=True,
        capture_output=True,
        text=True,
        env=child_env,
    )
    assert json.loads(child.stdout) == ["outcome_unknown"]

    memory = PersistentMemory(str(tmp_path / "cancelled-action.sqlite"))
    assert memory.list_task_actions(
        task["task_id"],
        unresolved_only=True,
    )[0]["status"] == "outcome_unknown"

    async def should_not_execute(*_args, **_kwargs):
        pytest.fail("a subsequent task action must be blocked")

    monkeypatch.setattr(runtime, "execute_tool_call", should_not_execute)
    blocked = await runtime._execute_task_tool_call(
        memory,
        task["task_id"],
        owner_id,
        {"id": "next-call", "name": "web_search", "args": {}},
    )
    assert "unresolved action" in blocked.content
    memory.close()


def test_table_schema_reports_constraints_and_blocks_uninspected_planner_writes(
    temp_db,
):
    from private_agent.tools import (
        clear_schema_reads,
        create_task_plan as create_plan_tool,
        read_table_schema,
        set_active_db_path,
        set_active_task_context,
    )

    memory = PersistentMemory(temp_db)
    schema = memory.get_table_schema("agent_todo_steps")
    assert schema["table"] == "agent_todo_steps"
    assert any(column["name"] == "evidence" for column in schema["columns"])
    assert schema["foreign_keys"]
    assert "do not imply a maximum string length" in schema["length_limit_note"]
    with pytest.raises(ValueError, match="approved agent tables"):
        memory.get_table_schema("sqlite_master")
    memory.close()

    clear_schema_reads()
    set_active_task_context("schema-session")
    result = create_plan_tool.invoke(
        {
            "goal": "Create a schema-aware plan",
            "task_type": "other",
            "steps": [{"step_id": "one", "description": "First step"}],
        }
    )
    assert "Read the destination table schema" in result

    schema_result = read_table_schema.invoke({"table_name": "agent_tasks"})
    assert '"requested_table": "agent_tasks"' in schema_result
    assert '"agent_plan_revisions"' in schema_result
    assert "schema visibility does not grant write approval" in schema_result
    saved = create_plan_tool.invoke(
        {
            "goal": "Create a schema-aware plan",
            "task_type": "other",
            "steps": [{"step_id": "one", "description": "First step"}],
        }
    )
    assert "Plan saved in SQLite" in saved
    set_active_task_context(None)
    clear_schema_reads()

    missing_database = str(Path(temp_db).with_name("must-not-be-created.sqlite"))
    set_active_db_path(missing_database)
    failed_read = read_table_schema.invoke({"table_name": "agent_tasks"})
    assert "Error reading SQLite schema" in failed_read
    assert not Path(missing_database).exists()
    set_active_db_path(temp_db)


def test_completed_task_accepts_every_documented_subjective_rating(temp_db):
    memory = PersistentMemory(db_path=temp_db)
    for rating in range(6):
        task = memory.create_task_plan(
            f"session-{rating}",
            f"Task rated {rating}",
            "research",
            [{"step_id": "verify", "description": "Verify the task outcome"}],
        )
        memory.approve_task_plan(
            task["task_id"],
            task["revision"],
            task["digest"],
            "research: approved",
        )
        memory.update_task_step(
            task["task_id"],
            task["revision"],
            "verify",
            "reported_done",
            "Reviewed the task result.",
        )
        memory.verify_task_step(
            task["task_id"],
            task["revision"],
            "verify",
            "Confirmed the task result against the evidence.",
        )
        memory.record_task_validation(
            task["task_id"],
            task["revision"],
            "Cross-checked the task outcome.",
        )
        memory.complete_task(task["task_id"])

        rated = memory.rate_completed_task(task["task_id"], rating)
        assert rated["user_rating"] == rating

    with pytest.raises(ValueError, match="integer from 0 to 5"):
        memory.rate_completed_task(task["task_id"], 6)
    memory.close()


def test_task_revision_invalidates_approval_and_rejects_stale_progress(temp_db):
    memory = PersistentMemory(db_path=temp_db)
    task = memory.create_task_plan(
        "session",
        "Research safe parser patterns",
        "research",
        [{"step_id": "docs", "description": "Review local coding guidance"}],
    )
    memory.approve_task_plan(
        task["task_id"], task["revision"], task["digest"], "research: parser"
    )
    revised = memory.revise_task_plan(
        task["task_id"],
        task["goal"],
        task["task_type"],
        [
            {"step_id": "docs", "description": "Review local coding guidance"},
            {"step_id": "test", "description": "Verify recommendations"},
        ],
        expected_revision=task["revision"],
    )

    assert revised["revision"] == 2
    assert revised["status"] == "awaiting_approval"
    assert revised["approved_digest"] is None
    assert revised["validation_revision"] is None
    assert revised["validation_evidence"] is None
    with pytest.raises(ValueError, match="No active approval"):
        memory.update_task_step(
            task["task_id"], 2, "docs", "in_progress"
        )
    with pytest.raises(ValueError, match="changed since it was read"):
        memory.revise_task_plan(
            task["task_id"],
            task["goal"],
            task["task_type"],
            [{"step_id": "docs", "description": "Stale revision"}],
            expected_revision=1,
        )
    memory.close()


def test_structured_plan_metadata_persists_and_enforces_dependencies(temp_db):
    memory = PersistentMemory(db_path=temp_db)
    task = memory.create_task_plan(
        "session",
        "Build and verify a small feature",
        "coding",
        [
            {
                "step_id": "implement",
                "description": "Implement the feature",
                "validation": ["Run focused unit tests"],
                "proof": ["Test output shows all cases passing"],
                "risks": ["Existing behavior may regress"],
                "edge_cases": ["Empty input"],
            },
            {
                "step_id": "review",
                "description": "Review and verify the change",
                "dependencies": ["implement"],
                "validation": ["Run the full relevant test module"],
                "proof": ["Record the test result"],
            },
        ],
        assumptions=["The existing API should remain compatible"],
        constraints=["Do not add runtime dependencies"],
        research_questions=["Check local guidance before coding"],
    )
    memory.approve_task_plan(
        task["task_id"],
        task["revision"],
        task["digest"],
        "coding: approved test",
    )
    with pytest.raises(ValueError, match="dependencies must be reported complete"):
        memory.update_task_step(
            task["task_id"], 1, "review", "in_progress"
        )

    memory.update_task_step(
        task["task_id"],
        1,
        "implement",
        "reported_done",
        "Implementation and focused tests are complete.",
    )
    updated = memory.update_task_step(
        task["task_id"],
        1,
        "review",
        "in_progress",
    )
    assert updated["plan"]["assumptions"] == [
        "The existing API should remain compatible"
    ]
    assert updated["plan"]["constraints"] == ["Do not add runtime dependencies"]
    assert updated["plan"]["research_questions"] == [
        "Check local guidance before coding"
    ]
    assert updated["steps"][0]["edge_cases"] == ["Empty input"]
    assert updated["steps"][0]["validation"] == ["Run focused unit tests"]
    assert updated["steps"][1]["dependencies"] == ["implement"]
    memory.close()


@pytest.mark.parametrize(
    "steps, message",
    [
        (
            [
                {"step_id": "a", "description": "A", "dependencies": ["missing"]}
            ],
            "unknown dependencies",
        ),
        (
            [
                {"step_id": "a", "description": "A", "dependencies": ["b"]},
                {"step_id": "b", "description": "B", "dependencies": ["a"]},
            ],
            "must not contain a cycle",
        ),
    ],
)
def test_task_plan_rejects_invalid_dependency_graphs(temp_db, steps, message):
    memory = PersistentMemory(db_path=temp_db)
    with pytest.raises(ValueError, match=message):
        memory.create_task_plan("session", "Invalid plan", "other", steps)
    memory.close()


def test_verified_task_completion_creates_searchable_structured_episode(temp_db):
    memory = PersistentMemory(db_path=temp_db)
    redacted = memory._sanitize_episode_text(
        "api_key=private-token /home/user/private-project"
    )
    assert "private-token" not in redacted
    assert "/home/user/private-project" not in redacted
    assert "[redacted]" in redacted
    assert "[local path]" in redacted
    long_text = "x" * 3000
    assert memory._sanitize_episode_text(long_text) == long_text
    task = memory.create_task_plan(
        "session",
        "Implement a CSV parser",
        "coding",
        [
            {
                "step_id": "parser",
                "description": "Implement CSV parsing",
                "validation": ["Run parser unit tests"],
                "proof": ["All parser cases pass"],
            }
        ],
    )
    memory.approve_task_plan(
        task["task_id"],
        task["revision"],
        task["digest"],
        "coding: approved scope",
    )
    memory.update_task_step(
        task["task_id"],
        task["revision"],
        "parser",
        "reported_done",
        "Parser implementation and tests are complete.",
    )
    memory.verify_task_step(
        task["task_id"],
        task["revision"],
        "parser",
        "Focused parser test suite passed.",
    )
    memory.complete_task(task["task_id"])

    assert memory.count_episodic_memories() == 1
    recent = memory.get_recent_episodic_memories(1)[0]
    assert recent["topic"] == "Implement a CSV parser"
    assert recent["task_summary"] == "Implement a CSV parser"
    assert recent["plan"]["steps"][0]["validation"] == ["Run parser unit tests"]
    assert "Focused parser test suite passed" not in recent["content"]
    assert "Completed after all steps were verified" in recent["content"]
    assert memory.get_episodic_memory(recent["id"])["outcome"].startswith("Completed")
    matches = memory.search_relevant_episodic_memories(
        "implement CSV parser tests",
    )
    assert [episode["id"] for episode in matches] == [recent["id"]]
    assert memory.search_relevant_episodic_memories("unrelated astronomy topic") == []
    assert memory.list_learned_items() == []
    memory.close()


def test_learned_preferences_are_scoped_correctable_and_globally_disabled(temp_db):
    memory = PersistentMemory(db_path=temp_db)
    item = memory.add_learned_item(
        "Prefer focused tests before the full suite.",
        "coding",
    )
    assert memory.get_relevant_learned_items("coding") == [item]
    assert memory.get_relevant_learned_items("research") == []

    corrected = memory.update_learned_item(
        item["item_id"],
        "Prefer focused tests and report their exact result.",
        "global",
    )
    assert corrected["provenance"] == "user-corrected"
    assert memory.get_relevant_learned_items("research") == [corrected]
    memory.set_learned_item_status(item["item_id"], "disabled")
    assert memory.get_relevant_learned_items("coding") == []

    memory.set_learned_item_status(item["item_id"], "active")
    memory.set_learning_enabled(False)
    assert memory.get_relevant_learned_items("coding") == []
    with pytest.raises(ValueError, match="disabled"):
        memory.add_learned_item("A new preference", "global")
    memory.set_learning_enabled(True)
    assert memory.delete_learned_item(item["item_id"])
    assert memory.list_learned_items() == []
    memory.close()


def test_learned_item_expiry_review_and_configured_size_limit(temp_db, monkeypatch):
    import private_agent.database.memory as memory_module

    monkeypatch.setattr(memory_module, "MAX_LEARNED_ITEM_CHARS", 12)
    memory = PersistentMemory(db_path=temp_db)
    with pytest.raises(ValueError, match="1 to 12"):
        memory.add_learned_item("A statement that is too long", "global")

    item = memory.add_learned_item("Focused", "coding")
    assert item["expires_at"]
    assert item["review_at"]
    assert not item["expired"]
    assert not item["review_due"]
    assert memory.get_relevant_learned_items("coding") == [item]

    memory.conn.execute(
        "UPDATE agent_learned_items SET expires_at = datetime('now', '-1 day'), "
        "review_at = datetime('now', '-1 day') WHERE item_id = ?",
        (item["item_id"],),
    )
    memory.conn.commit()
    expired = memory.get_learned_item(item["item_id"])
    assert expired["expired"]
    assert expired["review_due"]
    assert memory.get_relevant_learned_items("coding") == []

    reviewed = memory.review_learned_item(item["item_id"])
    assert not reviewed["expired"]
    assert not reviewed["review_due"]
    assert memory.get_relevant_learned_items("coding") == [reviewed]
    memory.close()


@pytest.mark.parametrize(
    "sensitive_statement",
    [
        "Store api_key = abc123",
        "My email is person@example.com",
        "My phone is (415) 555-0134",
        "My SSN is 123-45-6789",
        "Use /home/private-user/project",
        "-----BEGIN PRIVATE KEY-----",
    ],
)
def test_learned_preferences_reject_credentials_and_sensitive_data(
    temp_db,
    sensitive_statement,
):
    memory = PersistentMemory(db_path=temp_db)
    with pytest.raises(ValueError, match="cannot contain credentials"):
        memory.add_learned_item(sensitive_statement, "global")
    with pytest.raises(ValueError, match="cannot contain credentials"):
        memory.update_learned_item("missing", sensitive_statement, "global")
    memory.close()


def test_learned_preference_accepts_non_sensitive_dates(temp_db):
    memory = PersistentMemory(db_path=temp_db)
    item = memory.add_learned_item(
        "Review project decisions after 2026-10-04.",
        "global",
    )
    assert item["statement"] == "Review project decisions after 2026-10-04."
    memory.close()


def test_episodic_search_uses_index_when_available_and_lexical_fallback(temp_db):
    memory = PersistentMemory(db_path=temp_db)
    memory.save_summary("session-a", "Nebula search engine error handling")
    memory.save_summary("session-b", "Unrelated botanical greenhouse notes")

    matches = memory.search_relevant_episodic_memories(
        "nebula search engine",
    )
    assert [item["content"] for item in matches] == [
        "Nebula search engine error handling"
    ]
    if memory._episode_fts_enabled:
        assert memory.conn.execute(
            "SELECT COUNT(*) FROM agent_episode_fts"
        ).fetchone()[0] == 2

    nebula_id = memory.conn.execute(
        "SELECT id FROM chat_history WHERE content LIKE 'Nebula%'"
    ).fetchone()[0]
    assert memory.delete_episodic_memory(nebula_id)
    assert memory.search_relevant_episodic_memories("nebula search engine") == []

    memory._episode_fts_enabled = False
    memory.save_summary("session-c", "Nebula search engine lexical fallback")
    fallback = memory.search_relevant_episodic_memories("nebula search engine")
    assert [item["content"] for item in fallback] == [
        "Nebula search engine lexical fallback"
    ]
    memory.close()


@pytest.mark.parametrize("force_fallback", [False, True])
def test_episode_retrieval_ranks_best_overlap_first(temp_db, force_fallback):
    memory = PersistentMemory(db_path=temp_db)
    memory.save_summary("low", "Local migration guidance.")
    memory.save_summary("mid", "Local SQLite migration guidance.")
    memory.save_summary("best", "Local SQLite schema migration guidance.")
    if force_fallback:
        memory._episode_fts_enabled = False

    results = memory.search_relevant_episodic_memories(
        "local sqlite schema migration", limit=3
    )

    assert [item["session_id"] for item in results] == ["best", "mid", "low"]
    memory.close()


def test_corrupt_structured_episode_plan_degrades_with_diagnostic(temp_db, caplog):
    memory = PersistentMemory(db_path=temp_db)
    memory.save_summary("session", "legacy-compatible episode text")
    episode_id = memory.conn.execute(
        "SELECT id FROM chat_history WHERE role = 'summary'"
    ).fetchone()[0]
    memory.conn.execute(
        "UPDATE chat_history SET episode_plan_json = ? WHERE id = ?",
        ("{broken", episode_id),
    )

    episode = memory.get_recent_episodic_memories(1)[0]

    assert episode["content"] == "legacy-compatible episode text"
    assert episode["plan"] is None
    assert f"id={episode_id}" in caplog.text
    memory.close()


def test_episodic_storage_supports_in_memory_database():
    memory = PersistentMemory(":memory:")
    memory.save_summary("ephemeral", "in-memory legacy summary")
    episode = memory.get_recent_episodic_memories(1)[0]
    assert episode["content"] == "in-memory legacy summary"
    assert episode["id"]
    assert episode["timestamp"]
    assert memory.search_relevant_episodic_memories(
        "in-memory legacy summary"
    )[0]["id"] == episode["id"]
    assert memory.delete_episodic_memory(episode["id"])
    assert memory.search_relevant_episodic_memories(
        "in-memory legacy summary"
    ) == []
    memory.close()


def test_sqlite_memory_reset_also_removes_persisted_task_state(temp_db):
    memory = PersistentMemory(db_path=temp_db)
    task = memory.create_task_plan(
        "session",
        "A task",
        "other",
        [{"step_id": "step-1", "description": "Do the work"}],
    )

    memory.clear_history()

    assert memory.get_task(task["task_id"]) is None
    assert memory.list_tasks() == []
    memory.close()


def test_memory_constructor_closes_connection_when_initialization_fails(
    tmp_path, monkeypatch
):
    import private_agent.database.memory as memory_module

    class BrokenConnection:
        closed = False

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, *_args):
            raise sqlite3.DatabaseError("corrupt database")

        def close(self):
            self.closed = True

    connection = BrokenConnection()
    monkeypatch.setattr(
        memory_module.sqlite3,
        "connect",
        lambda *_args, **_kwargs: connection,
    )
    with pytest.raises(sqlite3.DatabaseError, match="corrupt database"):
        PersistentMemory(db_path=str(tmp_path / "broken.sqlite"))
    assert connection.closed


def test_memory_summary_retrieval_is_bounded_and_chronological(temp_db):
    memory = PersistentMemory(db_path=temp_db)
    for index in range(8):
        memory.save_summary(f"session-{index}", f"summary-{index}")
    assert memory.get_all_episodic_summaries(limit=3) == [
        "summary-5",
        "summary-6",
        "summary-7",
    ]
    memory.close()
    memory.close()


def test_memory_summaries_are_session_scoped(temp_db):
    memory = PersistentMemory(db_path=temp_db)
    memory.save_summary("session-a", "private summary")
    memory.save_summary("session-b", "other session")
    assert memory.get_all_episodic_summaries(session_id="session-a") == [
        "private summary"
    ]
    memory.close()


def test_memory_retention_prunes_messages_and_summaries(temp_db):
    memory = PersistentMemory(db_path=temp_db)
    memory.save_message("old", "human", "old")
    memory.save_summary("old", "old summary")
    memory.save_message("new", "human", "new")
    with memory.conn:
        memory.conn.execute(
            "UPDATE chat_history SET timestamp = '2000-01-01 00:00:00' "
            "WHERE session_id = 'old'"
        )
    assert memory.prune_history(30) == 2
    assert memory.load_history("old") == []
    assert memory.get_all_episodic_summaries(session_id="old") == []
    assert [message.content for message in memory.load_history("new")] == ["new"]
    memory.close()


def test_memory_summaries_are_bounded(temp_db, monkeypatch):
    monkeypatch.setattr("private_agent.database.memory.MAX_SUMMARY_CHARS", 12)
    memory = PersistentMemory(db_path=temp_db)
    with pytest.raises(ValueError, match="configured 12-character limit"):
        memory.save_summary("session", "x" * 100)
    memory.save_summary("session", "fits limit")
    assert memory.get_all_episodic_summaries(session_id="session") == [
        "fits limit"
    ]
    memory.close()


def test_episodic_memory_review_delete_and_database_maintenance(temp_db):
    memory = PersistentMemory(db_path=temp_db)
    memory.save_message("session-a", "human", "keep this conversation")
    memory.save_summary("session-a", "episode to review")
    memory.save_summary("session-b", "another episode")

    entries = memory.list_episodic_memories()
    assert len(entries) == 2
    assert entries[0]["session_id"] == "session-a"
    first_id = int(entries[0]["id"])
    assert memory.get_episodic_memory(first_id)["content"] == "episode to review"
    assert memory.get_episodic_memory(-1) is None

    assert memory.delete_episodic_memory(first_id)
    assert not memory.delete_episodic_memory(first_id)
    assert [entry["content"] for entry in memory.list_episodic_memories()] == [
        "another episode"
    ]
    assert [message.content for message in memory.load_history("session-a")] == [
        "keep this conversation"
    ]

    maintenance = memory.maintain_database()
    assert maintenance["integrity"] == "ok"
    assert maintenance["pages_after"] > 0
    memory.close()


def test_database_backup_and_restore_use_new_private_paths(tmp_path):
    source = tmp_path / "private-memory.sqlite"
    backup = tmp_path / "backup.sqlite"
    restored = tmp_path / "restored.sqlite"
    memory = PersistentMemory(str(source))
    memory.save_message("session", "human", "private conversation")
    memory.save_summary("session", "verified episode")
    memory.set_learning_enabled(True)
    learned_item = memory.add_learned_item("Prefer concise reports", "global")

    assert memory.backup_to(str(backup)) == str(backup)
    assert backup.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        memory.backup_to(str(backup))
    with pytest.raises(ValueError, match="differ"):
        memory.backup_to(str(source))

    restored_path = PersistentMemory.restore_backup(str(backup), str(restored))
    reopened = PersistentMemory(restored_path)
    assert reopened.load_history("session")[0].content == "private conversation"
    assert reopened.get_all_episodic_summaries(session_id="session") == [
        "verified episode"
    ]
    assert reopened.learning_is_enabled()
    assert reopened.get_learned_item(learned_item["item_id"])["statement"] == (
        "Prefer concise reports"
    )
    assert restored.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        PersistentMemory.restore_backup(str(backup), str(restored))
    memory.close()
    reopened.close()


def test_database_restore_rejects_corrupt_and_unrelated_files_without_overwrite(
    tmp_path,
):
    corrupt = tmp_path / "corrupt.sqlite"
    destination = tmp_path / "restored.sqlite"
    corrupt.write_bytes(b"not a SQLite database")

    with pytest.raises(sqlite3.DatabaseError):
        PersistentMemory.restore_backup(str(corrupt), str(destination))

    assert corrupt.read_bytes() == b"not a SQLite database"
    assert not destination.exists()
    assert not list(tmp_path.glob(".restored.sqlite.*.tmp"))


def test_skill_files_are_linked_to_sessions_and_counted(temp_db, tmp_path):
    memory = PersistentMemory(db_path=temp_db)
    skill_path = tmp_path / "skills" / "review.md"
    skill_path.parent.mkdir()
    skill_path.write_text("# Review skill\nInspect, test, and report.", encoding="utf-8")

    entry_id = memory.register_skill_file(
        "origin-session", str(skill_path), "Review a code change"
    )

    assert memory.count_registered_skill_files() == 1
    skill = memory.get_recent_skill_files(1)[0]
    assert skill["id"] == entry_id
    assert skill["session_id"] == "origin-session"
    assert skill["description"] == "Review a code change"
    assert skill["path"] == str(skill_path)
    row = memory.conn.execute(
        "SELECT role, is_skill_file, skill_file_path FROM chat_history WHERE id = ?",
        (entry_id,),
    ).fetchone()
    assert tuple(row) == ("skill", 1, str(skill_path))
    memory.close()


def test_database_concurrency(temp_db):
    memory = PersistentMemory(db_path=temp_db)
    session_id = "concurrent_session"

    def write_task(index):
        memory.save_message(session_id, "human", f"Message {index}")

    with ThreadPoolExecutor(max_workers=5) as executor:
        list(executor.map(write_task, range(10)))

    assert len(memory.load_history(session_id)) == 10


def test_memory_retention_prunes_by_record_type_and_preserves_resumable_links(
    tmp_path,
):
    memory = PersistentMemory(str(tmp_path / "retention.sqlite"))
    resumable = memory.create_task_plan(
        "session",
        "Resume the pending deployment research",
        "research",
        [{"step_id": "inspect", "description": "Inspect deployment notes"}],
    )
    memory.set_task_status(resumable["task_id"], "stale")
    memory.save_message("session", "human", "expired conversation")
    memory.save_summary("session", "unlinked expired episode")
    memory.save_summary("session", "resumable task episode")
    memory.register_skill_file(
        "session", str(tmp_path / "preserved.md"), "resumable skill",
        source_task_id=resumable["task_id"],
    )
    memory.register_skill_file(
        "session", str(tmp_path / "expired.md"), "expired skill"
    )
    old_timestamp = "2000-01-01 00:00:00"
    memory.conn.execute(
        "UPDATE chat_history SET timestamp = ?",
        (old_timestamp,),
    )
    memory.conn.execute(
        "UPDATE chat_history SET episode_task_id = ? "
        "WHERE role = 'summary' AND content = 'resumable task episode'",
        (resumable["task_id"],),
    )
    memory.conn.commit()

    result = memory.prune_retained_records(
        conversation_days=30,
        episode_days=30,
        skill_days=30,
    )

    assert result["conversation"] == 1
    assert result["episodes"] == 1
    assert result["skills"] == 1
    assert result["skill_paths"] == [str(tmp_path / "expired.md")]
    assert memory.get_task(resumable["task_id"])["status"] == "stale"
    assert [
        item["content"] for item in memory.list_episodic_memories()
    ] == ["resumable task episode"]
    assert memory.search_relevant_episodic_memories(
        "unlinked expired episode"
    ) == []
    assert memory.get_recent_skill_files(5)[0]["description"] == "resumable skill"
    memory.close()


def test_sqlite_concurrent_read_write_locking(temp_db):
    memory = PersistentMemory(db_path=temp_db)
    session_id = "lock_test_session"

    def worker(index):
        if index % 2 == 0:
            memory.save_message(session_id, "human", f"Message {index}")
        else:
            read_chat_history_from_sqlite.invoke({"limit": 5})

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(worker, range(20)))

    assert memory.load_history(session_id)


def test_sqlite_exhausted_write_lock_surfaces_without_losing_data(temp_db):
    memory = PersistentMemory(db_path=temp_db)
    blocker = sqlite3.connect(temp_db, timeout=0)
    try:
        blocker.execute("BEGIN IMMEDIATE")
        memory.conn.execute("PRAGMA busy_timeout = 25")

        with pytest.raises(sqlite3.OperationalError, match="locked"):
            memory.save_message("lock_test_session", "human", "blocked write")

        blocker.rollback()
        assert memory.load_history("lock_test_session") == []
        memory.save_message("lock_test_session", "human", "retry after lock")
        assert [
            message.content
            for message in memory.load_history("lock_test_session")
        ] == ["retry after lock"]
    finally:
        blocker.close()
        memory.close()


def test_sqlite_history_tools(temp_db):
    memory = PersistentMemory(db_path=temp_db)
    memory.save_message("session_a", "human", "Test message for sqlite tools")

    read_result = read_chat_history_from_sqlite.invoke({"limit": 5})
    assert "Test message for sqlite tools" in read_result

    delete_result = delete_chat_history_from_sqlite.invoke(
        {"session_id": "session_a"}
    )
    assert "Success" in delete_result

    wipe_result = delete_chat_history_from_sqlite.invoke({})
    assert "Success" in wipe_result


def test_sqlite_history_search_is_bounded_and_session_scoped(temp_db):
    memory = PersistentMemory(db_path=temp_db)
    memory.save_message("session-a", "human", "Project Juniper has a local database.")
    memory.save_message("session-b", "human", "Juniper is a different session.")
    memory.save_summary("session-a", "Juniper summary for the project.")

    all_results = search_chat_history_in_sqlite.invoke(
        {"query": "juniper", "limit": 2}
    )
    scoped_results = search_chat_history_in_sqlite.invoke(
        {"query": "juniper", "session_id": "session-a", "limit": 10}
    )

    assert all_results.count("Entry ID:") == 2
    assert "Juniper summary" in all_results
    assert "session-b" not in scoped_results
    assert scoped_results.count("Entry ID:") == 2
    assert "Session: session-a" in scoped_results
    memory.close()


def test_sqlite_history_entry_delete_requires_matching_session(temp_db):
    memory = PersistentMemory(db_path=temp_db)
    memory.save_message("session-a", "human", "specific record")
    entry_id = memory.conn.execute(
        "SELECT id FROM chat_history WHERE session_id = 'session-a'"
    ).fetchone()[0]
    memory.save_message("session-b", "human", "unrelated record")

    mismatch = delete_chat_history_entry_from_sqlite.invoke(
        {"entry_id": entry_id, "session_id": "session-b"}
    )
    assert "No matching" in mismatch
    assert len(memory.load_history("session-a")) == 1

    deleted = delete_chat_history_entry_from_sqlite.invoke(
        {"entry_id": entry_id, "session_id": "session-a"}
    )
    assert "Deleted SQLite history entry" in deleted
    assert memory.load_history("session-a") == []
    assert [message.content for message in memory.load_history("session-b")] == [
        "unrelated record"
    ]
    memory.close()


def test_runtime_requires_typed_confirmation_for_selective_history_delete(
    temp_db, monkeypatch
):
    from unittest.mock import MagicMock

    import private_agent.agent.runtime as agent

    memory = PersistentMemory(db_path=temp_db)
    memory.save_message("session-a", "human", "reviewed item")
    entry_id = memory.conn.execute(
        "SELECT id FROM chat_history WHERE session_id = 'session-a'"
    ).fetchone()[0]
    memory.close()

    monkeypatch.setattr(agent.sys, "stdin", MagicMock(isatty=lambda: True))
    monkeypatch.setattr(agent.console, "input", lambda _prompt: "delete 999")
    declined = asyncio.run(
        agent.execute_tool_call(
            {
                "name": "delete_chat_history_entry_from_sqlite",
                "args": {"entry_id": entry_id, "session_id": "session-a"},
                "id": "delete-declined",
            }
        )
    )
    assert "declined" in declined.content
    memory = PersistentMemory(db_path=temp_db)
    assert len(memory.load_history("session-a")) == 1
    memory.close()

    monkeypatch.setattr(
        agent.console,
        "input",
        lambda _prompt: f"delete {entry_id}",
    )
    confirmed = asyncio.run(
        agent.execute_tool_call(
            {
                "name": "delete_chat_history_entry_from_sqlite",
                "args": {"entry_id": entry_id, "session_id": "session-a"},
                "id": "delete-confirmed",
            }
        )
    )
    assert "Deleted SQLite history entry" in confirmed.content
    memory = PersistentMemory(db_path=temp_db)
    assert memory.load_history("session-a") == []
    memory.close()
