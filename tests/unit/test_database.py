from concurrent.futures import ThreadPoolExecutor

from private_agent.database import PersistentMemory
from private_agent.tools import (
    delete_chat_history_from_sqlite,
    read_chat_history_from_sqlite,
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
    memory.save_summary("session", "x" * 100)
    assert memory.get_all_episodic_summaries(session_id="session") == ["x" * 12]
    memory.close()


def test_database_concurrency(temp_db):
    memory = PersistentMemory(db_path=temp_db)
    session_id = "concurrent_session"

    def write_task(index):
        memory.save_message(session_id, "human", f"Message {index}")

    with ThreadPoolExecutor(max_workers=5) as executor:
        list(executor.map(write_task, range(10)))

    assert len(memory.load_history(session_id)) == 10


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
