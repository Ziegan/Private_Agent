import asyncio

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from private_agent.agent.compaction import (
    compact_history,
    compact_loop_messages,
    messages_tokens,
    needs_history_compaction,
    summary_system_message,
)


class FakeLLM:
    def __init__(self, text="SUMMARY", fail=False):
        self.text, self.fail, self.calls = text, fail, []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        if self.fail:
            raise RuntimeError("down")
        return AIMessage(content=self.text)


def history(count):
    return [
        (HumanMessage if i % 2 == 0 else AIMessage)(content=f"message {i} " + "word " * 50)
        for i in range(count)
    ]


def test_needs_compaction_by_tokens_count_and_disabled():
    h = history(4)
    assert not needs_history_compaction(h, 10, 100000, 0.8, 20)
    assert needs_history_compaction(h, 10, 100, 0.8, 20)
    assert needs_history_compaction(history(20), 10, 100000, 0.8, 20)
    assert not needs_history_compaction(h, 10, 100, 0, 20)
    assert not needs_history_compaction([], 10, 100, 0.8, 20)


def test_compact_history_keeps_recent_and_updates_summary():
    llm, notes = FakeLLM("NEW SUMMARY"), []
    h = history(10)
    recent, summary = asyncio.run(
        compact_history(llm, h, "old", keep_recent=4, max_chars=100, notify=notes.append)
    )
    assert recent == h[-4:] and summary == "NEW SUMMARY"
    assert "old" in llm.calls[0][1].content and "message 0" in llm.calls[0][1].content
    assert len(notes) == 2


def test_compact_history_failure_falls_back_without_losing_summary():
    recent, summary = asyncio.run(
        compact_history(FakeLLM(fail=True), history(10), "keep", keep_recent=4, max_chars=100)
    )
    assert len(recent) == 4 and summary == "keep"


def test_summary_message():
    assert summary_system_message("") is None
    assert "abc" in summary_system_message("abc").content


def test_loop_compaction_summarizes_older_and_truncates_newest():
    big = "data " * 400
    messages = [
        SystemMessage(content="sys"),
        HumanMessage(content="q"),
        AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "1"}]),
        ToolMessage(content=big, tool_call_id="1"),
        AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "2"}]),
        ToolMessage(content=big, tool_call_id="2"),
    ]
    before = messages_tokens(messages)
    changed = asyncio.run(
        compact_loop_messages(
            FakeLLM("digest"), messages, before // 2, threshold=0.8, max_chars=100
        )
    )
    assert changed
    assert messages[3].content.startswith("[Condensed tool results] digest")
    assert messages_tokens(messages) < before
    assert messages[3].tool_call_id == "1"


def test_loop_compaction_noop_when_small_or_disabled():
    messages = [HumanMessage(content="hi")]
    assert not asyncio.run(
        compact_loop_messages(FakeLLM(), messages, 10000, threshold=0.8, max_chars=100)
    )
    assert not asyncio.run(
        compact_loop_messages(FakeLLM(), messages, 1, threshold=0, max_chars=100)
    )
