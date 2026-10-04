from langchain_core.messages import AIMessage, HumanMessage

from private_agent.agent.session_stats import (
    SessionStats,
    build_stats_table,
    format_duration,
    response_usage,
)


def test_reported_usage_is_used_and_missing_usage_estimated():
    stats = SessionStats()
    reported = AIMessage(
        content="x", usage_metadata={"input_tokens": 10, "output_tokens": 4, "total_tokens": 14}
    )
    stats.record_model_call([HumanMessage(content="hi")], reported, 1.0, "m")
    assert (stats.input_tokens, stats.output_tokens, stats.estimated_calls) == (10, 4, 0)
    stats.record_model_call([HumanMessage(content="hello world")], AIMessage(content="ok"), 0.5)
    assert stats.estimated_calls == 1 and stats.model_calls == 2
    assert stats.total_tokens > 14


def test_ollama_metadata_and_tool_counts():
    msg = AIMessage(content="x", response_metadata={"prompt_eval_count": 7, "eval_count": 3})
    assert response_usage(msg) == (7, 3)
    stats = SessionStats()
    stats.record_tool_call("web_search")
    stats.record_tool_call("web_search")
    table = build_stats_table(stats, {"Model": "m"})
    cells = [c for col in table.columns for c in col._cells]
    assert any("web_search x2" in c for c in cells)


def test_format_duration():
    assert format_duration(5) == "5.00s"
    assert format_duration(125) == "2m 5s"
    assert format_duration(3725) == "1h 2m 5s"
