"""Session statistics collected during a run and printed as a table on exit."""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Tuple

from rich.table import Table

from .compaction import message_text
from .prompts import token_count


def _int_or_none(value: Any) -> Optional[int]:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def response_usage(response: Any) -> Tuple[Optional[int], Optional[int]]:
    """Return provider-reported (input, output) tokens when available."""
    usage = getattr(response, "usage_metadata", None)
    usage = usage if isinstance(usage, dict) else {}
    input_tokens = _int_or_none(usage.get("input_tokens"))
    output_tokens = _int_or_none(usage.get("output_tokens"))
    if input_tokens is None or output_tokens is None:
        metadata = getattr(response, "response_metadata", None)
        metadata = metadata if isinstance(metadata, dict) else {}
        alt = metadata.get("token_usage", metadata.get("usage", {}))
        alt = alt if isinstance(alt, dict) else {}
        if input_tokens is None:
            input_tokens = _int_or_none(
                alt.get("prompt_tokens", alt.get("input_tokens", metadata.get("prompt_eval_count")))
            )
        if output_tokens is None:
            output_tokens = _int_or_none(
                alt.get("completion_tokens", alt.get("output_tokens", metadata.get("eval_count")))
            )
    return input_tokens, output_tokens


@dataclass
class SessionStats:
    started: float = field(default_factory=time.monotonic)
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_calls: int = 0
    model_seconds: float = 0.0
    turns: int = 0
    tool_calls: int = 0
    tools: Counter = field(default_factory=Counter)
    models: Counter = field(default_factory=Counter)
    summary_tokens: int = 0
    summary_seconds: float = 0.0
    summary_saved: bool = False

    def record_model_call(
        self,
        messages: Iterable[Any],
        response: Any,
        elapsed: float,
        model_name: str = "",
    ) -> None:
        """Count one model call, estimating tokens the provider did not report."""
        reported_in, reported_out = response_usage(response)
        if reported_in is None or reported_out is None:
            self.estimated_calls += 1
        if reported_in is None:
            reported_in = sum(token_count(message_text(m)) for m in messages)
        if reported_out is None:
            reported_out = token_count(message_text(response))
        self.model_calls += 1
        self.input_tokens += reported_in
        self.output_tokens += reported_out
        self.model_seconds += max(0.0, elapsed)
        if model_name:
            self.models[model_name] += 1

    def record_tool_call(self, name: str) -> None:
        self.tool_calls += 1
        self.tools[name] += 1

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def session_seconds(self) -> float:
        return time.monotonic() - self.started


def format_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, secs = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m {secs}s" if hours else f"{minutes}m {secs}s"


def build_stats_table(stats: SessionStats, info: Dict[str, Any]) -> Table:
    """Build the statistics table; `info` holds descriptive session fields."""
    table = Table(title="Session Statistics", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", overflow="fold")

    def add(label: str, value: Any) -> None:
        table.add_row(label, str(value))

    for label, value in info.items():
        add(label, value)
    add("Total session time", format_duration(stats.session_seconds()))
    add("Time in model calls", format_duration(stats.model_seconds))
    add("Prompts this session", stats.turns)
    add("Model calls", stats.model_calls)
    add("Input tokens", f"{stats.input_tokens:,}")
    add("Output tokens", f"{stats.output_tokens:,}")
    add("Total tokens", f"{stats.total_tokens:,}")
    if stats.estimated_calls:
        add(
            "Token accuracy",
            f"{stats.estimated_calls} of {stats.model_calls} call(s) estimated "
            "(provider did not report usage)",
        )
    tool_detail = ", ".join(f"{n} x{c}" for n, c in stats.tools.most_common())
    add("Tool uses", f"{stats.tool_calls}" + (f" ({tool_detail})" if tool_detail else ""))
    if stats.models:
        add("Models used", ", ".join(f"{n} ({c} calls)" for n, c in stats.models.items()))
    add("Summary tokens generated", f"{stats.summary_tokens:,}")
    add("Summary generation time", format_duration(stats.summary_seconds))
    add("Summary stored in SQLite", "yes" if stats.summary_saved else "no")
    return table
