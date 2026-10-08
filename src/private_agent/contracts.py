"""Provider-neutral contracts shared across agent subsystems."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Mapping, Optional


@dataclass(frozen=True)
class ModelCapabilities:
    """Capabilities reported for a selected model; unknown is not support."""

    tools: Optional[bool] = None
    function_calls: Optional[bool] = None
    structured_output: Optional[bool] = None
    thinking: Optional[bool] = None
    vision: Optional[bool] = None
    audio: Optional[bool] = None
    computer_use: Optional[bool] = None
    context_window: Optional[int] = None


@dataclass(frozen=True)
class ModelMessage:
    """Provider-neutral text message sent to or returned by a model."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str


@dataclass(frozen=True)
class ModelRequest:
    """Bounded model invocation input independent of provider SDK classes."""

    messages: tuple[ModelMessage, ...]
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_output_tokens: Optional[int] = None
    timeout_seconds: Optional[float] = None


@dataclass(frozen=True)
class TokenUsage:
    """Token usage reported by a provider when available."""

    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


@dataclass(frozen=True)
class ToolCall:
    """A normalized model-requested tool invocation."""

    call_id: str
    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolResult:
    """Provider-neutral outcome for a normalized tool invocation."""

    call_id: str
    content: str
    is_error: bool = False


@dataclass(frozen=True)
class ModelResponse:
    """Provider-neutral model output with optional usage and tool calls."""

    content: str
    model: Optional[str] = None
    usage: Optional[TokenUsage] = None
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True)
class Evidence:
    """A traceable source excerpt or location supporting a claim."""

    source: str
    retrieved_at: datetime
    excerpt: Optional[str] = None
    locator: Optional[str] = None
    confidence: Optional[float] = None


@dataclass(frozen=True)
class MemoryRecord:
    """Provider- and storage-neutral record for attributable agent memory."""

    record_id: str
    kind: Literal["episodic", "semantic", "procedural", "repository"]
    content: str
    created_at: datetime
    scope: Optional[str] = None
    expires_at: Optional[datetime] = None
    evidence: tuple[Evidence, ...] = ()


class ExecutionError(RuntimeError):
    """Structured subsystem failure safe to classify across boundaries."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if not code.strip():
            raise ValueError("ExecutionError code must not be empty.")
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.details = dict(details or {})
