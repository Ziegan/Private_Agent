"""Prompt context, citation formatting, and context-budget helpers."""

import logging
from datetime import datetime
from functools import lru_cache

from ..config import MAX_CONTEXT_TOKENS
from ..run_logging import RUN_LOGGER, log_event


_TOKENIZER_FALLBACKS_REPORTED: set[str] = set()


def _report_tokenizer_fallback(stage: str, exc: Exception) -> None:
    """Trace a tokenizer fallback once per stage without logging prompt text."""
    if stage in _TOKENIZER_FALLBACKS_REPORTED:
        return
    _TOKENIZER_FALLBACKS_REPORTED.add(stage)
    log_event(
        RUN_LOGGER,
        "prompt.tokenizer_fallback",
        level=logging.WARNING,
        stage=stage,
        error_type=type(exc).__name__,
        fallback="character_estimate",
    )


def current_datetime_context() -> str:
    now = datetime.now().astimezone()
    return (
        "Current local date and time from the system clock: "
        f"{now:%A, %B} {now.day}, {now.year}; "
        f"{now:%H:%M:%S %Z} (UTC{now:%z}). "
        "Use this as the authoritative current date/time when answering; "
        "do not infer it from model knowledge or conversation history."
    )


@lru_cache(maxsize=1)
def _get_tokenizer():
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception as exc:
        _report_tokenizer_fallback("load", exc)
        return None


def estimate_context_window(
    chat_history: list,
    current_input: str,
    max_context: int = MAX_CONTEXT_TOKENS,
    system_prompts: list | None = None,
) -> dict:
    total_chars = len(current_input)
    system_text = "\n".join(
        str(message.content) for message in (system_prompts or [])
    )
    total_chars += len(system_text)
    for message in chat_history:
        content = message.content
        content_text = (
            "".join(str(item) for item in content)
            if isinstance(content, list)
            else str(content)
        )
        total_chars += len(content_text)

    encoding = _get_tokenizer()
    if encoding is not None:
        try:
            estimated_tokens = len(
                encoding.encode(
                    system_text
                    + current_input
                    + "".join(str(message.content) for message in chat_history)
                )
            )
        except Exception as exc:
            _report_tokenizer_fallback("estimate", exc)
            estimated_tokens = int(total_chars / 3.5)
    else:
        estimated_tokens = int(total_chars / 3.5)

    percentage = min(100.0, (estimated_tokens / max_context) * 100)
    return {
        "tokens": estimated_tokens,
        "max": max_context,
        "percent": round(percentage, 2),
    }


def trim_history_to_context_budget(
    chat_history: list,
    current_input: str,
    max_context_tokens: int = MAX_CONTEXT_TOKENS,
) -> list:
    """Retain the newest whole messages that fit alongside the current request."""
    encoding = _get_tokenizer()
    if encoding is not None:

        def count(text: str) -> int:
            try:
                return len(encoding.encode(text))
            except Exception as exc:
                _report_tokenizer_fallback("trim", exc)
                return max(1, len(text) // 4)
    else:

        def count(text: str) -> int:
            return max(1, len(text) // 4)

    remaining = max(0, max_context_tokens - count(current_input))
    selected = []
    for message in reversed(chat_history):
        content = message.content
        text = (
            "".join(str(item) for item in content)
            if isinstance(content, list)
            else str(content)
        )
        cost = count(text)
        if cost > remaining:
            break
        selected.append(message)
        remaining -= cost
    return list(reversed(selected))


def token_count(text: str) -> int:
    encoding = _get_tokenizer()
    if encoding is None:
        return max(1, len(text) // 4)
    try:
        return len(encoding.encode(text))
    except Exception as exc:
        _report_tokenizer_fallback("count", exc)
        return max(1, len(text) // 4)


def calculate_prompt_budgets(
    system_prompts: list,
    context_window: int,
    max_output_tokens: int,
) -> tuple[int, int]:
    """Reserve context for system instructions and generation before user context."""
    output_budget = min(max_output_tokens, max(1, context_window // 3))
    system_text = "\n".join(str(message.content) for message in system_prompts)
    safety_margin = max(16, context_window // 100)
    input_budget = (
        context_window - token_count(system_text) - output_budget - safety_margin
    )
    return max(0, input_budget), output_budget


def _truncate_to_tokens(text: str, budget: int) -> str:
    if budget <= 0 or not text:
        return ""
    encoding = _get_tokenizer()
    if encoding is not None:
        try:
            return encoding.decode(encoding.encode(text)[:budget])
        except Exception as exc:
            _report_tokenizer_fallback("truncate", exc)
            return text[: budget * 4]
    return text[: budget * 4]


def build_bounded_user_input(
    context_label: str,
    episodic_context: str,
    rag_context: str,
    user_input: str,
    budget: int = MAX_CONTEXT_TOKENS,
) -> str:
    prefix = f"{context_label}:\n"
    suffix = f"\n\n[User Query]: {user_input}"
    query_prefix = f"{prefix}\n\n[User Query]: "
    if token_count(query_prefix) > budget:
        bounded_query = _truncate_to_tokens(user_input, budget)
        while token_count(bounded_query) > budget and bounded_query:
            bounded_query = _truncate_to_tokens(
                bounded_query, max(0, token_count(bounded_query) - 1)
            )
        return bounded_query
    if token_count(query_prefix + user_input) > budget:
        query_budget = max(0, budget - token_count(query_prefix))
        bounded_query = _truncate_to_tokens(user_input, query_budget)
        result = query_prefix + bounded_query
        while token_count(result) > budget and bounded_query:
            bounded_query = _truncate_to_tokens(
                bounded_query, max(0, token_count(bounded_query) - 1)
            )
            result = query_prefix + bounded_query
        return result
    remaining = max(0, budget - token_count(prefix + suffix) - 8)
    rag_budget = min(remaining, max(0, int(remaining * 0.7)))
    bounded_rag = _truncate_to_tokens(rag_context, rag_budget)
    remaining -= token_count(bounded_rag) if bounded_rag else 0
    bounded_episodic = _truncate_to_tokens(episodic_context, remaining)
    result = f"{prefix}{bounded_episodic}\n{bounded_rag}{suffix}"
    while token_count(result) > budget:
        if bounded_rag:
            bounded_rag = _truncate_to_tokens(
                bounded_rag, max(0, token_count(bounded_rag) - 1)
            )
        elif bounded_episodic:
            bounded_episodic = _truncate_to_tokens(
                bounded_episodic, max(0, token_count(bounded_episodic) - 1)
            )
        else:
            return prefix + suffix
        result = f"{prefix}{bounded_episodic}\n{bounded_rag}{suffix}"
    return result
