"""Configurable text chunking strategies for the local RAG index."""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

from ..run_logging import RUN_LOGGER, log_event

STRATEGIES = (
    "auto",
    "fixed",
    "character",
    "word",
    "token",
    "sentence",
    "paragraph",
    "line",
    "recursive",
    "markdown",
    "code",
    "semantic",
)
DEFAULT_SEPARATORS = ("\n\n", "\n", ". ", " ")
MARKDOWN_EXTENSIONS = frozenset({".md", ".markdown"})
CODE_EXTENSIONS = frozenset({".py", ".rs", ".js", ".ts", ".html", ".json"})

_SENTENCE_RE = re.compile(r"(?<=[.!?])[\"')\]]*\s+|\n{2,}")
_TOKEN_RE = re.compile(r"\w+|[^\w\s]")
_HEADING_RE = re.compile(r"^(#{1,6})\s+\S", re.MULTILINE)
_CODE_BOUNDARY_RE = re.compile(
    r"^(?:async\s+def |def |class |fn |pub fn |pub struct |impl\b|function |"
    r"export |const \w+\s*=\s*(?:async\s*)?\(|@\w)",
    re.MULTILINE,
)

EmbedFunction = Callable[[List[str]], List[List[float]]]


@dataclass(frozen=True)
class ChunkingSettings:
    strategy: str = "character"
    chunk_size: int = 1200  # characters, words or tokens depending on `unit`
    chunk_overlap: int = 200
    unit: str = "chars"  # chars | words | tokens (token = word or punctuation mark)
    min_chunk_chars: int = 0
    separators: Sequence[str] = field(default=DEFAULT_SEPARATORS)
    semantic_breakpoint_percentile: float = 90.0
    semantic_buffer_sentences: int = 1
    semantic_max_sentences: int = 2000

    def __post_init__(self) -> None:
        if self.strategy not in STRATEGIES:
            raise ValueError(
                f"Unknown chunking strategy '{self.strategy}'; "
                f"choose one of: {', '.join(STRATEGIES)}."
            )
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be at least 1.")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size.")

    def signature(self) -> str:
        """Stable string used to re-index files when chunking settings change."""
        return "|".join(
            str(value)
            for value in (
                self.strategy,
                self.chunk_size,
                self.chunk_overlap,
                self.unit,
                self.min_chunk_chars,
                "".join(self.separators).encode("unicode_escape").decode(),
                self.semantic_breakpoint_percentile,
                self.semantic_buffer_sentences,
                self.semantic_max_sentences,
            )
        )


def _size(text: str, unit: str) -> int:
    if unit == "words":
        return len(text.split())
    if unit == "tokens":
        return len(_TOKEN_RE.findall(text))
    return len(text)


def _window_split(text: str, size: int, overlap: int, snap_to_word: bool) -> List[str]:
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if snap_to_word and end < len(text):
            boundary = text.rfind(" ", start + size // 2, end)
            if boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _unit_window_split(text: str, size: int, overlap: int, unit: str) -> List[str]:
    """Sliding window over words or tokens, preserving the original spacing."""
    pattern = re.compile(r"\S+") if unit == "words" else _TOKEN_RE
    spans = [match.span() for match in pattern.finditer(text)]
    chunks: List[str] = []
    start = 0
    while start < len(spans):
        end = min(start + size, len(spans))
        chunk = text[spans[start][0]:spans[end - 1][1]].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(spans):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _pack(
    pieces: Sequence[str],
    settings: ChunkingSettings,
    joiner: str,
) -> List[str]:
    """Greedily pack pieces up to chunk_size, carrying trailing overlap forward."""
    chunks: List[str] = []
    current: List[str] = []
    current_size = 0
    join_size = _size(joiner, settings.unit) if joiner else 0

    def flush() -> None:
        text = joiner.join(current).strip()
        if text:
            chunks.append(text)

    for piece in pieces:
        piece_size = _size(piece, settings.unit)
        if piece_size > settings.chunk_size:
            if current:
                flush()
                current, current_size = [], 0
            chunks.extend(_split_oversized(piece, settings))
            continue
        extra = join_size if current else 0
        if current and current_size + extra + piece_size > settings.chunk_size:
            flush()
            carried: List[str] = []
            carried_size = 0
            for previous in reversed(current):
                previous_size = _size(previous, settings.unit)
                if carried_size + previous_size > settings.chunk_overlap:
                    break
                carried.insert(0, previous)
                carried_size += previous_size + join_size
            if carried_size + piece_size > settings.chunk_size:
                carried, carried_size = [], 0
            current, current_size = carried, carried_size
            extra = join_size if current else 0
        current.append(piece)
        current_size += extra + piece_size
    if current:
        flush()
    return chunks


def _split_oversized(text: str, settings: ChunkingSettings) -> List[str]:
    if settings.unit == "chars":
        return _window_split(text, settings.chunk_size, settings.chunk_overlap, True)
    return _unit_window_split(
        text, settings.chunk_size, settings.chunk_overlap, settings.unit
    )


def _split_sentences(text: str) -> List[str]:
    sentences: List[str] = []
    position = 0
    for match in _SENTENCE_RE.finditer(text):
        sentence = text[position:match.start() + len(match.group().rstrip())]
        if sentence.strip():
            sentences.append(sentence.strip())
        position = match.end()
    tail = text[position:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def _recursive_split(
    text: str, settings: ChunkingSettings, separators: Sequence[str]
) -> List[str]:
    if _size(text, settings.unit) <= settings.chunk_size:
        stripped = text.strip()
        return [stripped] if stripped else []
    for index, separator in enumerate(separators):
        if separator and separator in text:
            split_parts = text.split(separator)
            parts = [part + separator for part in split_parts[:-1]] + split_parts[-1:]
            remaining = separators[index + 1:]
            pieces: List[str] = []
            for part in parts:
                if _size(part, settings.unit) > settings.chunk_size:
                    pieces.extend(_recursive_split(part, settings, remaining))
                elif part.strip():
                    pieces.append(part)
            return _pack(pieces, settings, "")
    return _split_oversized(text, settings)


def _split_markdown(text: str, settings: ChunkingSettings) -> List[str]:
    starts = [match.start() for match in _HEADING_RE.finditer(text)]
    if not starts:
        return _recursive_split(text, settings, settings.separators)
    boundaries = ([0] if starts[0] != 0 else []) + starts + [len(text)]
    sections = [
        text[begin:end]
        for begin, end in zip(boundaries, boundaries[1:])
        if text[begin:end].strip()
    ]
    chunks: List[str] = []
    buffer = ""
    for section in sections:
        if _size(section, settings.unit) > settings.chunk_size:
            if buffer.strip():
                chunks.append(buffer.strip())
                buffer = ""
            heading = section.split("\n", 1)[0].strip()
            for part in _recursive_split(section, settings, settings.separators):
                chunks.append(part if part.startswith(heading) else f"{heading}\n{part}")
        elif buffer and _size(buffer + section, settings.unit) > settings.chunk_size:
            chunks.append(buffer.strip())
            buffer = section
        else:
            buffer += section
    if buffer.strip():
        chunks.append(buffer.strip())
    return chunks


def _split_code(text: str, settings: ChunkingSettings) -> List[str]:
    starts = [match.start() for match in _CODE_BOUNDARY_RE.finditer(text)]
    if not starts:
        return _recursive_split(text, settings, ("\n\n", "\n", " "))
    boundaries = ([0] if starts[0] != 0 else []) + starts + [len(text)]
    blocks = [
        text[begin:end]
        for begin, end in zip(boundaries, boundaries[1:])
        if text[begin:end].strip()
    ]
    pieces: List[str] = []
    for block in blocks:
        if _size(block, settings.unit) > settings.chunk_size:
            pieces.extend(_recursive_split(block, settings, ("\n\n", "\n", " ")))
        else:
            pieces.append(block)
    return _pack(pieces, settings, "")


def _cosine_distance(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return 1.0 - dot / norm if norm else 1.0


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * min(max(percentile, 0.0), 100.0) / 100.0
    lower, upper = math.floor(rank), math.ceil(rank)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def _split_semantic(
    text: str, settings: ChunkingSettings, embed: Optional[EmbedFunction]
) -> List[str]:
    """Split where embedding distance between neighbouring sentences spikes."""
    sentences = _split_sentences(text)
    if (
        embed is None
        or len(sentences) < 3
        or len(sentences) > settings.semantic_max_sentences
    ):
        return _pack(sentences, settings, " ")
    buffer = settings.semantic_buffer_sentences
    windows = [
        " ".join(sentences[max(0, i - buffer):i + buffer + 1])
        for i in range(len(sentences))
    ]
    try:
        vectors = embed(windows)
    except Exception as exc:
        log_event(
            RUN_LOGGER,
            "rag.semantic_chunking_failed",
            level=logging.WARNING,
            error_type=type(exc).__name__,
            fallback="sentence_chunking",
        )
        return _pack(sentences, settings, " ")
    if len(vectors) != len(sentences):
        return _pack(sentences, settings, " ")
    distances = [
        _cosine_distance(vectors[i], vectors[i + 1]) for i in range(len(vectors) - 1)
    ]
    threshold = _percentile(distances, settings.semantic_breakpoint_percentile)
    groups: List[List[str]] = [[sentences[0]]]
    for sentence, distance in zip(sentences[1:], distances):
        if distance > threshold:
            groups.append([sentence])
        else:
            groups[-1].append(sentence)
    chunks: List[str] = []
    for group in groups:
        joined = " ".join(group)
        if _size(joined, settings.unit) <= settings.chunk_size:
            chunks.append(joined)
        else:
            chunks.extend(_pack(group, settings, " "))
    return chunks


def _merge_small(chunks: List[str], minimum: int) -> List[str]:
    if minimum <= 0 or len(chunks) < 2:
        return chunks
    merged: List[str] = []
    for chunk in chunks:
        if merged and len(chunk) < minimum:
            merged[-1] = f"{merged[-1]}\n{chunk}"
        else:
            merged.append(chunk)
    if len(merged) > 1 and len(merged[0]) < minimum:
        merged[1] = f"{merged[0]}\n{merged[1]}"
        merged.pop(0)
    return merged


def resolve_strategy(strategy: str, extension: str = "") -> str:
    if strategy != "auto":
        return strategy
    extension = extension.lower()
    if extension in MARKDOWN_EXTENSIONS:
        return "markdown"
    if extension in CODE_EXTENSIONS:
        return "code"
    return "recursive"


def split_text(
    text: str,
    settings: ChunkingSettings,
    extension: str = "",
    embed: Optional[EmbedFunction] = None,
) -> List[str]:
    """Split text using the configured strategy."""
    if not text or not text.strip():
        return []
    strategy = resolve_strategy(settings.strategy, extension)
    size, overlap, unit = settings.chunk_size, settings.chunk_overlap, settings.unit
    if strategy == "fixed":
        chunks = (
            _window_split(text, size, overlap, False)
            if unit == "chars"
            else _unit_window_split(text, size, overlap, unit)
        )
    elif strategy == "character":
        chunks = (
            _window_split(text, size, overlap, True)
            if unit == "chars"
            else _unit_window_split(text, size, overlap, unit)
        )
    elif strategy in ("word", "token"):
        chunks = _unit_window_split(
            text, size, overlap, "words" if strategy == "word" else "tokens"
        )
    elif strategy == "sentence":
        chunks = _pack(_split_sentences(text), settings, " ")
    elif strategy == "paragraph":
        chunks = _pack(re.split(r"\n\s*\n", text), settings, "\n\n")
    elif strategy == "line":
        chunks = _pack(text.splitlines(), settings, "\n")
    elif strategy == "recursive":
        chunks = _recursive_split(text, settings, settings.separators)
    elif strategy == "markdown":
        chunks = _split_markdown(text, settings)
    elif strategy == "code":
        chunks = _split_code(text, settings)
    else:
        chunks = _split_semantic(text, settings, embed)
    return _merge_small([chunk for chunk in chunks if chunk.strip()], settings.min_chunk_chars)
