import pytest

from private_agent.rag.chunking import _TOKEN_RE, STRATEGIES, ChunkingSettings, split_text

PROSE = " ".join(f"Sentence number {i} talks about topic {i % 3}." for i in range(60))


def settings(strategy, size=200, overlap=40, **kwargs):
    return ChunkingSettings(strategy=strategy, chunk_size=size, chunk_overlap=overlap, **kwargs)


@pytest.mark.parametrize("strategy", [s for s in STRATEGIES if s != "semantic"])
def test_every_strategy_covers_text_and_respects_size(strategy):
    chunks = split_text(PROSE, settings(strategy), ".txt")
    assert len(chunks) > 1
    if strategy == "word":
        assert all(len(chunk.split()) <= 200 for chunk in chunks)
    elif strategy == "token":
        assert all(len(_TOKEN_RE.findall(chunk)) <= 200 for chunk in chunks)
    else:
        assert all(len(chunk) <= 260 for chunk in chunks)
    assert "Sentence number 0" in chunks[0] and "number 59" in chunks[-1]


def test_fixed_is_exact_windows_with_overlap():
    chunks = split_text("a" * 100, settings("fixed", 40, 10))
    assert [len(c) for c in chunks] == [40, 40, 40]


def test_word_and_token_units():
    words = split_text(" ".join(map(str, range(50))), settings("word", 10, 2))
    assert all(len(c.split()) <= 10 for c in words)
    tokens = split_text("a, b, c, d, e, f", settings("token", 4, 0))
    assert len(tokens) == 3


def test_unit_words_applies_to_sentence_packing():
    chunks = split_text(PROSE, settings("sentence", 20, 0, unit="words"))
    assert all(len(c.split()) <= 20 for c in chunks)


def test_paragraph_and_line_keep_boundaries():
    text = "alpha one\n\nbeta two\n\ngamma three"
    assert split_text(text, settings("paragraph", 12, 0)) == ["alpha one", "beta two", "gamma three"]
    assert split_text("l1\nl2\nl3", settings("line", 5, 0)) == ["l1\nl2", "l3"]


def test_markdown_keeps_headings_with_sections():
    text = "# A\n" + "x " * 30 + "\n# B\nshort body\n"
    chunks = split_text(text, settings("markdown", 70, 0))
    assert chunks[0].startswith("# A")
    assert any(c.startswith("# B") for c in chunks)


def test_code_splits_on_definitions_and_auto_selects():
    code = "\n".join(f"def f{i}():\n    return {i}\n" for i in range(20))
    chunks = split_text(code, settings("code", 80, 0))
    assert all(c.startswith("def") for c in chunks)
    assert split_text(code, settings("auto", 80, 0), ".py") == chunks


def test_overlap_between_sentence_chunks():
    chunks = split_text(PROSE, settings("sentence", 200, 80))
    assert chunks[0].split(". ")[-1][:20] in chunks[1] or chunks[0][-30:] in chunks[1]


def test_min_chunk_chars_merges_small_pieces():
    chunks = split_text("aaaa bbbb\ncc", settings("line", 9, 0, min_chunk_chars=5))
    assert all(len(c) >= 5 for c in chunks)


def test_semantic_splits_at_topic_shift_and_traces_fallback(monkeypatch):
    import private_agent.rag.chunking as chunking

    text = " ".join(["Cats purr softly."] * 4 + ["Stocks fell sharply."] * 4)
    events = []
    monkeypatch.setattr(
        chunking,
        "log_event",
        lambda _logger, event, **fields: events.append((event, fields)),
    )

    def embed(texts):
        return [[1.0, 0.0] if i < 4 else [0.0, 1.0] for i in range(len(texts))]

    s = settings("semantic", 500, 0, semantic_buffer_sentences=0, semantic_breakpoint_percentile=80)
    chunks = split_text(text, s, embed=embed)
    assert len(chunks) == 2 and "Cats" in chunks[0] and "Stocks" in chunks[1]

    def broken(texts):
        raise RuntimeError("down")

    assert split_text(text, s, embed=broken) == split_text(text, s, embed=None)
    assert events == [
        (
            "rag.semantic_chunking_failed",
            {
                "level": chunking.logging.WARNING,
                "error_type": "RuntimeError",
                "fallback": "sentence_chunking",
            },
        )
    ]


def test_invalid_settings_rejected():
    with pytest.raises(ValueError):
        ChunkingSettings(strategy="bogus")
    with pytest.raises(ValueError):
        ChunkingSettings(chunk_size=10, chunk_overlap=10)
