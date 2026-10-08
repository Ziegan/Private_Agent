import os
import pytest
import pathlib
import json
import sys
import types
from unittest.mock import patch, MagicMock
from rich.console import Console
from langchain_core.messages import HumanMessage, AIMessage

from private_agent.database import PersistentMemory
from private_agent.agent import (
    trim_history_to_context_budget,
    _build_bounded_user_input,
    _token_count,
    format_retrieved_citations,
)
try:
    from private_agent.agent import get_robust_chat_model
except ImportError:
    # Fallback definition if not explicitly exposed in private_agent.agent
    def get_robust_chat_model(primary_model_name, fallback_model_name, tools=None):
        from langchain_ollama import ChatOllama
        try:
            model = ChatOllama(model=primary_model_name)
            if tools:
                model = model.bind_tools(tools)
            return model
        except Exception:
            model = ChatOllama(model=fallback_model_name)
            if tools:
                model = model.bind_tools(tools)
            return model

from private_agent.rag import (
    initialize_knowledge_base,
    HybridRAGRetriever,
)
from private_agent.rag.indexing import (
    _split_into_chunks,
    ChunkingSettings,
    _read_index_state,
    _write_index_state_atomic,
)

console = Console()

def test_context_budget_preserves_newest_messages():
    history = [
        HumanMessage(content="older " * 200),
        AIMessage(content="middle " * 80),
        HumanMessage(content="newest"),
    ]
    bounded = trim_history_to_context_budget(history, "question", 30)
    assert bounded == [history[-1]]


def test_prompt_tokenizer_fallback_is_traceable_without_prompt_content(monkeypatch):
    import private_agent.agent.prompts as prompts

    events = []

    class BrokenTokenizer:
        def encode(self, _text):
            raise ValueError("failure contained private user input")

    monkeypatch.setattr(prompts, "_get_tokenizer", lambda: BrokenTokenizer())
    monkeypatch.setattr(
        prompts,
        "_TOKENIZER_FALLBACKS_REPORTED",
        set(),
    )
    monkeypatch.setattr(
        prompts,
        "log_event",
        lambda _logger, event, **fields: events.append((event, fields)),
    )

    assert prompts.token_count("private user input") == 4
    assert events == [
        (
            "prompt.tokenizer_fallback",
            {
                "level": prompts.logging.WARNING,
                "stage": "count",
                "error_type": "ValueError",
                "fallback": "character_estimate",
            },
        )
    ]
    assert "private user input" not in repr(events)


def test_context_budget_keeps_query_and_bounds_local_context():
    bounded = _build_bounded_user_input(
        "Local context",
        "past summary " * 10000,
        "source notes " * 10000,
        "answer this exact question",
        120,
    )
    assert "answer this exact question" in bounded
    assert _token_count(bounded) <= 120


def test_context_budget_includes_system_prompt_and_bounds_long_query():
    from langchain_core.messages import SystemMessage

    from private_agent.agent.prompts import calculate_prompt_budgets

    system_prompts = [SystemMessage(content="rules " * 100)]
    budget, output_budget = calculate_prompt_budgets(
        system_prompts, context_window=2000, max_output_tokens=400
    )
    assert budget + output_budget + _token_count(system_prompts[0].content) < 2000

    bounded = _build_bounded_user_input(
        "Context", "", "", "long question " * 1000, budget=40
    )
    assert _token_count(bounded) <= 40


def test_retrieved_source_citations_are_independent_of_model_response():
    documents = [
        types.SimpleNamespace(
            metadata={"source": "/docs/guide.pdf", "page": 5, "chunk": 2}
        ),
        types.SimpleNamespace(
            metadata={"source": "/docs/guide.pdf", "page": 5, "chunk": 2}
        ),
        types.SimpleNamespace(metadata={"source": "/docs/readme.md", "chunk": 0}),
    ]
    assert format_retrieved_citations(documents) == [
        "/docs/guide.pdf (page 5) (chunk 2)",
        "/docs/readme.md (chunk 0)",
    ]

def test_rag_chunker_splits_long_documents_with_overlap():
    chunks = _split_into_chunks(
        "x" * 2500, ChunkingSettings(strategy="fixed", chunk_size=1200, chunk_overlap=200)
    )
    assert len(chunks) == 3
    assert all(chunks)

def test_reset_knowledge_base_removes_only_configured_index(tmp_path):
    from private_agent.rag import reset_knowledge_base

    index = tmp_path / "chroma"
    index.mkdir()
    (index / "index.bin").write_bytes(b"data")
    sibling = tmp_path / "keep.txt"
    sibling.write_text("keep", encoding="utf-8")

    assert pathlib.Path(reset_knowledge_base(str(index))) == index
    assert not index.exists()
    assert sibling.read_text(encoding="utf-8") == "keep"

def test_reset_knowledge_base_refuses_symlinks_and_protected_paths(tmp_path):
    from private_agent.rag import reset_knowledge_base

    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "index-link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic-link"):
        reset_knowledge_base(str(link))
    with pytest.raises(ValueError, match="protected"):
        reset_knowledge_base(str(pathlib.Path.cwd()))
    assert target.is_dir()

def test_local_data_reset_requires_typed_confirmation(monkeypatch, tmp_path):
    import private_agent.agent.runtime as agent

    index = tmp_path / "chroma"
    index.mkdir()
    memory = PersistentMemory(db_path=str(tmp_path / "memory.sqlite"))
    memory.save_message("session", "human", "private")
    memory.save_summary("session", "private summary")
    monkeypatch.setattr(agent, "RAG_INDEX_PATH", str(index))
    monkeypatch.setattr(agent.sys, "stdin", MagicMock(isatty=lambda: True))
    monkeypatch.setattr(agent.console, "input", MagicMock(side_effect=["3", "no"]))

    agent._offer_local_data_reset(memory)

    assert index.is_dir()
    assert memory.load_history("session")[0].content == "private"
    assert memory.get_all_episodic_summaries(session_id="session") == [
        "private summary"
    ]
    memory.close()

def test_local_data_reset_clears_both_stores_after_confirmation(monkeypatch, tmp_path):
    import private_agent.agent.runtime as agent

    index = tmp_path / "chroma"
    index.mkdir()
    memory = PersistentMemory(db_path=str(tmp_path / "memory.sqlite"))
    memory.save_message("session", "human", "private")
    memory.save_summary("session", "private summary")
    monkeypatch.setattr(agent, "RAG_INDEX_PATH", str(index))
    monkeypatch.setattr(agent.sys, "stdin", MagicMock(isatty=lambda: True))
    monkeypatch.setattr(agent.console, "input", MagicMock(side_effect=["3", "RESET"]))

    agent._offer_local_data_reset(memory)

    assert not index.exists()
    assert memory.load_history("session") == []
    assert memory.get_all_episodic_summaries(session_id="session") == []
    memory.close()

@patch("private_agent.rag.indexing.OllamaEmbeddings")
@patch("private_agent.rag.indexing.Chroma")
def test_rag_initialization_mocked(mock_chroma_class, mock_embeddings_class, tmp_path):
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    (doc_dir / "notes.txt").write_text("Private agent vector search test content.", encoding="utf-8")

    mock_vectorstore = MagicMock()
    mock_chroma_class.from_documents.return_value = mock_vectorstore

    vs = initialize_knowledge_base(str(doc_dir))
    assert vs is not None

def test_rag_initialization_with_invalid_or_empty_paths():
    """Ensure initialize_knowledge_base returns None immediately for empty, None, or invalid paths."""
    assert initialize_knowledge_base(None) is None
    assert initialize_knowledge_base("") is None
    assert initialize_knowledge_base("  ") is None
    assert initialize_knowledge_base("/nonexistent/directory/path/12345") is None

@patch("private_agent.rag.indexing.OllamaEmbeddings", side_effect=Exception("Connection refused"))
def test_rag_embedding_failure_fallback(mock_embeddings_class):
    vs = initialize_knowledge_base("/dummy/path")
    assert vs is None

def test_rag_corrupt_index_state_fails_closed_before_embedding(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    index = tmp_path / "index"
    index.mkdir()
    state_path = index / ".index_state.json"
    state_path.write_text("{broken", encoding="utf-8")
    with patch("private_agent.rag.indexing.OllamaEmbeddings") as embeddings:
        assert initialize_knowledge_base(str(docs), index_path=str(index)) is None
    embeddings.assert_not_called()
    assert state_path.read_text(encoding="utf-8") == "{broken"

def test_rag_guided_recovery_preserves_corrupt_index(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "notes.txt").write_text("rebuild this index", encoding="utf-8")
    index = tmp_path / "index"
    index.mkdir()
    (index / ".index_state.json").write_text("{bad", encoding="utf-8")
    (index / "chroma.sqlite3").write_text("preserve", encoding="utf-8")
    monkeypatch.setattr("private_agent.rag.indexing.OllamaEmbeddings", lambda **kwargs: object())
    vectorstore = MagicMock()
    monkeypatch.setattr("private_agent.rag.indexing.Chroma", MagicMock())
    monkeypatch.setattr(
        "private_agent.rag.indexing.Chroma.from_documents", MagicMock(return_value=vectorstore)
    )

    retriever = initialize_knowledge_base(
        str(docs),
        index_path=str(index),
        confirm_rebuild=lambda detail: "unreadable" in detail,
    )
    assert retriever is not None
    backups = list(tmp_path.glob("index.backup-*"))
    assert len(backups) == 1
    assert (backups[0] / "chroma.sqlite3").read_text(encoding="utf-8") == "preserve"
    assert _read_index_state(index / ".index_state.json")

def test_rag_limits_file_size_and_keeps_state_write_atomic(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "large.txt").write_text("this is larger than allowed", encoding="utf-8")
    index = tmp_path / "index"
    monkeypatch.setattr("private_agent.rag.indexing.RAG_MAX_FILE_BYTES", 4)

    class EmptyVectorStore:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr("private_agent.rag.indexing.OllamaEmbeddings", lambda **kwargs: object())
    monkeypatch.setattr("private_agent.rag.indexing.Chroma", EmptyVectorStore)
    retriever = initialize_knowledge_base(str(docs), index_path=str(index))
    assert retriever is not None
    assert _read_index_state(index / ".index_state.json") == {}

    state_path = index / ".index_state.json"
    original_state = '{"unchanged": {"mtime_ns": 1, "size": 1}}'
    state_path.write_text(original_state, encoding="utf-8")
    with patch("private_agent.rag.indexing.os.replace", side_effect=OSError("disk error")):
        with pytest.raises(OSError, match="disk error"):
            _write_index_state_atomic(state_path, {"new": {"mtime_ns": 2, "size": 2}})
    assert state_path.read_text(encoding="utf-8") == original_state
    assert list(index.glob("*.tmp")) == []
    assert list(index.glob(".*.tmp")) == []

def test_rag_vector_failure_does_not_advance_index_state(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    source = docs / "changed.txt"
    source.write_text("updated document content", encoding="utf-8")
    index = tmp_path / "index"
    index.mkdir()
    (index / "chroma.sqlite3").touch()
    state_path = index / ".index_state.json"
    old_state = {
        str(source.resolve()): {"mtime_ns": 1, "size": 1}
    }
    state_path.write_text(json.dumps(old_state), encoding="utf-8")

    class FailingVectorStore:
        def __init__(self, **kwargs):
            pass

        def get(self, where):
            return {"ids": []}

        def add_documents(self, documents, ids):
            raise RuntimeError("embedding write failed")

    monkeypatch.setattr("private_agent.rag.indexing.OllamaEmbeddings", lambda **kwargs: object())
    monkeypatch.setattr("private_agent.rag.indexing.Chroma", FailingVectorStore)
    assert initialize_knowledge_base(str(docs), index_path=str(index)) is None
    assert _read_index_state(state_path) == old_state


def test_rag_modified_and_removed_sources_update_vector_store_and_state(
    tmp_path, monkeypatch
):
    from private_agent.rag import indexing

    docs = tmp_path / "docs"
    docs.mkdir()
    modified_source = docs / "modified.txt"
    modified_source.write_text("new shorter text", encoding="utf-8")
    removed_source = str((docs / "removed.txt").resolve())
    index = tmp_path / "index"
    index.mkdir()
    (index / "chroma.sqlite3").touch()
    old_state = {
        str(modified_source.resolve()): {"mtime_ns": 1, "size": 100},
        removed_source: {"mtime_ns": 1, "size": 20},
    }
    state_path = index / ".index_state.json"
    state_path.write_text(json.dumps(old_state), encoding="utf-8")

    class ExistingVectorStore:
        def __init__(self, **kwargs):
            self.deleted = []
            self.added = []

        def get(self, where):
            assert where == {"source": str(modified_source.resolve())}
            return {"ids": ["obsolete-chunk"]}

        def add_documents(self, documents, ids):
            self.added.extend(ids)

        def delete(self, *, ids=None, where=None):
            self.deleted.append({"ids": ids, "where": where})

    store = ExistingVectorStore()
    monkeypatch.setattr(indexing, "OllamaEmbeddings", lambda **kwargs: object())
    monkeypatch.setattr(indexing, "Chroma", lambda **kwargs: store)

    retriever = initialize_knowledge_base(str(docs), index_path=str(index))

    assert retriever is not None
    assert len(store.added) == 1
    assert store.deleted == [
        {"ids": ["obsolete-chunk"], "where": None},
        {"ids": None, "where": {"source": removed_source}},
    ]
    assert set(_read_index_state(state_path)) == {str(modified_source.resolve())}


def test_rag_corrupt_vector_database_fails_closed_without_advancing_state(
    tmp_path, monkeypatch
):
    from private_agent.rag import indexing

    docs = tmp_path / "docs"
    docs.mkdir()
    source = docs / "notes.txt"
    source.write_text("source remains indexed", encoding="utf-8")
    index = tmp_path / "index"
    index.mkdir()
    (index / "chroma.sqlite3").write_text("corrupt", encoding="utf-8")
    state_path = index / ".index_state.json"
    old_state = {str(source.resolve()): {"mtime_ns": 1, "size": 1}}
    state_path.write_text(json.dumps(old_state), encoding="utf-8")

    monkeypatch.setattr(indexing, "OllamaEmbeddings", lambda **kwargs: object())

    def corrupt_vector_store(**kwargs):
        raise RuntimeError("database is corrupt")

    monkeypatch.setattr(indexing, "Chroma", corrupt_vector_store)

    assert initialize_knowledge_base(str(docs), index_path=str(index)) is None
    assert _read_index_state(state_path) == old_state
    assert (index / "chroma.sqlite3").read_text(encoding="utf-8") == "corrupt"


def test_rag_corrupt_vector_database_can_be_explicitly_preserved_and_rebuilt(
    tmp_path, monkeypatch
):
    from private_agent.rag import indexing

    docs = tmp_path / "docs"
    docs.mkdir()
    source = docs / "notes.txt"
    source.write_text("rebuild from this source", encoding="utf-8")
    index = tmp_path / "index"
    index.mkdir()
    corrupt_file = index / "chroma.sqlite3"
    corrupt_file.write_text("corrupt bytes", encoding="utf-8")
    old_state = {str(source.resolve()): {"mtime_ns": 1, "size": 1}}
    (index / ".index_state.json").write_text(json.dumps(old_state), encoding="utf-8")
    vectorstore = MagicMock()
    chroma_calls = []

    class RecoveringChroma:
        def __new__(cls, **kwargs):
            chroma_calls.append("open")
            raise RuntimeError("vector database corruption")

        @classmethod
        def from_documents(cls, documents, embeddings, **kwargs):
            chroma_calls.append("rebuild")
            assert len(documents) == 1
            return vectorstore

    monkeypatch.setattr(indexing, "OllamaEmbeddings", lambda **kwargs: object())
    monkeypatch.setattr(indexing, "Chroma", RecoveringChroma)

    retriever = initialize_knowledge_base(
        str(docs),
        index_path=str(index),
        confirm_rebuild=lambda detail: "vector database corruption" in detail,
    )

    backups = list(tmp_path.glob("index.backup-*"))
    assert retriever is not None
    assert chroma_calls == ["open", "rebuild"]
    assert len(backups) == 1
    assert (backups[0] / "chroma.sqlite3").read_text(encoding="utf-8") == "corrupt bytes"
    recorded = _read_index_state(index / ".index_state.json")
    for fingerprint in recorded.values():
        assert fingerprint.pop("chunking")
    assert recorded == {
        str(source.resolve()): {
            "mtime_ns": source.stat().st_mtime_ns,
            "size": source.stat().st_size,
        }
    }


def test_rag_partial_vector_update_requires_confirmation_then_rebuilds(
    tmp_path, monkeypatch
):
    from private_agent.rag import indexing

    docs = tmp_path / "docs"
    docs.mkdir()
    source = docs / "changed.txt"
    source.write_text("new source content", encoding="utf-8")
    index = tmp_path / "index"
    index.mkdir()
    (index / "chroma.sqlite3").write_text("previous state", encoding="utf-8")
    old_state = {str(source.resolve()): {"mtime_ns": 1, "size": 1}}
    state_path = index / ".index_state.json"
    state_path.write_text(json.dumps(old_state), encoding="utf-8")

    class PartiallyFailingStore:
        def __init__(self, **kwargs):
            self.index_path = pathlib.Path(kwargs["persist_directory"])

        def get(self, where):
            return {"ids": ["old-chunk"]}

        def add_documents(self, documents, ids):
            (self.index_path / "partial-update").write_text(
                ",".join(ids), encoding="utf-8"
            )

        def delete(self, **kwargs):
            raise RuntimeError("delete failed after vector upsert")

    rebuilt = MagicMock()

    class RecoveringChroma:
        def __new__(cls, **kwargs):
            return PartiallyFailingStore(**kwargs)

        @classmethod
        def from_documents(cls, documents, embeddings, **kwargs):
            assert len(documents) == 1
            return rebuilt

    monkeypatch.setattr(indexing, "OllamaEmbeddings", lambda **kwargs: object())
    monkeypatch.setattr(indexing, "Chroma", RecoveringChroma)

    refused = initialize_knowledge_base(
        str(docs), index_path=str(index), confirm_rebuild=lambda _detail: False
    )
    assert refused is None
    assert _read_index_state(state_path) == old_state
    assert (index / "partial-update").exists()

    confirmed = initialize_knowledge_base(
        str(docs), index_path=str(index), confirm_rebuild=lambda _detail: True
    )
    backups = list(tmp_path.glob("index.backup-*"))
    assert confirmed is not None
    assert len(backups) == 1
    assert (backups[0] / "chroma.sqlite3").read_text(encoding="utf-8") == "previous state"
    assert (backups[0] / "partial-update").exists()
    assert not (index / "partial-update").exists()
    assert set(_read_index_state(state_path)) == {str(source.resolve())}


def test_rag_enforces_corpus_byte_limit(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    first = docs / "a.txt"
    second = docs / "b.txt"
    first.write_text("abc", encoding="utf-8")
    second.write_text("def", encoding="utf-8")
    index = tmp_path / "index"
    monkeypatch.setattr("private_agent.rag.indexing.RAG_MAX_FILE_BYTES", 10)
    monkeypatch.setattr("private_agent.rag.indexing.RAG_MAX_CORPUS_BYTES", 4)

    class RecordingVectorStore:
        @classmethod
        def from_documents(cls, documents, embeddings, **kwargs):
            cls.sources = {doc.metadata["source"] for doc in documents}
            return cls()

    monkeypatch.setattr("private_agent.rag.indexing.OllamaEmbeddings", lambda **kwargs: object())
    monkeypatch.setattr("private_agent.rag.indexing.Chroma", RecordingVectorStore)
    assert initialize_knowledge_base(str(docs), index_path=str(index)) is not None
    assert RecordingVectorStore.sources == {str(first.resolve())}
    assert set(_read_index_state(index / ".index_state.json")) == {
        str(first.resolve())
    }

def test_rag_enforces_pdf_page_and_document_chunk_limits(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "too-many-pages.pdf").write_bytes(b"pdf")
    (docs / "too-many-chunks.md").write_text("x" * 2400, encoding="utf-8")
    index = tmp_path / "index"
    index.mkdir()
    (index / "chroma.sqlite3").touch()
    monkeypatch.setattr("private_agent.rag.indexing.RAG_MAX_PDF_PAGES", 1)
    monkeypatch.setattr("private_agent.rag.indexing.RAG_MAX_DOCUMENTS", 1)
    pdf_module = types.ModuleType("pypdf")
    pdf_module.PdfReader = lambda path: types.SimpleNamespace(
        pages=[object(), object()]
    )
    monkeypatch.setitem(sys.modules, "pypdf", pdf_module)

    class EmptyVectorStore:
        def __init__(self, **kwargs):
            pass

        def get(self, where):
            return {"ids": []}

    monkeypatch.setattr("private_agent.rag.indexing.OllamaEmbeddings", lambda **kwargs: object())
    monkeypatch.setattr("private_agent.rag.indexing.Chroma", EmptyVectorStore)
    assert initialize_knowledge_base(str(docs), index_path=str(index)) is not None
    assert _read_index_state(index / ".index_state.json") == {}

def test_hybrid_rag_selects_top_bm25_candidates_without_full_sort():
    class FakeBM25:
        def get_scores(self, query):
            return [0.4, 0.9, 0.9, 0.2, 0.7]

    documents = [
        types.SimpleNamespace(
            page_content=f"document {index}",
            metadata={"source": f"source-{index}", "chunk": index},
        )
        for index in range(5)
    ]
    retriever = HybridRAGRetriever.__new__(HybridRAGRetriever)
    retriever.documents = documents
    retriever.bm25 = FakeBM25()
    retriever.vectorstore = MagicMock()
    retriever.vectorstore.similarity_search.return_value = []

    results = retriever.similarity_search("query", k=2)

    assert results == [documents[1], documents[2]]


def _capture_indexing(monkeypatch):
    captured = []

    class RecordingStore:
        @classmethod
        def from_documents(cls, docs, embeddings, **kwargs):
            captured.extend(docs)
            return MagicMock()

    monkeypatch.setattr(
        "private_agent.rag.indexing.OllamaEmbeddings", lambda **kwargs: object()
    )
    monkeypatch.setattr("private_agent.rag.indexing.Chroma", RecordingStore)
    return captured


def test_rag_follows_symlinks_and_indexes_hardlinks_once(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "linked.txt").write_text("outside symlink content", encoding="utf-8")
    (outside / "tree").mkdir()
    (outside / "tree" / "deep.md").write_text("deep linked content", encoding="utf-8")
    (docs / "file_link.txt").symlink_to(outside / "linked.txt")
    (docs / "dir_link").symlink_to(outside / "tree", target_is_directory=True)
    (docs / "loop").symlink_to(docs, target_is_directory=True)
    (docs / "real.txt").write_text("real content", encoding="utf-8")
    os.link(docs / "real.txt", docs / "hard.txt")
    captured = _capture_indexing(monkeypatch)

    assert initialize_knowledge_base(
        str(docs), index_path=str(tmp_path / "index")
    ) is not None

    texts = sorted(doc.page_content for doc in captured)
    assert texts == [
        "deep linked content",
        "outside symlink content",
        "real content",
    ]


def test_rag_indexes_sqlite_databases_read_only(tmp_path, monkeypatch):
    import sqlite3

    docs = tmp_path / "docs"
    docs.mkdir()
    db_path = docs / "people.sqlite"
    connection = sqlite3.connect(db_path)
    connection.execute("CREATE TABLE people (name TEXT, note TEXT, data BLOB)")
    connection.execute("INSERT INTO people VALUES ('Ada', 'mathematician', x'00ff')")
    connection.commit()
    connection.close()
    (docs / "fake.db").write_text("not a database", encoding="utf-8")
    captured = _capture_indexing(monkeypatch)

    assert initialize_knowledge_base(
        str(docs), index_path=str(tmp_path / "index")
    ) is not None

    assert len(captured) == 1
    content = captured[0].page_content
    assert "SQLite table people" in content
    assert "name=Ada" in content and "note=mathematician" in content
    assert "<blob 2 bytes>" in content


def test_knowledge_base_signature_changes_with_new_or_updated_files(tmp_path):
    from private_agent.rag import knowledge_base_signature

    docs = tmp_path / "docs"
    docs.mkdir()
    first = docs / "a.txt"
    first.write_text("one", encoding="utf-8")
    baseline = knowledge_base_signature(str(docs))
    assert knowledge_base_signature(str(docs)) == baseline

    (docs / "b.md").write_text("two", encoding="utf-8")
    after_add = knowledge_base_signature(str(docs))
    assert after_add != baseline

    first.write_text("one changed", encoding="utf-8")
    assert knowledge_base_signature(str(docs)) != after_add
    assert knowledge_base_signature(str(tmp_path / "missing")) is None
