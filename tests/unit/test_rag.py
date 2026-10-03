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
    chunks = _split_into_chunks("x" * 2500, chunk_size=1200, overlap=200)
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
