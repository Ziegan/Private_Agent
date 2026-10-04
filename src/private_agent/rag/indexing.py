import os
import pathlib
import json
import hashlib
import tempfile
import shutil
import sqlite3
import stat as stat_module
import uuid
from urllib.parse import quote
from datetime import datetime, timezone
from typing import Optional, List, Callable
from rich.console import Console
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

from ..config import (
    EMBEDDING_MODEL,
    HARDWARE_ACCELERATION_MODE,
    OLLAMA_BASE_URL,
    RAG_INDEX_PATH,
    RAG_MAX_CORPUS_BYTES,
    RAG_MAX_DOCUMENTS,
    RAG_MAX_FILE_BYTES,
    RAG_MAX_PDF_PAGES,
    CHUNKING_MIN_CHUNK_CHARS,
    CHUNKING_OVERLAP,
    CHUNKING_SEMANTIC_BUFFER,
    CHUNKING_SEMANTIC_MAX_SENTENCES,
    CHUNKING_SEMANTIC_PERCENTILE,
    CHUNKING_SEPARATORS,
    CHUNKING_SIZE,
    CHUNKING_STRATEGY,
    CHUNKING_UNIT,
    RAG_SQLITE_MAX_ROWS_PER_TABLE,
)
from ..hardware import ollama_acceleration_options
from ..tools import ollama_langchain_client_kwargs, track_ollama_http_clients
from .chunking import ChunkingSettings, split_text
from .retrieval import HybridRAGRetriever

console = Console()


SQLITE_EXTENSIONS = frozenset({".db", ".sqlite", ".sqlite3", ".db3"})
SUPPORTED_EXTENSIONS = frozenset(
    {".pdf", ".txt", ".md", ".py", ".json", ".csv", ".rs", ".js", ".ts", ".html"}
) | SQLITE_EXTENSIONS
_SQLITE_HEADER = b"SQLite format 3\x00"
_SQLITE_ROWS_PER_CHUNK = 10
_SQLITE_MAX_CELL_CHARS = 200


def _iter_source_files(root: pathlib.Path):
    """Yield (path, stat) for supported files under root.

    Symbolic links to files and directories are followed (targets may live
    outside root). Hard links and symlinks that reach the same file are indexed
    once, and directory cycles are not re-entered.
    """
    seen_directories = set()
    seen_files = set()
    for directory, subdirectories, filenames in os.walk(root, followlinks=True):
        real_directory = os.path.realpath(directory)
        if real_directory in seen_directories:
            subdirectories[:] = []
            continue
        seen_directories.add(real_directory)
        subdirectories[:] = sorted(
            name for name in subdirectories if not name.startswith(".")
        )
        for name in sorted(filenames):
            if name.startswith("."):
                continue
            if pathlib.Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            file_path = pathlib.Path(directory) / name
            try:
                file_stat = file_path.stat()
            except OSError:
                continue
            if not stat_module.S_ISREG(file_stat.st_mode):
                continue
            identity = (file_stat.st_dev, file_stat.st_ino)
            if identity in seen_files:
                continue
            seen_files.add(identity)
            yield file_path, file_stat


def _source_fingerprint(file_path: pathlib.Path, file_stat) -> dict:
    size = file_stat.st_size
    mtime_ns = file_stat.st_mtime_ns
    if file_path.suffix.lower() in SQLITE_EXTENSIONS:
        # Writes to a WAL-mode database may live only in the -wal file.
        try:
            wal_stat = file_path.with_name(file_path.name + "-wal").stat()
            size += wal_stat.st_size
            mtime_ns = max(mtime_ns, wal_stat.st_mtime_ns)
        except OSError:
            pass
    return {"mtime_ns": mtime_ns, "size": size}


def knowledge_base_signature(docs_path: Optional[str]) -> Optional[str]:
    """Cheap digest of the indexable files' paths, sizes and mtimes."""
    if not docs_path or not str(docs_path).strip():
        return None
    root = pathlib.Path(docs_path).expanduser().resolve()
    if not root.is_dir():
        return None
    digest = hashlib.sha256()
    for file_path, file_stat in _iter_source_files(root):
        fingerprint = _source_fingerprint(file_path, file_stat)
        digest.update(
            f"{file_path}\0{fingerprint['mtime_ns']}\0{fingerprint['size']}\n".encode(
                "utf-8", "surrogateescape"
            )
        )
    return digest.hexdigest()


def _sqlite_cell(value) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, (bytes, bytearray)):
        return f"<blob {len(value)} bytes>"
    text = str(value).replace("\n", " ")
    if len(text) > _SQLITE_MAX_CELL_CHARS:
        text = text[:_SQLITE_MAX_CELL_CHARS] + "..."
    return text


def _read_sqlite_pages(file_path: pathlib.Path) -> list:
    """Render each table's rows as text batches using a read-only connection."""
    with file_path.open("rb") as database_file:
        if database_file.read(len(_SQLITE_HEADER)) != _SQLITE_HEADER:
            raise ValueError("not a SQLite database file")
    connection = sqlite3.connect(
        f"file:{quote(str(file_path))}?mode=ro", uri=True, timeout=5
    )
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.text_factory = lambda raw: raw.decode("utf-8", "replace")
        tables = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        pages = []
        for table in tables:
            quoted = '"' + table.replace('"', '""') + '"'
            cursor = connection.execute(
                f"SELECT * FROM {quoted} LIMIT ?", (RAG_SQLITE_MAX_ROWS_PER_TABLE,)
            )
            columns = [column[0] for column in cursor.description]
            batch = []
            rows_seen = 0
            for row in cursor:
                rows_seen += 1
                batch.append(
                    " | ".join(
                        f"{column}={_sqlite_cell(value)}"
                        for column, value in zip(columns, row)
                    )
                )
                if len(batch) >= _SQLITE_ROWS_PER_CHUNK:
                    pages.append((f"SQLite table {table}\n" + "\n".join(batch), None))
                    batch = []
            if batch:
                pages.append((f"SQLite table {table}\n" + "\n".join(batch), None))
            if not rows_seen:
                pages.append(
                    (f"SQLite table {table} (columns: {', '.join(columns)}; no rows)", None)
                )
        return pages
    finally:
        connection.close()


def reset_knowledge_base(index_path: Optional[str] = None) -> str:
    """Remove only the configured Chroma index directory after path safety checks."""
    configured_path = pathlib.Path(index_path or RAG_INDEX_PATH).expanduser()
    if configured_path.is_symlink():
        raise ValueError("Refusing to reset a symbolic-link Chroma index path.")
    target = configured_path.resolve()
    protected_paths = {
        pathlib.Path("/").resolve(),
        pathlib.Path.home().resolve(),
        pathlib.Path.cwd().resolve(),
    }
    if target in protected_paths:
        raise ValueError(f"Refusing to reset protected directory '{target}'.")
    if target.exists() and not target.is_dir():
        raise ValueError(f"Chroma index path '{target}' is not a directory.")
    if target.exists():
        shutil.rmtree(target)
    return str(target)


def load_chunking_settings() -> ChunkingSettings:
    """Build chunking settings from config, falling back to defaults when invalid."""
    try:
        return ChunkingSettings(
            strategy=CHUNKING_STRATEGY,
            chunk_size=CHUNKING_SIZE,
            chunk_overlap=CHUNKING_OVERLAP,
            unit=CHUNKING_UNIT if CHUNKING_UNIT in ("chars", "words", "tokens") else "chars",
            min_chunk_chars=CHUNKING_MIN_CHUNK_CHARS,
            separators=CHUNKING_SEPARATORS,
            semantic_breakpoint_percentile=CHUNKING_SEMANTIC_PERCENTILE,
            semantic_buffer_sentences=CHUNKING_SEMANTIC_BUFFER,
            semantic_max_sentences=CHUNKING_SEMANTIC_MAX_SENTENCES,
        )
    except ValueError as exc:
        console.print(
            f"[yellow][Warning] Invalid chunking settings ({exc}); "
            "using defaults.[/yellow]"
        )
        return ChunkingSettings(chunk_size=1200, chunk_overlap=200)


def _split_into_chunks(
    content: str,
    settings: Optional[ChunkingSettings] = None,
    extension: str = "",
    embed=None,
) -> List[str]:
    return split_text(content, settings or load_chunking_settings(), extension, embed)


def _read_index_state(state_path: pathlib.Path) -> dict:
    if not state_path.exists():
        return {}
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"RAG index state at '{state_path}' is unreadable; preserve the index "
            "and repair or explicitly rebuild it before indexing."
        ) from exc
    if not isinstance(state, dict) or any(
        not isinstance(source, str)
        or not source
        or not pathlib.Path(source).is_absolute()
        or not isinstance(fingerprint, dict)
        or type(fingerprint.get("mtime_ns")) is not int
        or type(fingerprint.get("size")) is not int
        or fingerprint["mtime_ns"] < 0
        or fingerprint["size"] < 0
        for source, fingerprint in state.items()
    ):
        raise ValueError(
            f"RAG index state at '{state_path}' has an unsupported format; "
            "preserve the index and repair or explicitly rebuild it before indexing."
        )
    return state


def _write_index_state_atomic(state_path: pathlib.Path, state: dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=state_path.parent,
            prefix=f".{state_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as state_file:
            temporary_path = pathlib.Path(state_file.name)
            json.dump(state, state_file, indent=4)
            state_file.flush()
            os.fsync(state_file.fileno())
        os.replace(temporary_path, state_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _preserve_corrupt_index(index_path: pathlib.Path) -> pathlib.Path:
    backup_path = index_path.with_name(
        f"{index_path.name}.backup-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex}"
    )
    os.replace(index_path, backup_path)
    index_path.mkdir(parents=True, exist_ok=False)
    return backup_path


def initialize_knowledge_base(
    docs_path: Optional[str],
    *,
    index_path: Optional[str] = None,
    confirm_rebuild: Optional[Callable[[str], bool]] = None,
):
    """Initializes the Chroma vector store exclusively when an explicit, valid knowledge base folder path is provided."""
    if not docs_path or not str(docs_path).strip():
        return None

    path = pathlib.Path(docs_path).expanduser().resolve()
    if not path.is_dir():
        console.print(f"[yellow][Warning] Provided KB path '{path}' is not a valid directory. Skipping knowledge base initialization.[/yellow]")
        return None

    chroma_persist_dir = os.path.abspath(index_path or RAG_INDEX_PATH)
    state_file_path = pathlib.Path(chroma_persist_dir) / ".index_state.json"
    os.makedirs(chroma_persist_dir, exist_ok=True)
    try:
        index_state = _read_index_state(state_file_path)
    except ValueError as exc:
        if confirm_rebuild is None or not confirm_rebuild(str(exc)):
            console.print(f"[red][RAG state error] {exc}[/red]")
            return None
        try:
            backup_path = _preserve_corrupt_index(state_file_path.parent)
        except OSError as recovery_error:
            console.print(
                f"[red][RAG recovery error] Could not preserve the damaged "
                f"index: {recovery_error}[/red]"
            )
            return None
        console.print(
            f"[yellow][RAG recovery][/yellow] Preserved the previous index at "
            f"'{backup_path}' and will build a new index."
        )
        index_state = {}

    console.print(f"[cyan][Info] Initializing local embedding model '{EMBEDDING_MODEL}' via Ollama...[/cyan]")
    try:
        embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=OLLAMA_BASE_URL,
            **ollama_langchain_client_kwargs(),
            **ollama_acceleration_options(HARDWARE_ACCELERATION_MODE),
        )
        track_ollama_http_clients(embeddings)
    except Exception as e:
        console.print(f"[red][Error] Could not bind embedding model: {e}[/red]")
        return None

    console.print(f"[bold green][Chroma DB][/bold green] Initializing local project vector store at: [cyan]{chroma_persist_dir}[/cyan]")

    db_exists = os.path.exists(chroma_persist_dir) and any(f for f in os.listdir(chroma_persist_dir) if f != ".index_state.json")

    changed_docs = []
    all_parsed_docs = []
    updated_state = {}
    failed_sources = set()
    total_source_bytes = 0

    chunking = load_chunking_settings()

    for file_path, stat in _iter_source_files(path):
        ext = file_path.suffix.lower()
        try:
            file_key = str(file_path)
            if stat.st_size > RAG_MAX_FILE_BYTES:
                console.print(
                    f"[yellow][Warning] Skipping {file_path.name}: file size "
                    f"{stat.st_size} exceeds the {RAG_MAX_FILE_BYTES}-byte limit.[/yellow]"
                )
                failed_sources.add(file_key)
                continue
            if total_source_bytes + stat.st_size > RAG_MAX_CORPUS_BYTES:
                console.print(
                    f"[yellow][Warning] Skipping {file_path.name}: corpus byte "
                    f"limit ({RAG_MAX_CORPUS_BYTES}) reached.[/yellow]"
                )
                failed_sources.add(file_key)
                continue
            total_source_bytes += stat.st_size
            source_fingerprint = {
                **_source_fingerprint(file_path, stat),
                "chunking": chunking.signature(),
            }

            if ext == ".pdf":
                try:
                    from pypdf import PdfReader
                    reader = PdfReader(str(file_path))
                    if len(reader.pages) > RAG_MAX_PDF_PAGES:
                        console.print(
                            f"[yellow][Warning] Skipping {file_path.name}: PDF has "
                            f"{len(reader.pages)} pages; limit is {RAG_MAX_PDF_PAGES}.[/yellow]"
                        )
                        failed_sources.add(file_key)
                        continue
                    pages = [
                        (page.extract_text() or "", page_number)
                        for page_number, page in enumerate(reader.pages, 1)
                    ]
                except ImportError:
                    console.print(f"[yellow][Warning] pypdf not installed, skipping PDF: {file_path.name}[/yellow]")
                    failed_sources.add(file_key)
                    continue
            elif ext in SQLITE_EXTENSIONS:
                pages = _read_sqlite_pages(file_path)
            else:
                pages = [(file_path.read_text(encoding="utf-8", errors="ignore"), None)]

            file_docs = []
            chunk_number = 0
            for content, page_number in pages:
                for chunk_text in _split_into_chunks(
                    content, chunking, ext, getattr(embeddings, "embed_documents", None)
                ):
                    metadata = {"source": file_key, "chunk": chunk_number}
                    if page_number is not None:
                        metadata["page"] = page_number
                    chunk_id = hashlib.sha256(
                        f"{file_key}\0{chunk_number}\0{chunk_text}".encode("utf-8")
                    ).hexdigest()
                    metadata["chunk_id"] = chunk_id
                    file_docs.append(Document(page_content=chunk_text, metadata=metadata))
                    chunk_number += 1

            if file_docs:
                if len(all_parsed_docs) + len(file_docs) > RAG_MAX_DOCUMENTS:
                    console.print(
                        f"[yellow][Warning] Skipping {file_path.name}: document "
                        f"chunk limit ({RAG_MAX_DOCUMENTS}) would be exceeded.[/yellow]"
                    )
                    failed_sources.add(file_key)
                    continue
                all_parsed_docs.extend(file_docs)
                updated_state[file_key] = source_fingerprint
                if file_key not in index_state or index_state[file_key] != source_fingerprint:
                    changed_docs.extend(file_docs)
        except Exception as e:
            failed_sources.add(str(file_path))
            console.print(f"[yellow][Warning] Failed parsing {file_path.name}: {e}[/yellow]")

    state_write_started = False
    try:
        if db_exists:
            vectorstore = Chroma(persist_directory=chroma_persist_dir, embedding_function=embeddings)
            changed_sources = {doc.metadata["source"] for doc in changed_docs}
            old_ids_by_source = {}
            for source in changed_sources:
                previous = vectorstore.get(where={"source": source})
                old_ids_by_source[source] = set(previous.get("ids", []))
            if changed_docs:
                console.print(f"[green][Incremental Indexing][/green] Embedding {len(changed_docs)} new or modified chunk(s)...")
                vectorstore.add_documents(
                    changed_docs,
                    ids=[doc.metadata["chunk_id"] for doc in changed_docs],
                )
                for source in changed_sources:
                    current_ids = {
                        doc.metadata["chunk_id"]
                        for doc in changed_docs
                        if doc.metadata["source"] == source
                    }
                    obsolete_ids = old_ids_by_source[source] - current_ids
                    if obsolete_ids:
                        vectorstore.delete(ids=list(obsolete_ids))
            else:
                console.print("[green][Incremental Indexing][/green] No document changes detected. Skipping re-embedding.")
            removed_sources = set(index_state) - set(updated_state) - failed_sources
            for source in removed_sources:
                vectorstore.delete(where={"source": source})
        else:
            if all_parsed_docs:
                console.print(f"[green][Initial Indexing][/green] Creating vector store with {len(all_parsed_docs)} chunk(s)...")
                vectorstore = Chroma.from_documents(
                    all_parsed_docs,
                    embeddings,
                    persist_directory=chroma_persist_dir,
                    ids=[doc.metadata["chunk_id"] for doc in all_parsed_docs],
                )
            else:
                vectorstore = Chroma(persist_directory=chroma_persist_dir, embedding_function=embeddings)

        for source in failed_sources:
            if source in index_state:
                updated_state[source] = index_state[source]
        state_write_started = True
        _write_index_state_atomic(state_file_path, updated_state)
        
        # Return Hybrid RAG retriever wrapper
        return HybridRAGRetriever(vectorstore, all_parsed_docs)
    except Exception as e:
        if db_exists and not state_write_started and confirm_rebuild is not None:
            if confirm_rebuild(
                f"Vector database initialization/update failed: {e}. "
                "The damaged index will be preserved before rebuilding."
            ):
                try:
                    backup_path = _preserve_corrupt_index(
                        pathlib.Path(chroma_persist_dir)
                    )
                    console.print(
                        f"[yellow][RAG recovery][/yellow] Preserved the previous "
                        f"vector index at '{backup_path}'. Rebuilding from current "
                        "knowledge-base files."
                    )
                    if all_parsed_docs:
                        vectorstore = Chroma.from_documents(
                            all_parsed_docs,
                            embeddings,
                            persist_directory=chroma_persist_dir,
                            ids=[
                                doc.metadata["chunk_id"] for doc in all_parsed_docs
                            ],
                        )
                    else:
                        vectorstore = Chroma(
                            persist_directory=chroma_persist_dir,
                            embedding_function=embeddings,
                        )
                    _write_index_state_atomic(state_file_path, updated_state)
                    return HybridRAGRetriever(vectorstore, all_parsed_docs)
                except Exception as recovery_error:
                    console.print(
                        f"[red][RAG recovery error] Rebuild failed after preserving "
                        f"the previous index: {recovery_error}[/red]"
                    )
                    return None
        console.print(f"[red][Error] Vector DB initialization error: {e}[/red]")
        return None
