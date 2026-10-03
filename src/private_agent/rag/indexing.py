import os
import pathlib
import json
import hashlib
import tempfile
import shutil
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
    RAG_CHUNK_SIZE_CHARS,
    RAG_CHUNK_OVERLAP_CHARS,
)
from ..hardware import ollama_acceleration_options
from ..tools import ollama_langchain_client_kwargs, track_ollama_http_clients
from .retrieval import HybridRAGRetriever

console = Console()


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


def _split_into_chunks(
    content: str,
    chunk_size: int = RAG_CHUNK_SIZE_CHARS,
    overlap: int = RAG_CHUNK_OVERLAP_CHARS,
) -> List[str]:
    chunks = []
    start = 0
    while start < len(content):
        end = min(start + chunk_size, len(content))
        if end < len(content):
            boundary = content.rfind(" ", start + chunk_size // 2, end)
            if boundary > start:
                end = boundary
        chunk = content[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(content):
            break
        start = max(end - overlap, start + 1)
    return chunks


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
        backup_path = state_file_path.parent.with_name(
            f"{state_file_path.parent.name}.backup-"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        )
        if backup_path.exists():
            console.print(
                f"[red][RAG recovery error] Backup path already exists: "
                f"{backup_path}; no index files were moved.[/red]"
            )
            return None
        try:
            os.replace(state_file_path.parent, backup_path)
            state_file_path.parent.mkdir(parents=True, exist_ok=False)
        except OSError as recovery_error:
            console.print(
                f"[red][RAG recovery error] Could not preserve the damaged "
                f"index at '{backup_path}': {recovery_error}[/red]"
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
    supported_exts = {".pdf", ".txt", ".md", ".py", ".json", ".csv", ".rs", ".js", ".ts", ".html"}
    total_source_bytes = 0

    for file_path in sorted(path.glob("**/*.*")):
        ext = file_path.suffix.lower()
        if ext not in supported_exts or any(part.startswith(".") for part in file_path.relative_to(path).parts):
            continue
        
        try:
            if file_path.is_symlink():
                continue
            resolved_file_path = file_path.resolve()
            if not resolved_file_path.is_relative_to(path):
                console.print(
                    f"[yellow][Warning] Skipping file outside knowledge base: "
                    f"{file_path}[/yellow]"
                )
                continue
            stat = file_path.stat()
            file_key = str(resolved_file_path)
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
            source_fingerprint = {"mtime_ns": stat.st_mtime_ns, "size": stat.st_size}

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
            else:
                pages = [(file_path.read_text(encoding="utf-8", errors="ignore"), None)]

            file_docs = []
            chunk_number = 0
            for content, page_number in pages:
                for chunk_text in _split_into_chunks(content):
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
            failed_sources.add(str(file_path.resolve()))
            console.print(f"[yellow][Warning] Failed parsing {file_path.name}: {e}[/yellow]")

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
                console.print(f"[green][Incremental Indexing][/green] No document changes detected. Skipping re-embedding.")
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
        _write_index_state_atomic(state_file_path, updated_state)
        
        # Return Hybrid RAG retriever wrapper
        return HybridRAGRetriever(vectorstore, all_parsed_docs)
    except Exception as e:
        console.print(f"[red][Error] Vector DB initialization error: {e}[/red]")
        return None
