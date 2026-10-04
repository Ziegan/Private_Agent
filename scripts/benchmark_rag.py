"""Benchmark local RAG indexing and retrieval against labeled queries."""

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from rich.console import Console
from rich.table import Table

from private_agent.config import HARDWARE_ACCELERATION_MODE, OLLAMA_BASE_URL
from private_agent.hardware import format_hardware_status, inspect_ollama_hardware
from private_agent.rag import initialize_knowledge_base
from private_agent.tools import close_outbound_http_clients

MAX_QUERY_FILE_BYTES = 1_048_576
MAX_BENCHMARK_QUERIES = 1_000
MAX_QUERY_CHARS = 2_048
console = Console()


def load_labeled_queries(query_file: Path) -> list[dict[str, Any]]:
    """Read bounded query and relevant-source labels from a JSON file."""
    if query_file.stat().st_size > MAX_QUERY_FILE_BYTES:
        raise ValueError(
            f"Query file exceeds the {MAX_QUERY_FILE_BYTES}-byte benchmark limit."
        )
    try:
        entries = json.loads(query_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read benchmark queries: {exc}") from exc
    if (
        not isinstance(entries, list)
        or not entries
        or len(entries) > MAX_BENCHMARK_QUERIES
    ):
        raise ValueError(
            f"Queries must be a nonempty JSON list with at most "
            f"{MAX_BENCHMARK_QUERIES} entries."
        )

    queries = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Query entry {index + 1} must be an object.")
        query = entry.get("query")
        expected_sources = entry.get("expected_sources")
        if (
            not isinstance(query, str)
            or not query.strip()
            or len(query) > MAX_QUERY_CHARS
        ):
            raise ValueError(
                f"Query entry {index + 1} needs a nonempty query of at most "
                f"{MAX_QUERY_CHARS} characters."
            )
        if (
            not isinstance(expected_sources, list)
            or not expected_sources
            or len(expected_sources) > 100
            or any(not isinstance(source, str) or not source.strip() for source in expected_sources)
        ):
            raise ValueError(
                f"Query entry {index + 1} needs 1–100 expected_sources strings."
            )
        normalized_sources = set()
        for source in expected_sources:
            path = Path(source)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(
                    f"Expected source '{source}' must be relative to the documents folder."
                )
            normalized_sources.add(path.as_posix())
        queries.append(
            {"query": query.strip(), "expected_sources": normalized_sources}
        )
    return queries


def _document_source(document, documents_root: Path) -> str | None:
    source = document.metadata.get("source")
    if not isinstance(source, str):
        return None
    try:
        return Path(source).resolve().relative_to(documents_root).as_posix()
    except (OSError, RuntimeError, ValueError):
        return None


def _retrieval_metrics(retrieved, expected_sources: set[str], documents_root: Path):
    retrieved_sources = []
    first_relevant_rank = None
    for rank, document in enumerate(retrieved, start=1):
        source = _document_source(document, documents_root)
        retrieved_sources.append(source)
        if first_relevant_rank is None and source in expected_sources:
            first_relevant_rank = rank
    relevant_found = len(set(retrieved_sources) & expected_sources)
    recall = relevant_found / len(expected_sources) if expected_sources else 0.0
    reciprocal_rank = 1.0 / first_relevant_rank if first_relevant_rank else 0.0
    return recall, reciprocal_rank


def _benchmark_queries(retriever, queries, documents_root, top_k, repeats):
    latencies = []
    recalls = []
    reciprocal_ranks = []
    for entry in queries:
        query_recall = []
        query_rr = []
        for _ in range(repeats):
            started = time.perf_counter()
            retrieved = retriever.similarity_search(entry["query"], k=top_k)
            latencies.append(time.perf_counter() - started)
            recall, reciprocal_rank = _retrieval_metrics(
                retrieved,
                entry["expected_sources"],
                documents_root,
            )
            query_recall.append(recall)
            query_rr.append(reciprocal_rank)
        recalls.append(statistics.mean(query_recall))
        reciprocal_ranks.append(statistics.mean(query_rr))
    return {
        "queries": len(queries),
        "repeats": repeats,
        "top_k": top_k,
        "median_latency_ms": statistics.median(latencies) * 1_000,
        "mean_latency_ms": statistics.mean(latencies) * 1_000,
        "mean_recall_at_k": statistics.mean(recalls),
        "mean_reciprocal_rank": statistics.mean(reciprocal_ranks),
    }


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--documents",
        type=Path,
        required=True,
        help="Folder of local documents to index into a temporary benchmark index.",
    )
    parser.add_argument(
        "--queries",
        type=Path,
        required=True,
        help="JSON file containing query strings and relevant relative source paths.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    if not 1 <= args.top_k <= 50:
        parser.error("--top-k must be between 1 and 50.")
    if not 1 <= args.repeats <= 20:
        parser.error("--repeats must be between 1 and 20.")
    return args


def main(argv=None) -> int:
    args = _parse_args(argv)
    documents_root = args.documents.expanduser().resolve()
    if not documents_root.is_dir():
        raise ValueError(f"Documents folder '{documents_root}' is not a directory.")
    queries = load_labeled_queries(args.queries.expanduser())
    console.print(
        "[cyan]Building a temporary local RAG index; "
        "document contents are sent only to the configured Ollama endpoint.[/cyan]"
    )
    try:
        with TemporaryDirectory(prefix="private-agent-rag-benchmark-") as temporary:
            index_path = Path(temporary) / "index"
            started = time.perf_counter()
            retriever = initialize_knowledge_base(
                str(documents_root),
                index_path=str(index_path),
            )
            indexing_seconds = time.perf_counter() - started
            if retriever is None:
                raise RuntimeError(
                    "Could not initialize the temporary RAG index; check local "
                    "Ollama availability, embedding model, and document limits."
                )
            metrics = _benchmark_queries(
                retriever,
                queries,
                documents_root,
                args.top_k,
                args.repeats,
            )
            metrics["indexing_seconds"] = indexing_seconds
            hardware = inspect_ollama_hardware(
                OLLAMA_BASE_URL,
                HARDWARE_ACCELERATION_MODE,
            )
            console.print(
                f"[green]RAG benchmark[/green] "
                f"({metrics['queries']} queries × {metrics['repeats']} repeats, "
                f"top_k={metrics['top_k']})"
            )
            table = Table(box=None, show_header=False, padding=(0, 1))
            table.add_column("Metric", style="bold cyan")
            table.add_column("Value")
            table.add_row("Index build", f"{indexing_seconds:.3f} s")
            table.add_row("Median retrieval", f"{metrics['median_latency_ms']:.2f} ms")
            table.add_row("Mean retrieval", f"{metrics['mean_latency_ms']:.2f} ms")
            table.add_row("Mean recall@k", f"{metrics['mean_recall_at_k']:.3f}")
            table.add_row("Mean reciprocal rank", f"{metrics['mean_reciprocal_rank']:.3f}")
            console.print(table)
            console.print(
                "[bold cyan]Ollama hardware status[/bold cyan]\n"
                + format_hardware_status(hardware)
            )
            del retriever
    finally:
        asyncio.run(close_outbound_http_clients())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
