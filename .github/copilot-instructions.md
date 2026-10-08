# Copilot Instructions

## Project shape

Private Agent is a local-first Python CLI built around Ollama, SQLite, and a
workspace-scoped tool system. The entry points `main.py`, `python -m
private_agent`, and the `private-agent` console script all lead to
`private_agent.cli:main`.

`src/private_agent/agent/runtime.py` orchestrates startup, provider/model
selection, conversation turns, planning and coding workflows, tool execution,
and shutdown. Supporting agent modules handle interactive input, permissions,
provider configuration, prompt construction, context compaction, and session
statistics. `agent/__init__.py` is a compatibility facade over the runtime.

The runtime combines several stateful subsystems:

- `config.py` loads and migrates per-user JSON configuration; its settings are
  grouped by concern and exposed as runtime constants.
- `database/memory.py` owns SQLite conversation history, episodic and learned
  memory, and persistent task plans and progress.
- `rag/` chunks local files, indexes them in Chroma, and combines vector
  similarity with BM25 retrieval. The runtime adds retrieved documents and
  citations to model context.
- `tools/` contains the built-in LangChain tools and their network, media,
  memory, planning, filesystem, and shell helpers. The runtime registers
  configured MCP tools alongside these tools.
- `sandbox/` enforces workspace path boundaries and supports isolated or
  governed coding tasks; Linux command execution depends on the fail-closed
  sandbox prerequisites documented in the README.

Keep permission decisions and side effects distinct: session tool permissions,
coding workspace access, external-network consent, and per-use local media
approval are separate policies. Preserve bounded execution and explicit
failure reporting when extending tools or runtime workflows.

## Roadmap-driven development

`IMPLEMENTATION_ROADMAP.md` is the target-state vision and feature-priority
reference; `structured implementation.md` is the dependency-ordered execution
plan. Treat roadmap capabilities as future direction unless the current code
proves they are implemented. Use the execution plan's phases and exit gates to
sequence work, preserve compatibility, and avoid starting downstream
implementation before its required interfaces and migration paths are ready.
Do not attempt to implement the whole roadmap in one change unless explicitly
asked. Update the relevant plan when an explicitly requested scope, dependency,
migration strategy, or completion gate changes.

Apply the roadmap's privacy-first and enterprise goals without sacrificing
correctness or resource efficiency. Prefer bounded memory and context,
incremental or cached work where data changes are detectable, controlled
concurrency, and deterministic cleanup of clients, tasks, and temporary state.
For performance-sensitive changes, use representative benchmarks or existing
measurements to establish a baseline and verify the effect; avoid claiming
improvements without evidence. Preserve permission boundaries, failure
visibility, and regression coverage while optimizing.

## Performance coding practices

Treat `optimization_guide.md` as a source of hypotheses, not as measured
results for this application. Its sample timings and blanket recommendations
do not establish which change is faster or safe for this codebase.

- Find the measured bottleneck and record a repeatable baseline with
  representative input sizes. Compare the same workload before and after;
  include latency and peak-memory evidence where relevant, and verify output
  quality as well as speed. For retrieval changes, use the RAG benchmark
  documented in the README: it measures index-build time, query latency,
  recall@k, and mean reciprocal rank against labeled queries.
- Prefer algorithmic improvements when the access pattern supports them. For
  example, a set can improve repeated membership checks, but account for its
  construction and memory cost and preserve ordering or duplicate semantics
  where callers rely on them.
- Remove copies or materialization only when object ownership and lifetime are
  clear. Do not mutate caller-owned or shared state merely to save an
  allocation. Use lazy/bounded iteration only when the consumer does not need
  the complete result.
- Add caches only with an explicit bound, lifetime, and invalidation strategy.
  Keep context, task, request, and concurrency limits intact; avoid trading
  predictable resource use for unbounded retained memory.
- Apply `__slots__`, preallocation, local-function rewrites, `bisect`, or
  operator substitutions only when profiling the actual workload demonstrates
  a benefit and compatibility/semantic constraints are covered by tests.
  Consider the full algorithmic cost (for example, insertion into a Python
  list remains linear even when locating an insertion point is logarithmic).
- Review `ruff check src tests --select PERF` findings individually. A
  `try`/`except` or snapshot copy inside a loop may be intentional for
  per-item error isolation or safe mutation during iteration; do not remove
  those guarantees to silence a lint finding. Keep expected failures
  observable through the existing logging and user-facing error paths.

## Build, test, and lint

Install the package and development tools with:

```sh
python -m pip install -e ".[dev]"
```

Run the CLI with `python main.py`, `python -m private_agent`, or (after
installation) `private-agent`.

The tests use pytest and mock model/services, so they do not require Ollama,
network access, or a GPU:

```sh
pytest -q
pytest tests/unit -q
pytest tests/integration -q
pytest tests/unit/test_database.py -q
pytest tests/unit/test_database.py::test_persistent_memory -q
pytest -k "permission and full" -q
```

Run the documented lint check with:

```sh
ruff check src tests --select F
```

Runtime dependencies are declared in `pyproject.toml`; `requirements.txt` is
generated from that list. After changing runtime dependencies, run
`python scripts/sync_requirements.py`; use `python scripts/sync_requirements.py
--check` to verify they match. RAG retrieval can be benchmarked with
`python scripts/benchmark_rag.py` (see the README for required arguments and
Ollama setup).

## Repository conventions

- Add runtime dependencies to `[project].dependencies` in `pyproject.toml`,
  then regenerate `requirements.txt` with the sync script. Development and
  optional media dependencies belong in their existing optional-dependency
  groups.
- Add configuration defaults to the appropriate section of `DEFAULT_CONFIG`
  in `config.py`; preserve the existing categorized configuration and legacy
  key migration behavior.
- Tool implementations belong with their domain helpers under `tools/`.
  Keep tool authorization, network consent, workspace validation, and
  per-operation approval behavior aligned with the existing policies and
  schemas.
- Keep persistent state in the existing SQLite/memory and RAG layers rather
  than adding parallel stores. RAG indexing/retrieval should retain source
  metadata and citations.
- Tests live under `tests/unit/` and `tests/integration/`. They commonly use
  pytest fixtures, `monkeypatch`, temporary paths/databases, and mocked
  external clients; async cases use `pytest.mark.asyncio`. Reuse the
  `temp_db` and `temp_workspace` fixtures from `tests/conftest.py` where
  applicable.
