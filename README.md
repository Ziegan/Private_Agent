# Private Agent

## Summary

Private Agent is a local-first, privacy-focused command-line AI agent. It runs
chat models through local runners such as [Ollama](https://ollama.com/), LM
Studio, and other configured OpenAI-compatible runtimes, keeps
long-term memory and task plans in a local SQLite database, grounds answers in
your own documents with a local RAG index (ChromaDB + BM25), extends itself
with Markdown skills, and acts on your workspace through permission-gated
tools and optional MCP servers. Nothing leaves the machine unless you approve
an online provider or an online research request.

## Goal of the Project

- A **fully private** agent: local inference, local memory, local retrieval, no
  telemetry, offline-by-default Hugging Face/Transformers environment.
- An **iterative, agentic runtime** that can plan, act with tools, verify, fix,
  and re-run until a task is genuinely done, within bounded limits.
- **Explicit user control**: three session permission modes, five coding
  workspace access levels, approved-plan gating, typed confirmations for
  destructive actions, and consent before any network or device access.
- **Capability-aware** behaviour: vision, thinking, tool/function calling and
  audio are used automatically only when the chosen model declares support.
- **Honest degradation**: when something is unavailable (Ollama, internet,
  sandbox tools, a model capability) the agent reports it and continues safely
  instead of failing silently or claiming unverified results.

## Core Functionality and Operational Behaviour

**Session flow.** `private-agent` loads `~/.private_agent.conf`, discovers
reachable local inference runtimes on supported desktop platforms (Ollama,
LM Studio, and configured loopback OpenAI-compatible servers such as llama.cpp), locks the
workspace root, asks for a session permission mode, then asks for an available
provider and model. If no local runtime is found, online mode is the default.
When an online API key is configured, startup offers
"Online — configured in config" as the default provider and a separate manual
entry option. The selected online configuration is reused for the rest of that
run, including after `switch`. It detects the model's capabilities, loads
skills, indexes RAG documents, opens the SQLite memory, prints a compact status
(only non-zero counts) and starts the prompt loop. Completed turns are saved as
they finish.

**Agentic loop.** Each request runs a plan → act → verify loop. Every tool call
is checked against the permission mode, workspace boundary and output-size
limits. Local models have no configured round-count or tool-call-count cap.
When an online model is allowed to call tools, the CLI asks whether to apply
`max_tool_iterations` and `max_tool_calls` for that session; declining (the
default) leaves those counts uncapped. `max_task_seconds` remains in effect
for all providers. Failures are fed back to the model so it can analyse, fix
and retry within the active time budget.

**Permission model (kept separate per role).**

| Role | Policy source | Behaviour |
| --- | --- | --- |
| Interactive session / researcher | Startup choice: 1 Manual, 2 Auto, 3 Full | Manual asks before every tool action; Auto asks before edits, commands and untrusted MCP; Full approves all tool calls (including web-research consent) for the session |
| Planner | `/plan` | No code-execution tools; saves a reviewable checklist in SQLite; web search only with a per-request prompt |
| Coding agent | Per-task workspace access level | `full_hitl`, `full_governed`, `full_monitored` (edit in the real workspace) or `isolated_governed`, `isolated_hitl` (edit an isolated copy). Only coding requires a saved, approved plan |

**Planning and tasks.** Plans (goal, type, ordered steps, assumptions,
constraints, research questions) are stored in SQLite with revisions, digests
and approvals. `/tasks` lists, views, approves, resumes, pauses, cancels,
verifies steps, validates, completes, reconciles and deletes plans. Approving
a plan automatically starts it; a failed approval offers a revised strategy.
Interrupted tasks are recoverable and use leases with a heartbeat.

**Research.** Before any web request the agent checks for a live internet
connection. If offline, you can restore access and retry, continue with
incomplete offline data, or cancel. Online results are labelled and never
presented as verified without evidence.

**Memory.** SQLite stores chat history, episodic summaries, user-confirmed
learned items, session-linked skills and task plans. On `exit` or `Ctrl+C` the
session is summarized and saved with short stage messages
(`Summary 1/3` … `3/3`); pressing `Ctrl+C` again skips the summary and exits.

**Failure handling.** Startup errors exit with a non-zero status; model, tool,
network, database and sandbox errors are caught, reported concisely and the
session continues where safe. Empty model responses are retried once within
the current turn's time budget; if the retry also fails or is empty, the agent
reports that in the current session rather than restarting it.

## Features Implemented

- **Models and providers**: Ollama and configured local OpenAI-compatible
  runtimes (including LM Studio and llama.cpp), plus OpenAI-compatible online endpoints (HTTPS
  or loopback only). Online credentials may be entered for one session or
  explicitly loaded from the saved configuration; manually entered keys are
  never written to disk. Automatic capability detection (tools, function
  calls, structured output, thinking, vision, audio); toggleable thinking with
  configurable effort (`/think`, `/think-effort`, `/think-status`); streaming
  output; optional visible reasoning; Ollama hardware-placement report
  (`/hardware-status`).
- **Built-in tools**: read/edit/list/search files, text and symbol search, git
  inspection, patch preview/apply, rename/delete with snapshots, sandboxed
  shell (`run_shell_command`), SQLite history read/search/delete and schema
  lookup, task-plan create/inspect/update/revise, `create_skill`, local
  date/time, `web_search`, `fetch_webpage`, `download_web_file`,
  `fetch_current_location` (Local provider only) and `get_weather` (forecast up to 16 days
  ahead or history for a past date/week, optional hour). Both check for an active internet
  connection first and follow the network consent policy. Privacy: `fetch_current_location` sends
  this machine's public IP to ipwho.is (fallback ipapi.co) for city-level lookup, and `get_weather`
  sends coordinates to Open-Meteo. Online sessions must pass a place name to `get_weather`.
- **Local media**: workspace image and sampled video input, webcam capture,
  microphone recording and local Whisper / PCM-WAV transcription, each with
  per-use approval and model-capability checks.
- **Iterative coding tasks**: isolated workspace copies, checkpoints, git
  helpers and project test commands run in a Bubblewrap/`prlimit` sandbox
  (Linux), re-run after fixes until they pass, with test-output limits.
- **Local RAG**: ChromaDB vector store plus BM25 keyword retrieval, PDF/text
  chunking with overlap, SQLite databases (`.db`, `.sqlite`, `.sqlite3`,
  `.db3`; opened read-only, rows rendered per table), symlinked, soft- and
  hard-linked files and folders (hard links and duplicate targets are indexed
  once; link loops are skipped), automatic index updates when files are added,
  changed or removed under the RAG path (checked before each request, at most
  every `rag.auto_refresh_seconds`), size/page/document caps, citations, status, list,
  reindex and reset via `/maintenance`.
- **Skills**: Markdown skills with relevance matching, a per-skill tool-round
  limit that is enforced only when online tool limits are consented, `/skills`
  listing and explicit selection with auto-completion, and skill creation
  through a tool.
- **MCP**: stdio (sandboxed) and remote HTTP MCP servers configured in the
  config file, with an auto-approve allow-list for trusted tools.
- **Interactive CLI**: command and `@file` auto-completion, session-wide prompt
  history, `/help`, `/status`, `/context`, `/add`, `/list_tools`,
  `/maintenance`, `/plan`, `/tasks`, `/skills`, `switch`,
  `exit`, compact startup counts for memory and saved plans.
- **Maintenance and safety**: retention policies, stale-task handling,
  schema-aware writes, debug run logs, and fail-closed sandboxing.
- **Provider recovery**: an empty streaming response is retried once using
  non-streaming invocation within the existing task deadline; persistent
  empty responses or retry failures are reported in the current session.

## Installation and Standalone Run Steps

Requirements: Python 3.10–3.14 (validated on 3.14), at least one supported
local runtime or an online OpenAI-compatible endpoint. Local runtime discovery
uses loopback APIs and is available across desktop platforms. On Linux,
`bubblewrap` (`bwrap`) plus `util-linux` (`prlimit`) are required for shell,
code-task tests and stdio MCP (these features are disabled, not unsandboxed,
without them).

```sh
# 1. Get the code
git clone https://github.com/Ziegan/Private_Agent.git && cd Private_Agent
# Install the command in an isolated system-wide environment:
pipx install .
```

Alternatively, install the command in a virtual environment (activate that
environment whenever you use it):

```sh
python -m venv .venv && source .venv/bin/activate
python -m pip install .
# Optional: media input (webcam, video, microphone, Whisper)
python -m pip install ".[media]"
# Development and tests
python -m pip install -e ".[dev]" && pytest -q
```

```sh
# 2. Optional local inference: start a supported local runner
# Ollama example:
ollama pull qwen3:4b          # any tool-capable chat model you prefer
ollama pull nomic-embed-text  # needed only for RAG
# LM Studio and llama.cpp use their configured loopback OpenAI-compatible APIs.

# 3. Run the installed command from the project you want it to work on
cd /path/to/your/project
private-agent
# The source checkout can also be run with `python main.py` or
# `python -m private_agent`.
```

Pass `--verbose-startup` to show configured paths, saved-memory counts, local
runtime details, detailed hardware diagnostics, and successful MCP/skill
discovery messages. The concise Ready summary includes OS/architecture and
whether OS-isolated command execution is available. In normal startup, Ollama
hardware placement is shown in the Ready summary only when an Ollama model is
selected; verbose startup can show diagnostics for all discovered Ollama
servers. `/hardware-status` reports Ollama placement when Ollama is selected,
and explains when another local runner does not expose a standardized
hardware-placement API. Destructive
local data resets are available under
`/maintenance` → `data reset` and still require typing `RESET`.

When launched from a project directory, `private-agent` automatically uses any
`workspace`, `rag`, and `skills` directories found directly there, under
`resources/`, or under `private_agent/resources/`. For each folder name, lookup
priority is project root, `resources/`, then `private_agent/resources/`;
symbolic links and paths escaping the project root are ignored. The matching
local directories override the corresponding configured defaults for that run.
If no local `workspace` folder is found, the normal configured workspace prompt
is used; if no local `rag` folder is found, the configured RAG path/prompt is
used; and if no local `skills` folder is found, the configured skills path is
used.

The first run creates `~/.private_agent.conf` (JSON) in your home directory;
legacy flat keys are migrated. Memory is stored in `~/.local_ai_memory.db`.
Put documents for RAG in `~/.private_agent/rag/` and skills in
`~/.private_agent/resources/skills/`.

Session commands: `exit`, `switch`, `/help`, `/status`, `/context`,
`/context clear`, `/list_tools`, `/maintenance` (`data reset` resets local
stores after typed confirmation), `/plan`, `/tasks`, `/skills [key]`, `/think on|off`, `/think-effort
low|medium|high`, `/think-status`, `/hardware-status`, `/summarize` (summarize and store the session now; `exit` will not re-summarize unless new prompts followed), `/compact` (replace the context window with a short working summary, without storing it), `/add path/to/folder` (include supported UTF-8 text files from an in-workspace folder in the next request only), and `@path/to/file`
to attach a workspace text file as request-only context. Folder and file context
are limited to the configured `tools.max_read_file_bytes` total; folder context
also skips hidden paths and is limited to 100 supported text files. Attached
contents are treated as untrusted data and are not saved in chat history.
The CLI requires an interactive terminal; non-TTY startup prints a message and
exits before opening a session. In the interactive prompt, Up/Down navigate
history and Ctrl-R searches it; prompt history is retained in memory for the
current application session only.

On exit the agent saves the session summary (timed) and prints a **Session Statistics** table: SQL session ID, session/agent/permission mode, provider, model, thinking, total session time, input/output/total tokens (provider-reported when available, otherwise estimated), tool uses per tool, summary tokens and generation time.

Backups: stop the agent, then copy the database with SQLite's backup API
(`sqlite3.Connection.backup`) and copy the whole `paths.rag_index` directory.
Approved edits keep snapshots in `.agent_snapshots` inside the workspace.

## Testing, Requirements Sync and Benchmark

### Run the tests

Tests use `pytest` (with `pytest-asyncio`) and need no running Ollama, network
or GPU; models and services are mocked. Unit tests are in `tests/unit/`
(config, database, interactive, providers, RAG, skills, tools, media,
network, hardware, code tasks) and runtime tests are in `tests/integration/`.

```sh
python -m pip install -e ".[dev]"                 # pytest, pytest-asyncio, ruff
pytest -q                                         # whole suite (~17 s)
pytest tests/unit -q                              # unit tests only
pytest tests/integration -q                       # integration tests only
pytest tests/unit/test_database.py -q             # one file
pytest -k "permission and full" -q                # filter by name
ruff check src tests --select F                   # lint for real errors
```

With `uv`, use `uv run pytest -q`.

### Sync requirements (`scripts/sync_requirements.py`)

`pyproject.toml` (`[project].dependencies`) is the single source of truth for
runtime dependencies. `requirements.txt` is generated from it so the two never
drift. Run the script after adding, removing or re-pinning a dependency:

```sh
python scripts/sync_requirements.py            # regenerate requirements.txt
python scripts/sync_requirements.py --check    # verify only; exit 1 if out of date (use in CI)
```

On Python < 3.11 the script needs `tomli` (included in the `dev` extra).

### RAG retrieval benchmark (`scripts/benchmark_rag.py`)

Measures how well and how fast local retrieval works on your own documents:
indexing time, median/mean query latency, recall@k and mean reciprocal rank
(MRR), plus Ollama's loaded-model hardware status. It builds a temporary
index (removed on exit), so your real RAG index is untouched. Ollama must be
running with the configured `embedding_model` pulled.

```sh
python scripts/benchmark_rag.py \
  --documents ./knowledge \
  --queries ./rag-benchmark-queries.json \
  --top-k 5 \
  --repeats 3
```

`--documents` is the folder to index, `--queries` the labeled query file
(below), `--top-k` the results per query (1–50, default 5) and `--repeats` the
timing repetitions (1–20, default 3).

The query file is a JSON list (max 1,000 entries); `expected_sources` are
paths relative to `--documents`:

```json
[
  {
    "query": "How do I recover the local index?",
    "expected_sources": ["recovery-guide.md"]
  }
]
```

Higher recall@k and MRR mean the right documents are found (and ranked
earlier); lower latency is better. Results depend on the embedding model,
chunk settings (`rag.*`) and machine load. Embeddings go to the configured
`ollama_base_url`, so use a local server if the data must stay on this machine.

## Supported RAG Document and File Types

Files under `paths.rag_documents` (or the folder you enter at startup) are
indexed when their extension is in this list (case-insensitive):

| Type | Extensions | How it is read |
| --- | --- | --- |
| Plain text / Markdown | `.txt`, `.md` | UTF-8 text (undecodable bytes ignored) |
| PDF | `.pdf` | Text extracted per page with `pypdf` (page numbers kept; scanned/image-only PDFs yield no text) |
| Source code | `.py`, `.rs`, `.js`, `.ts` | Plain text |
| Structured / web text | `.json`, `.csv`, `.html` | Plain text (not parsed or rendered) |
| SQLite databases | `.db`, `.sqlite`, `.sqlite3`, `.db3` | Opened read-only; each table's rows are rendered as `column=value` text in batches of 10 rows, up to `rag.sqlite_max_rows_per_table` rows per table; BLOBs are summarised by size; files without a valid SQLite header are skipped |

Other behaviour:

- Folders are scanned recursively. Symlinked files and folders (including
  targets outside the RAG folder) are followed, hard links and duplicate
  targets are indexed once, and link loops are skipped.
- Hidden files and folders (names starting with `.`) and unsupported
  extensions (for example `.docx`, `.xlsx`, images) are ignored.
- Limits: `rag.max_file_bytes` per file, `rag.max_corpus_bytes` in total,
  `rag.max_pdf_pages` per PDF and `rag.max_documents` chunks; files over a limit
  are skipped with a warning.
- Text is split by the configurable `chunking` section (strategies: `fixed`,
  `character`, `word`, `token`, `sentence`, `paragraph`, `line`, `recursive`,
  `markdown`, `code`, `semantic`, or `auto` to pick by file extension). Each
  retrieved chunk is cited by source path (and page for PDFs). Changing the
  chunking settings re-embeds the affected files on the next index update.
  `semantic` embeds sentences with the local embedding model and cuts where
  topic similarity drops; it falls back to `sentence` if embedding fails.
- The index updates automatically when files are added, changed or removed
  (see `rag.auto_refresh_seconds`).

## Configuration Variables

Settings live in `~/.private_agent.conf` (JSON). The block below shows the
defaults with descriptions; real JSON files cannot contain comments, so omit
the `//` text in your file. Flat legacy keys are still accepted; categorized
values win on conflict.

```jsonc
{
  "paths": {
    "database": "~/.local_ai_memory.db",              // SQLite memory file
    "workspace": ".",                                 // Root the agent may read/edit (locked at startup)
    "code_output": "~/.private_agent/projects",       // Where generated code projects go
    "skills": "~/.private_agent/resources/skills",    // Folder of Markdown skills
    "rag_documents": "~/.private_agent/rag/",         // Documents indexed for RAG
    "rag_index": "~/.private_agent/rag_index"         // Persistent ChromaDB index
  },
  "models": {
    "ollama_base_url": "http://localhost:11434",      // Ollama server URL
    "local_openai_compatible_endpoints": [            // Loopback local APIs discovered on supported desktop platforms
      {"name": "LM Studio", "base_url": "http://127.0.0.1:1234/v1"},
      {"name": "llama.cpp", "base_url": "http://127.0.0.1:8080/v1"}
    ],
    "hardware_acceleration": "auto",                  // auto | cpu | accelerator (preference only)
    "preferred_model": null,                          // Default local model name
    "online_base_url": "https://api.openai.com/v1",   // OpenAI-compatible endpoint (HTTPS or loopback)
    "online_model": null,                             // Default online model id
    "online_permission_override": false,              // true = online runs get ALL tools (incl. local-only location/media) without the per-session tool prompt; arguments/results go to the provider. Workspace/permission-mode approvals still apply
    "online_api_key": null,                           // Optional saved API key; online startup offers saved details or manual URL/key entry. Plain text: restrict the file (chmod 600)
    "temperature": 0.1,                               // Sampling temperature
    "embedding_model": "nomic-embed-text",            // Ollama embedding model for RAG
    "thinking_enabled_by_default": false,             // Start with thinking on
    "thinking_effort": "medium",                      // low | medium | high
    "online_request_timeout_seconds": 60,             // Online API request timeout
    "online_model_list_timeout_seconds": 10,          // Timeout for listing online models
    "online_max_retries": 1,                          // Online API retries
    "online_model_list_limit": 30                     // Max online models shown
  },
  "agent": {
    "permission_mode": "auto",                        // Default session mode: manual | auto | full
    "max_tool_iterations": 15,                        // Online plan/act rounds, only if consented
    "max_tool_calls": 60,                             // Online calls per request, only if consented
    "max_task_seconds": 600,                          // Wall-clock budget per request
    "max_tool_output_chars": 12000,                   // Tool output kept in context
    "max_history_messages": 20,                       // Chat messages kept in context
    "max_context_tokens": 12000,                      // Prompt token budget
    "max_output_tokens": 2048,                        // Model output token cap
    "max_model_capability_cache_entries": 128,        // Cached model capability lookups
    "summary_prompt_sentences": 2,                    // Target sentences for session summary
    "context_compaction_threshold": 0.8,              // Auto-summarize older context once the prompt reaches this fraction of the window (0 = off, old messages are simply dropped)
    "context_keep_recent_messages": 6,                // Newest messages kept verbatim when history is summarized
    "context_summary_chars": 1500,                    // Max size of the running session context summary
    "rag_context_results": 2,                         // RAG chunks added to each prompt
    "streaming_output": true,                         // Stream model output as it arrives
    "visible_reasoning": false,                       // Show model reasoning when available
    "system_prompt": "<packaged default>"             // System prompt (empty = packaged default)
  },
  "memory": {
    "conversation_retention_days": 0,                 // Delete chat older than N days (0 = keep)
    "episode_retention_days": 0,                      // Delete episode summaries older than N days (0 = keep)
    "registered_skill_retention_days": 0,             // Expire session-linked skills (0 = keep)
    "task_stale_after_days": 0,                       // Mark idle tasks stale (0 = never)
    "max_summary_chars": 2000,                        // Max episodic summary length
    "max_learned_item_chars": 1000,                   // Max learned memory item length
    "learned_item_expiry_days": 365,                  // Learned item expiry
    "learned_item_review_days": 180,                  // Learned item review interval
    "max_resume_state_chars": 12000,                  // Max saved task resume state
    "max_memory_context_tokens": 1200,                // Memory tokens injected per prompt
    "max_relevant_episodes": 3,                       // Relevant episodes recalled
    "default_history_messages": 20,                   // Default history rows read
    "default_episodic_summaries": 5                   // Default summaries read
  },
  "rag": {
    "max_file_bytes": 5242880,                        // Per-file size cap (5 MiB)
    "max_corpus_bytes": 26214400,                     // Total corpus cap (25 MiB)
    "max_pdf_pages": 250,                             // Pages read per PDF
    "max_documents": 20000,                           // Max indexed chunks/documents
    "chunk_size_chars": 1200,                         // Legacy chunk size (used when chunking.chunk_size is null)
    "chunk_overlap_chars": 200,                       // Legacy chunk overlap (used when chunking.chunk_overlap is null)
    "similarity_results": 4,                          // Candidates fetched per query
    "sqlite_max_rows_per_table": 5000,                // Rows indexed per table of an indexed SQLite file
    "auto_refresh_seconds": 30                        // Min seconds between auto-update checks (0 = off)
  },
  "chunking": {
    "strategy": "character",                          // fixed | character | word | token | sentence | paragraph | line | recursive | markdown | code | semantic | auto
    "chunk_size": null,                               // Max chunk size in `unit`s (null = rag.chunk_size_chars)
    "chunk_overlap": null,                            // Overlap in `unit`s, smaller than size (null = rag.chunk_overlap_chars)
    "unit": "chars",                                  // chars | words | tokens (token = word or punctuation mark)
    "min_chunk_chars": 0,                             // Merge chunks shorter than this into a neighbour (0 = off)
    "separators": ["\n\n", "\n", ". ", " "],           // Split order for the recursive strategy (and markdown/auto sections)
    "semantic_breakpoint_percentile": 90,             // Semantic: split where neighbour distance exceeds this percentile (0-100)
    "semantic_buffer_sentences": 1,                   // Semantic: neighbouring sentences blended into each embedding
    "semantic_max_sentences": 2000                    // Semantic: larger documents fall back to sentence chunking
  },
  "network": {
    "web_research_enabled": true,                     // Allow online research tools
    "web_research_consent": "ask",                    // ask | session | never (Full mode skips the prompt)
    "internet_check_host": "1.1.1.1",                 // Host used for the connectivity check
    "internet_check_port": 443,                       // Port used for the connectivity check
    "internet_check_timeout_seconds": 3.0,            // Connectivity check timeout
    "max_concurrent_requests": 4,                     // Parallel network requests
    "request_timeout_seconds": 30,                    // HTTP request timeout
    "max_stream_timeout_seconds": 600,                // Max streamed download time
    "max_webpage_bytes": 2097152,                     // Max page bytes fetched
    "max_download_bytes": 26214400,                   // Max file download size
    "max_http_redirects": 5,                          // Redirects followed
    "max_search_query_chars": 2000,                   // Max search query length
    "max_search_results": 3,                          // Search results returned
    "max_fetched_webpage_chars": 12000                // Page text kept in context
  },
  "tools": {
    "max_read_file_bytes": 1048576,                   // Max bytes read per file
    "file_read_chunk_bytes": 65536,                   // Read chunk size
    "binary_probe_bytes": 2048,                       // Bytes probed to detect binary files
    "shell_command_timeout_seconds": 30,              // Sandboxed command timeout
    "default_history_read_limit": 20,                 // Default rows for history-read tools
    "override_tool_list": []                          // Restrict the exposed tool list (empty = all)
  },
  "media": {
    "max_captured_image_bytes": 2097152,              // Max image size
    "max_images_per_turn": 3,                         // Images attached per turn
    "max_microphone_seconds": 30,                     // Max recording length
    "default_microphone_seconds": 5.0,                // Default recording length
    "microphone_sample_rate": 16000,                  // Recording sample rate (Hz)
    "default_camera_device_index": 0,                 // Default webcam index
    "max_camera_device_index": 32,                    // Highest webcam index probed
    "max_microphone_device_index": 128,               // Highest microphone index allowed
    "max_image_width": 1280,                          // Image downscale width
    "max_image_height": 720,                          // Image downscale height
    "jpeg_quality": 85                                // JPEG quality for encoded frames
  },
  "skills": {
    "max_name_chars": 64,                             // Skill name length cap
    "max_description_chars": 500,                     // Skill description length cap
    "max_instruction_chars": 20000,                   // Skill instruction length cap
    "max_preview_chars": 12000,                       // Skill preview length
    "max_iterations": 15,                             // Default tool budget while a skill is active
    "description_preview_chars": 100,                 // Description length in listings
    "relevance_threshold": 1.5,                       // Minimum score to auto-select a skill
    "relevance_exact_match_score": 3.0,               // Score for an exact match
    "relevance_prefix_match_score": 2.0,              // Score for a prefix match
    "relevance_word_match_score": 1.0                 // Score per matching word
  },
  "code_tasks": {
    "git_command_timeout_seconds": 30,                // Git helper timeout
    "test_command_timeout_seconds": 180,              // Project test timeout
    "test_commands": [],                              // Explicit test commands (empty = auto-detect)
    "successful_test_output_chars": 6000,             // Passing test output kept
    "failed_test_output_chars": 3000                  // Failing test output kept
  },
  "logging": {
    "debug_enabled": 0,                               // 1 writes a private JSON Lines run trace (~/.private_agent/logs/, file mode 0600; terminal output is unchanged)
    "level": "INFO"                                   // DEBUG | INFO | WARNING | ERROR | CRITICAL
  },
  "mcp": {
    "auto_approve_tools": [],                         // MCP tool names that skip confirmation
    "servers": {}                                     // MCP servers: {"name": {"command"/"url": ...}}
  }
}
```

Debug logs are one JSON record per line and include UTC time, run ID, source
location, lifecycle event, and full exception tracebacks. Console prompts,
responses, and rendered output are not copied into the log; tool and model
events record operational metadata rather than prompt, result, or argument
bodies. Captured exception details redact known credential patterns and
argument values echoed by tools.

## Source Layout

`src/private_agent/`: `cli.py` (entry point), `config.py`, `agent/`
(`runtime.py` session loop, `interactive.py` prompt and completion,
`permissions.py`, `providers.py`, `prompts.py`), `tools/` (filesystem, network,
media, memory, planning, schemas), `database/memory.py` (SQLite),
`rag/` (indexing, retrieval, citations), `skills/`, `sandbox/` (Linux
isolation), `code_tasks/` (workspaces, checkpoints, tests), `hardware/`.
`main.py` is a source-checkout launcher; `scripts/` holds
`benchmark_rag.py` and `sync_requirements.py`; tests are in `tests/`.
