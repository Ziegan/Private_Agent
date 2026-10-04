# Private Agent

Private Agent is a local-first command-line assistant built on Ollama. It
combines local chat models, SQLite conversation memory, optional local document
retrieval, Markdown skills, built-in tools, and explicitly configured MCP
servers. Multi-step work can use an iterative tool loop with configurable
limits.

## Source layout

The installable Python package is `private_agent`, located under `src/`.
Feature packages include `tools/` (including `media_tools.py`), `rag/`,
`skills/`, `sandbox/`, `database/`, `hardware/`, and `code_tasks/`.
`private_agent.agent` coordinates these packages and `private_agent.config`
provides shared configuration. `main.py` remains a small source-checkout
compatibility launcher; the package also supports `python -m private_agent`
after installation.

## Privacy and execution boundaries

- Chat and embedding requests use the configured Ollama endpoint. Keep
  `ollama_base_url` pointed at a trusted local service for local-only
  inference.
- Web research is enabled by default and does not require a saved plan.
  With `web_research_consent: "ask"` (default), every outbound search, fetch,
  or download shows the request and asks for consent, regardless of the general
  tool-permission mode. `"session"` asks once per agent session; `"never"`
  blocks all online research. The agent checks
  connectivity before making any request.
  If offline, it offers a retry or allows the user to continue with clearly
  incomplete offline information. Set `enable_web_research` to `false` to
  disable it, or set `web_research_consent` to `never` to block all network
  research regardless of permission mode.
- At startup, choose a tool-permission mode: **Manual** asks before every tool
  action, **Auto** allows reads/searches and asks before changes, commands, and
  MCP tools not on the allowlist, and **Full** permits model-requested tool
  actions for that session. `agent_permission_mode` sets the default
  (`manual`, `auto`, or `full`). Full mode does not bypass disabled network
  research, the `never` network policy, or explicit confirmation for deleting
  SQLite chat history.
- At startup, an optional local-data reset menu can clear the configured
  Chroma index, SQLite conversation messages and summaries, or both. Reset
  requires typing `RESET`; a declined or invalid confirmation leaves the data
  untouched.
- Model selection offers an explicit **Online (OpenAI-compatible API)** mode.
  Remote endpoints receive no conversation history or RAG context unless you
  opt in for that session. Local/MCP tools are disabled for the remote model
  unless you separately opt in; if enabled, their arguments and results may be
  sent to the provider.
- Online API keys are entered with hidden input and held for the active
  session only. They are not saved in `.private_agent.conf`. Use only providers
  you trust; remote inference is not private/local inference.
- Configured stdio MCP servers run in the Linux Bubblewrap sandbox with an
  isolated temporary workspace and no network access; they cannot read
  arbitrary host files. They require `bwrap` and `prlimit` and fail closed
  elsewhere. Remote HTTP MCP endpoints use peer-validated, IP-pinned
  connections and may target public hosts or explicitly selected local/private
  IP addresses. MCP tool calls require approval in Manual mode; Auto mode asks
  unless their tool names are explicitly listed in `mcp_auto_approve_tools`.
  Full mode permits model-requested MCP calls for that session. Supported
  transports are stdio, SSE, and Streamable HTTP; WebSocket transport is not
  supported.
- The `create_skill` tool is available when the user explicitly requests a
  reusable skill. The agent drafts Markdown instructions from that request
  and relevant context, then applies the active permission mode before writing
  only a new `.md` file in the configured skills directory; existing skills
  are never overwritten. Approved skills are available immediately and on
  later runs. A duplicate-name error is shown in the terminal with the tool's
  returned explanation; the same error details are sent back to the model for
  recovery. Successful tool results are shown in a bounded terminal preview
  while the full result is sent to the model. Tool error output is bounded by
  `agent.max_tool_output_chars`, and sensitive values copied from tool
  arguments are redacted.
- Optional local media tools can capture a webcam frame, load a workspace
  image, or sample up to three frames from a workspace video (maximum 120
  seconds) for a vision-capable Ollama model. Media files must be within the
  active workspace; video and audio files are limited to 25 MiB and image
  files to 2 MiB. Image headers and video dimensions are checked against a
  20-megapixel limit before frame decoding. Audio input supports local
  microphone recording and PCM WAV files up to 30 seconds,
  transcribed with a locally installed Whisper model. Manual and Auto modes
  prompt before each media access; Full mode permits it for the session.
  Captured/normalized image frames remain transient and are not saved. Audio
  is transcribed locally and is not sent to a remote service. Media tools are
  disabled for Online providers. Transcripts are ordinary conversation content
  and may appear in replies and saved conversation history. Direct audio
  input to a chat model is not implemented.
- Workspace file, text, and Python/JavaScript/TypeScript/Go/Rust symbol searches
  are bounded and read-only, as is Git status/diff/log/branch inspection.
  Symbol search matches exact case-insensitive declaration names in supported
  source files; it is not a full language-server index. File patch previews use
  exact unique text context;
  patch application, rename, deletion, and ordinary edits require approval and
  create recoverable snapshots inside the active workspace. Search skips
  hidden paths and symlinks. File tools validate paths against the workspace
  root and use no-follow, descriptor-relative operations for writes, patches,
  renames, and deletes; those mutations fail closed on platforms without the
  required OS support. Shell commands require per-command approval and run
  without shell interpretation in a Bubblewrap
  Linux sandbox with workspace-only writes, dropped capabilities, and CPU,
  memory, process, and file-size limits. Network is disabled by default. A
  coding plan can explicitly approve network-enabled commands; this shares the
  host network namespace and may reach local/private network destinations.
  Generated
  project tests use the same sandbox. Execution fails closed if Linux,
  `bwrap`, or `prlimit` is unavailable; this sandbox is not currently
  implemented on macOS or Windows.
- Web searches, webpage fetches, downloads, Ollama requests, remote HTTP MCP,
  and explicit Online API requests use a peer-validated, IP-pinned transport
  with environment proxies disabled. Web destinations must resolve publicly;
  explicitly configured Ollama/MCP and Online endpoints may use local/private
  IP literals or `localhost`. Web redirects are revalidated and responses
  have size/time limits. DNS checks are application-level safeguards, not a
  substitute for network egress controls against a hostile or rebinding DNS
  environment.
- Media input is optional: webcam and sampled video frames reach only a
  vision-capable local model. Microphone and PCM WAV speech reach the model as
  locally generated transcripts. Direct audio input to the chat model remains
  unsupported.

## Setup

### Supported environments

| Component | Support |
| --- | --- |
| Python | 3.10–3.14 are declared; local validation was run with Python 3.14 |
| Operating system | Linux is supported; macOS and Windows can use non-isolated features but are not fully verified |
| Isolated shell, code-task tests, and stdio MCP | Linux with `bubblewrap` (`bwrap`) and `prlimit`; these features are disabled when unavailable |

If the configured Ollama server is unavailable, startup continues and offers
Online provider selection; local inference and RAG embeddings remain
unavailable until Ollama is reachable. If Linux isolation or its required
tools are unavailable, shell execution, isolated code tasks, and stdio MCP
servers are disabled rather than run without a sandbox. Other available
features, including local file tools and remote HTTP MCP, remain usable.

1. Install a supported Python version and project dependencies:

   ```sh
   python -m pip install -r requirements.txt

   # For development and tests:
   python -m pip install -e ".[dev]"
   ```

2. Install and start [Ollama](https://ollama.com/), then pull a chat model and
   (if using RAG) an embedding model supported by your local Ollama server.
   On Linux, install `bubblewrap` (`bwrap`) and `util-linux` (`prlimit`) to
   enable shell, project-test, and stdio MCP execution; these features are
   intentionally disabled when either isolation tool is unavailable.
3. Run the CLI:

   ```sh
   python main.py
   ```

After installing the package, `private-agent` and `python -m private_agent`
are equivalent entry points.

The first run creates `~/.private_agent.conf` in the current user's home
directory, regardless of the installation method or launch directory. Older
flat-format config files are migrated to categorized sections while preserving
their values. The same per-user file is used by both `python main.py` and the
installed `private-agent` command. Conversation history is stored in
`~/.local_ai_memory.db`.

Use `/maintenance` during a session to review every saved episodic summary and
the local RAG index. Enter `view <ID>` to inspect one, `ask <ID> <question>` to
ask the selected model about it (local episodes are not sent to an Online
provider without the session's explicit context-sharing approval), or
`delete <ID>` and then type the exact confirmation to remove only that summary.
RAG maintenance supports `rag status`, `rag list`, confirmed `rag reindex`, and
typed-confirmation `rag reset`; reset removes only the local index, not source
documents. Local startup also provides a bounded set of recent episode summaries
and registered skill files as reference context, after checking that skill files
are inside the configured skills directory.
Startup reports the active SQLite file plus saved conversation and episode
counts. Conversation messages are stored separately from episodic summaries;
slash commands such as `/maintenance` are not chat turns, and exiting a session
without chat history or a verified task completion does not create an episode.
Completing a task after every step is verified creates a structured episode
with its topic, description, summary, plan, outcome, and evidence. Local-context
eligible requests can retrieve a small lexical match set from prior episodes;
online providers receive none of that context unless sharing is approved for the
session.
Leaving maintenance runs a SQLite integrity check, optimization, and `VACUUM`;
database errors are reported rather than hidden.

`PersistentMemory.backup_to(path)` creates a transactionally consistent SQLite
backup with owner-only file permissions and refuses to overwrite any existing
file. Restore with `PersistentMemory.restore_backup(source, new_path)` from
`private_agent.database`: it validates the source and restored database, writes
to a new path, and never replaces an existing database. Point the configured
database path at that new file only after reviewing it; keep the original
database untouched until the restored copy is verified. This is a full
database backup, including learned preferences and their enabled/disabled
state; there is no separate learning-only export format.

Use `/skills` to list loaded skills and their command keys. Enter
`/skills <command-key>` to apply a skill to the next request, including a
planning-only request; Tab completes available skill keys. Skill files are
local context, so they are not sent to an Online provider unless local-context
sharing was approved for that session.

For coding tasks, enter `/plan` before describing the request. Coding
implementation is blocked until its exact request has an approved plan and
workspace access scope. For any other task, planning is optional: enter
`/plan` only when you explicitly want a saved checklist. The model can then use
planner tools to save it to the configured SQLite database. Use `/tasks` to
inspect and approve a plan before treating it as active. Plans can include assumptions, constraints, research questions,
step dependencies, edge cases, risks, validation criteria, and proof points.
Dependencies are checked as progress is recorded. Coding approval is scoped
to the requested goal, selected source project, and configured isolated output
root; web research still requires separate connectivity and network consent
checks. After you approve a plan, the agent automatically starts it using the
saved goal; you do not need to repeat the request. If a tool fails or the
approved strategy cannot proceed, the agent should explain what remains
unverified and ask before changing the plan or scope. While drafting a plan,
the model may propose a focused web search when
current information is needed. Runtime checks connectivity and applies the
configured web-consent policy before sending that query; declined/offline results leave research questions
unresolved. Titles and HTTPS URLs are attached to the SQLite plan as untrusted
citations only after malformed URLs, local destinations, and duplicate links
are rejected. An explicitly created research plan stays marked incomplete until
current-revision validation evidence is recorded; approval starts execution but
does not verify its facts or sources. The cited links remain untrusted and
require a separate relevance, authority, and freshness check. Citations cannot
approve the plan or authorize later tool calls. Todo
progress is saved as the model reports it, but reported completion
is not treated as verified until you use
`verify <TASK_ID> <STEP_ID> <evidence>` in `/tasks`. Use `pause` to keep a
resumable task, then `resume` and reapprove it in a later session. `cancel`
ends it; `discard` archives it as abandoned. SQLite reset also removes saved
task plans and todos.
Task plan approval and state-changing `/tasks` commands are read-only in
non-interactive sessions; missing or piped input cannot implicitly approve,
resume, reconcile, or delete task state.
`memory.task_stale_after_days` defaults to `0` (disabled). When enabled, old
unfinished plans with no live process lease are marked stale and their prior
approval is cleared, but their plan, progress, and recovery data are retained.
Review a stale plan with `/tasks`, then explicitly resume and reapprove it or
discard/delete it; stale-task marking never removes saved work.
Record separate task-level validation with
`validate <TASK_ID> <concrete test or cross-check evidence>` before skill
capture; this evidence is tied to the current plan revision and is cleared if
the plan changes. Successful isolated coding-task final tests/checkpoints are
recorded automatically. Research-task validation must be supplied explicitly.

Approved task execution holds a SQLite lease so another process cannot silently
run the same task. Tool calls made for an active task are journaled before
execution; only argument/output digests are retained. If a process stops during
a tool call, the outcome is treated as unknown and later task actions are
blocked until you inspect it with `/tasks view` and explicitly reconcile it
using `/tasks reconcile <TASK_ID> <ACTION_ID> completed|not-run`. Verify the
real-world or workspace result before selecting either resolution. Reapproval
after a restart does not bypass normal tool permissions or confirmations.
Use `/tasks delete <TASK_ID>` to permanently remove an inactive task and its
linked episode, registered skill, and task-derived learned preference after an
explicit confirmation. Linked skill files are removed only when they resolve
inside the configured skills directory; unsafe external paths prevent deletion.

After a completed coding or research task with current-revision validation
evidence, `/tasks complete` can record a 0–5 completeness rating. A rating of 5 separately asks whether to
draft a skill; naming and reviewing the preview are not enough to save it, as a
final explicit confirmation is required. Skill files are not overwritten and
are linked to their source session in SQLite. User-confirmed adaptive learning
is managed through `/maintenance` commands such as `learn add`, `learn list`,
`learn correct`, `learn review`, `learn disable`, `learn delete`, and `learn off`;
use `learn propose <TASK_ID> <scope> <statement>` to submit your own preference,
or `learn suggest <TASK_ID> [scope]` to ask the selected local model for one
candidate grounded in a completed task's verified steps and current-revision
validation evidence. Suggestions are unavailable with an online provider; the
model proposal is shown for review and is saved only after separate preview
and save confirmations.
Learned items are scoped, removable, and included only in eligible local-context
turns. Items expire after the configured `memory.learned_item_expiry_days`
(default 365) and stop being retrieved when their scheduled review date arrives
(`memory.learned_item_review_days`, default 180). `learn list` shows expiry and
review state; `learn review <ID>` renews an item only after confirmation.
The statement budget is configurable with `memory.max_learned_item_chars`
(default 1000). The startup memory index is capped by
`memory.max_memory_context_tokens` (default 1200 tokens), and per-request
episodic retrieval is capped by `memory.max_relevant_episodes` (default 3).
When these budgets omit context, the prompt reports omitted episode or skill
reference counts. Ratings alone never create learned preferences.
Task recovery stores a bounded resume capsule containing plan/todo references
and the next safe action, not hidden reasoning, credentials, transcripts, or
raw tool output. Its character limit is `memory.max_resume_state_chars`
(default 12000). Skill previews are similarly bounded by
`skills.max_preview_chars` (default 12000); an oversized draft is rejected
rather than silently truncated.

Model-authored SQLite writes are schema-aware: the model must first call the
read-only `read_table_schema` tool for the destination and related tables, then
prepare values that respect their declared types and application constraints.
Text that exceeds a stated field budget must be semantically summarized for
that field; values are validated rather than silently truncated. This tool
does not expose arbitrary SQL or grant write permission. Runtime-owned chat
message persistence and typed database maintenance remain application
operations rather than model-generated SQL.

The installed command and `python main.py` use these same config paths,
independent of the package installation location. At startup the effective RAG
documents, RAG index, and skills paths are shown. The default RAG source is
`~/.private_agent/rag/`; press Enter at the RAG prompt to use that configured
directory, or enter a project-specific directory to use it for the current
run. The default skills directory is
`~/.private_agent/resources/skills`. Tilde and relative paths in the config
are resolved from the user's home directory; a relative path entered at the
RAG prompt is resolved from the current working directory.

Set `paths.rag_documents` and `paths.skills` in `~/.private_agent.conf` to
change these defaults. The RAG index defaults to `~/.private_agent/rag_index`,
separate from its source documents.

The default system instructions are packaged separately from the Python code
and copied into `agent.system_prompt` when `~/.private_agent.conf` is created.
Edit that string in the config to customize the assistant's identity and
working standards; runtime permission, privacy, and tool-budget rules remain
enforced independently. For coding requests with RAG enabled, the agent also
searches the local index specifically for coding manuals and project standards.

To explicitly include tools normally hidden by platform, provider, or model
capability filtering, add their exact names to `tools.override_tool_list` in
`~/.private_agent.conf`, for example:

```json
"tools": {
  "override_tool_list": ["run_shell_command"]
}
```

`/list_tools` marks unavailable tools with strikethrough; an unavailable tool
included in this override is shown in red. The override only exposes tools to
the model: it does not bypass execution-time permission checks, provider
restrictions, network consent, or OS sandbox enforcement. Unsupported
operations continue to fail safely; it also cannot add function-calling
support to a model that does not have it.

Set `"logging": {"debug_enabled": 1}` in `~/.private_agent.conf` to write a trace file for
each run under `~/.private_agent/logs/`; set it to `0` to disable logging (the
default). Trace files and their directory are created with owner-only
permissions. A trace includes rendered agent output and interactive prompt
responses and phase timings for startup, MCP loading, RAG indexing/search, and
model calls, which can contain private conversation or tool data; enable it
only when needed and review/delete the resulting files securely. Model text
responses are streamed to the terminal when `agent.streaming_output` is true
(default); set it to `false` to use complete-response mode. Models without a
streaming API always use complete-response mode. After each response the CLI
reports model output tokens per second, preferring provider usage metadata and
otherwise estimating from visible text. The terminal labels model-processing
and tool-output phases; when supported thinking is enabled, it indicates the
effort level without displaying private reasoning text. Set
`agent.visible_reasoning` to `true` (default `false`) to show a bounded
provider-supplied `reasoning_summary` when available; raw reasoning/thinking
tokens are never displayed. The legacy config spelling `VISIBILE_REASONING`
is also accepted.

### Backups and recovery

Stop the agent before making filesystem copies or restoring its local data. The
SQLite database path is `paths.database` (by default
`~/.local_ai_memory.db`). SQLite's backup API produces a consistent backup,
including WAL-managed data:

```sh
python - <<'PY'
import sqlite3
from pathlib import Path

source = Path.home() / ".local_ai_memory.db"
backup = source.with_name(source.name + ".backup")
original = sqlite3.connect(source)
try:
    target = sqlite3.connect(backup)
    try:
        original.backup(target)
    finally:
        target.close()
finally:
    original.close()
PY
chmod 600 ~/.local_ai_memory.db.backup
```

Keep the backup in a private location. To restore, stop the agent, preserve
the damaged database for diagnosis, then copy the chosen backup to the exact
configured `paths.database` location. The agent does not silently recreate a
corrupt conversation database.

Back up the whole configured `paths.rag_index` directory while the agent is
stopped; by default it is `~/.private_agent/rag_index`. If the index state is
corrupt or incompatible, startup offers to move the existing index to a
timestamped backup before rebuilding. Review `paths.rag_documents` and reindex
from those original documents if the vector database cannot be recovered.
Approved workspace edits keep snapshots under `.agent_snapshots` inside the
active workspace; inspect and remove old snapshots manually when no longer
needed.

Linux is the only platform with a verified execution-isolation backend.
Shell, project-test, and stdio MCP execution fail closed where Bubblewrap and
`prlimit` are unavailable; other operating systems are not yet supported for
those operations.

### Optional pip installation

Installing the CLI package is optional; running from the source checkout is
also supported. From the project root, install the package and its
`private-agent` command with:

```sh
python -m pip install .
private-agent
```

For editable development installation with test tools, use
`python -m pip install -e ".[dev]"`. This project configuration provides a
local pip-installable distribution; the commands above install from the
checkout rather than assuming a published PyPI release.

Camera, video, and audio integrations are optional and are not installed with
the base package. Install them from the checkout with
`python -m pip install '.[media]'`. Linux microphone support may also require
the system PortAudio library. Set `PRIVATE_AGENT_WHISPER_MODEL_PATH` to an
existing local faster-whisper model directory before using microphone or WAV
transcription; the application will not download a speech model.

Explicit code creation or modification requests require an approved plan.
When approving a coding plan, choose one of five access levels: (1) direct
access to the selected workspace with approval before every tool action;
(2) direct access with read/search allowed and approval before edits/commands;
(3) direct access with edits allowed and approval/visibility for every
command; (4) an isolated copy with moderate governance (the default); or
(5) an isolated copy with approval before every tool action. Direct access is
limited to the workspace root selected at session start and does not edit
outside that directory. Levels 1–3 allow approved commands to use the network
(for example, to install packages into a project venv); commands still run in
Bubblewrap with workspace-only writes and resource limits. Because this shares
the host network namespace, commands may reach local/private network
destinations. Levels 4–5 keep command networking disabled. For copied projects,
Git metadata, symlinks, likely secrets, and common caches/build outputs are
excluded. Git is initialized only inside the generated project, with no remote
configured. Direct workspaces do not create Git commits. If Git is unavailable
or cannot create the initial checkpoint, the CLI asks before continuing without
Git. Code-task checkpoints run the project tests and require a unit-test file;
final verification runs the tests again.
Python (`pytest` or
`unittest`), npm, Go, and Cargo test projects are recognized. This is not an
OS-level container boundary, but supported Linux project commands run in the
Bubblewrap sandbox described above. Tests cannot read host files outside the
explicit read-only runtime mounts or write outside the project and temporary
filesystem. With network-enabled access levels, network isolation is
intentionally relaxed and host loopback/local-network access may be possible.
If the sandbox prerequisites are missing, code-task verification is blocked
rather than run on the host.
For other test runners, configure `code_tasks.test_commands` as a list of
argument arrays, for example `[["my-test-runner", "--all"]]`. Commands run
without a shell, under the same sandbox and timeout as detected test suites.
Only configure commands you trust; they execute project-controlled code.

RAG indexing is bounded by `rag.max_file_bytes` (5 MiB per file),
`rag.max_corpus_bytes` (25 MiB total), `rag.max_pdf_pages` (250 per PDF), and
`rag.max_documents` (20,000 chunks). The index location is configurable with
`paths.rag_index`. If its state file is corrupt or incompatible, the CLI offers
to move the old index intact to a timestamped backup before rebuilding.
If opening or updating the Chroma database itself fails, the CLI likewise
offers an explicit `REBUILD` confirmation; the previous index is moved intact
to a unique sibling backup before reconstructing the current documents.

Conversation context is limited by `agent.max_context_tokens` (12,000 tokens by
default), or the smaller context window Ollama declares for the selected model.
The system prompt and a bounded generation reserve are accounted for before
history and retrieved context; `agent.max_output_tokens` caps generated output
at 2,048 tokens by default. Online-compatible providers receive the same output
cap, while their context window remains bounded by the configured agent limit.
Retention is disabled by default (`0` days). Configure
`memory.conversation_retention_days` to expire conversation messages and
summaries; this preserves older configurations that applied retention to both.
Set `memory.episode_retention_days` to override the summary retention period,
and `memory.registered_skill_retention_days` to independently expire skill
records. Old episodes and registered skills linked to resumable tasks are
retained until the task is no longer resumable. Expired skill metadata is
removed; its file is deleted only if it resolves inside the configured skills
directory. Session summaries are also capped by `memory.max_summary_chars`.
The agent can search bounded local
history and summaries, then delete a specific result only after confirming its
displayed entry ID; broad per-session/all-history deletion remains separately
approval-gated. All outbound HTTP for web tools, Ollama,
MCP over HTTP, and Online APIs validates DNS answers immediately before
connecting and connects to the validated IP. Web tools only reach public
addresses. Explicitly configured Ollama, MCP, and Online endpoints may use
loopback or private IP literals; configure them only for trusted services.

### Hardware acceleration

Ollama selects inference hardware on the machine running the Ollama server.
Private Agent reports loaded-model VRAM usage from Ollama's runtime API when
available; it does not claim a particular accelerator backend because that API
does not identify it. With a remote `ollama_base_url`, the report describes
that server, not the CLI host. The active Ollama build and platform determine
whether CUDA, ROCm, Vulkan, Apple Metal, Intel/OpenVINO, or CPU execution is
available. Unsupported accelerators remain an Ollama installation/runtime
limitation; Private Agent does not install GPU drivers or switch inference to
an online provider.

Set `hardware_acceleration` to `auto` (default), `cpu`, or `accelerator`.
`auto` delegates device selection to Ollama. `cpu` requests `num_gpu=0` from
Ollama; actual behavior depends on server/runtime support. `accelerator`
is a preference label only: Ollama's request API cannot force an accelerator,
so it uses the same automatic selection as `auto`. Neither mode can select a
specific vendor backend. Use `/hardware-status` to refresh runtime placement
information after loading a model.

## Configuration

Runtime limits and defaults are grouped by subsystem in
`~/.private_agent.conf`. Existing flat keys are still accepted for backwards
compatibility; when the same setting appears in both forms, the categorized
value takes precedence. The following are the generated defaults:

```json
{
  "paths": {
    "database": "~/.local_ai_memory.db",
    "workspace": ".",
    "code_output": "~/.private_agent/projects",
    "skills": "~/.private_agent/resources/skills",
    "rag_documents": "~/.private_agent/rag/",
    "rag_index": "~/.private_agent/rag_index"
  },
  "models": {
    "ollama_base_url": "http://localhost:11434",
    "hardware_acceleration": "auto",
    "preferred_model": null,
    "online_base_url": "https://api.openai.com/v1",
    "online_model": null,
    "temperature": 0.1,
    "embedding_model": "nomic-embed-text",
    "thinking_enabled_by_default": false,
    "thinking_effort": "medium",
    "online_request_timeout_seconds": 60,
    "online_model_list_timeout_seconds": 10,
    "online_max_retries": 1,
    "online_model_list_limit": 30
  },
  "agent": {
    "permission_mode": "auto",
    "max_tool_iterations": 15,
    "max_tool_calls": 60,
    "max_task_seconds": 600,
    "max_tool_output_chars": 12000,
    "max_history_messages": 20,
    "max_context_tokens": 12000,
    "max_output_tokens": 2048,
    "max_model_capability_cache_entries": 128,
    "summary_prompt_sentences": 2,
    "rag_context_results": 2,
    "streaming_output": true,
    "visible_reasoning": false
  },
  "memory": {
    "conversation_retention_days": 0,
    "episode_retention_days": 0,
    "registered_skill_retention_days": 0,
    "task_stale_after_days": 0,
    "max_summary_chars": 2000,
    "max_learned_item_chars": 1000,
    "learned_item_expiry_days": 365,
    "learned_item_review_days": 180,
    "max_resume_state_chars": 12000,
    "max_memory_context_tokens": 1200,
    "max_relevant_episodes": 3,
    "default_history_messages": 20,
    "default_episodic_summaries": 5
  },
  "rag": {
    "max_file_bytes": 5242880,
    "max_corpus_bytes": 26214400,
    "max_pdf_pages": 250,
    "max_documents": 20000,
    "chunk_size_chars": 1200,
    "chunk_overlap_chars": 200,
    "similarity_results": 4
  },
  "network": {
    "web_research_enabled": true,
    "web_research_consent": "ask",
    "internet_check_host": "1.1.1.1",
    "internet_check_port": 443,
    "internet_check_timeout_seconds": 3.0,
    "max_concurrent_requests": 4,
    "request_timeout_seconds": 30,
    "max_stream_timeout_seconds": 600,
    "max_webpage_bytes": 2097152,
    "max_download_bytes": 26214400,
    "max_http_redirects": 5,
    "max_search_query_chars": 2000,
    "max_search_results": 3,
    "max_fetched_webpage_chars": 12000
  },
  "tools": {
    "max_read_file_bytes": 1048576,
    "file_read_chunk_bytes": 65536,
    "binary_probe_bytes": 2048,
    "shell_command_timeout_seconds": 30,
    "default_history_read_limit": 20
  },
  "media": {
    "max_captured_image_bytes": 2097152,
    "max_decoded_image_pixels": 20000000,
    "max_media_file_bytes": 26214400,
    "max_images_per_turn": 3,
    "max_microphone_seconds": 30,
    "max_video_seconds": 120,
    "default_microphone_seconds": 5.0,
    "microphone_sample_rate": 16000,
    "default_camera_device_index": 0,
    "max_camera_device_index": 32,
    "max_microphone_device_index": 128,
    "max_image_width": 1280,
    "max_image_height": 720,
    "jpeg_quality": 85
  },
  "skills": {
    "max_name_chars": 64,
    "max_description_chars": 500,
    "max_instruction_chars": 20000,
    "max_preview_chars": 12000,
    "max_iterations": 15,
    "description_preview_chars": 100,
    "relevance_threshold": 1.5,
    "relevance_exact_match_score": 3.0,
    "relevance_prefix_match_score": 2.0,
    "relevance_word_match_score": 1.0
  },
  "code_tasks": {
    "git_command_timeout_seconds": 30,
    "test_command_timeout_seconds": 180,
    "test_commands": [],
    "successful_test_output_chars": 6000,
    "failed_test_output_chars": 3000
  },
  "logging": {
    "debug_enabled": 0
  },
  "mcp": {
    "auto_approve_tools": [],
    "servers": {}
  }
}
```

Settings are read at process startup; restart `private-agent` after changing
them. Existing settings are retained when the legacy flat config is migrated.
For example, raise the file-reader size limit by editing
`"tools": {"max_read_file_bytes": 10485760}` (10 MiB). This is independent of
`agent.max_tool_output_chars`, which separately limits how much tool output is
passed back to the model. Security invariants such as workspace path checks,
network address validation, and OS sandbox resource ceilings are intentionally
not user-configurable.

`requirements.txt` is generated from the runtime dependencies in
`pyproject.toml`; after changing those dependencies, regenerate it with
`python scripts/sync_requirements.py`. Install test and lint dependencies with
`python -m pip install -e ".[dev]"`.
`python -m pytest -q` runs the test suite.

### RAG retrieval benchmark

To measure indexing time, median/mean retrieval latency, recall@k, and mean
reciprocal rank against labeled queries, run:

```sh
python scripts/benchmark_rag.py \
  --documents ./knowledge \
  --queries ./rag-benchmark-queries.json \
  --top-k 5 \
  --repeats 3
```

The query file is a JSON list with query text and relevant paths relative to the
documents folder:

```json
[
  {
    "query": "How do I recover the local index?",
    "expected_sources": ["recovery-guide.md"]
  }
]
```

The benchmark builds a temporary index and removes it on exit. Indexing and
queries are sent to the configured Ollama endpoint for embeddings/retrieval;
use a local Ollama server if benchmark data must remain on this machine. The
report also includes Ollama's loaded-model hardware status. Search timing and
quality depend on the selected local models, index state, and machine load.

The default embedding model in the generated configuration may differ from
this example; set it to a model actually installed in your Ollama instance.
Supported thinking effort values are `low`, `medium`, and `high`. The runtime
uses Ollama model metadata to identify declared tool, thinking, vision, and
audio support; unsupported features are not silently assumed.

MCP server entries use the configuration format expected by
`langchain-mcp-adapters`; configure them under `mcp.servers`. For example, a
local stdio server may be configured as:

```json
{
  "mcp": {
    "servers": {
      "example": {
        "transport": "stdio",
        "command": "python",
        "args": ["-m", "example_mcp_server"]
      }
    }
  }
}
```

Only configure servers you trust. Tool names in `mcp.auto_approve_tools`
persistently bypass the per-call interactive approval prompt; leave this list
empty unless the server and the named tools have been reviewed. Stdio servers
must be installed in the application/runtime locations mounted read-only into
the sandbox; arbitrary host scripts and data paths are intentionally
inaccessible.

## CLI controls

- `exit`: end the session and save an episodic summary. Completed turns are
  saved as they finish; pressing `Ctrl+C` interrupts and exits without making
  a summary, and an in-progress turn may not be saved.
- `switch`: return to model selection.
- `/think on` or `/think off`: enable or disable supported model thinking.
- `/think-effort low|medium|high`: request a supported reasoning effort.
- `/think-status`: show requested/effective thinking state and model support.
- `/hardware-status`: refresh Ollama's loaded-model hardware placement report.
- `/list_tools`: show the currently available tools, descriptions, origins, and
  permission boundaries (including tools loaded from configured MCP servers).
- `/help`: print available session commands and workspace-file context syntax.
- `@path/to/file` in a prompt: include a UTF-8 text file from the active
  workspace as untrusted, request-only context. Tab completes commands and
  workspace paths in an interactive terminal. Attached file contents are sent
  to the selected model but are not saved to chat history.

The agent offers to resume the most recent session when run interactively.
Choose **Online** from the provider screen to configure a compatible endpoint.
The API base URL and model identifier can be preconfigured, but the API key is
requested interactively each run and is never written to configuration. The
provider's model list is used when the endpoint supports it; otherwise enter
the model identifier manually. Remote endpoints must use HTTPS except for
loopback development services.

## Improvement roadmap

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for prioritized fixes,
acceptance criteria, and remaining work. In particular, the current runtime
does not yet provide an OS-isolated code execution environment or image/audio
input support.
