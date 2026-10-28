# Code Review: Issues, Deviations, Gaps, Runtime, UI, Platform and Provider Plans, and Fix Roadmap

Review date: 2026-10-28

## How this was checked

- **Tests:** 499 passed, 10 failed, 1 skipped (about 35s). `pytest` is not
  installed in `.venv`, so the suite was run through `uv` with `PYTHONPATH=src`.
- **Lint:** `ruff` 0.16.10 (the version pinned in `pyproject.toml`). The repo
  has no ruff config, so ruff's defaults were used, plus extra rule sets where
  noted.

## Strengths

- **Packaging:** standard `src/` layout with a `pyproject.toml`, exact
  dependency pins, and optional `dev` and `media` extras.
- **Security defaults:** command execution refuses to run when the sandbox
  tools are missing. Tool calls are checked against a workspace boundary,
  sensitive tool arguments are redacted, network calls check for private
  addresses, and there is no `shell=True`, `eval` or `exec`.
- **Database:** SQLite foreign keys are on, there is a busy timeout, and schema
  versions are tracked with `user_version`.
- **Tests and docs:** about 13k lines of tests against about 21k lines of
  source, and a detailed README.

---

## Part A: Code Quality Issues

### A1. Very large modules (highest priority)

- `src/private_agent/agent/runtime.py` is 6,311 lines.
  `_run_agent_cli_session` (lines 3522–6096) is about 2,500 lines in one
  function.
- `src/private_agent/database/memory.py` is 2,929 lines in one
  `PersistentMemory` class.
- Ruff complexity checks flag:
  - 38 overly complex functions (`C901`)
  - 22 functions with too many branches (`PLR0912`)
  - 25 functions with too many arguments (`PLR0913`)
  - 15 functions with too many statements (`PLR0915`)

**Standard practice:** small, single-purpose modules. For example, separate
command handlers for `/tasks`, `/plan` and memory upkeep, and a repository
class for each table.

### A2. Settings loaded at import time as global constants

- `config.py` reads `~/.private_agent.conf` and builds `APP_CONFIG` as soon as
  it is imported, then exposes dozens of module-level constants. `runtime.py`
  imports about 50 of them.
- On import, `config.py` also sets `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE`,
  which affects anything that imports the package.
- Mutable state is kept in module globals: `global` appears 17 times across
  `tools/__init__.py`, `tools/_task_state.py`, `tools/_sqlite_state.py`,
  `tools/media.py` and `run_logging.py`.

**Standard practice:** a typed settings object (for example, a pydantic
model), loaded once at startup and passed to the code that needs it.

### A3. Test-only logic inside production code

- `src/private_agent/config.py:384`: `sys_modules_safe()` checks whether
  `"pytest" in sys.modules` and changes behaviour if so.

**Standard practice:** handle this in tests with fixtures or monkeypatching.

### A4. The test environment is not standard

- `pytest` is listed under the `dev` extras but is not installed in `.venv`.
- Plain `pytest` fails with `ModuleNotFoundError: No module named
  'private_agent'` because `[tool.pytest.ini_options] pythonpath = ["."]`
  should be `["src"]`.
- `.vscode/settings.json` turns pytest off and enables unittest with the
  pattern `*test.py`, which does not match the `test_*.py` files.
- The 10 failing tests:

  | Test | Cause |
  |---|---|
  | `tests/unit/test_tools.py::test_shell_command_tool` | Environmental: `bwrap` is not installed |
  | `tests/unit/test_tools.py::test_workspace_image_input_validates_path_and_size` | Environmental: `media` extras are not installed |
  | `tests/unit/test_code_tasks.py::test_code_workspace_isolated_git_and_test_gated_checkpoint` | Git-related, possibly a real bug (not traced) |
  | `tests/unit/test_code_tasks.py::test_existing_code_workspace_copy_excludes_git_secrets_and_caches` | Git-related, possibly a real bug (not traced) |
  | `tests/unit/test_code_tasks.py::test_project_runner_rejects_ignored_test_files` | Git-related, possibly a real bug: `RuntimeError: Could not determine whether project test 'ignored_test.py' is ignored` |
  | `tests/integration/test_agent_runtime.py::test_repeated_cli_runs_remove_mcp_tools_and_sandboxes` | Not traced |
  | `tests/integration/test_agent_runtime.py::test_mcp_startup_failure_does_not_skip_other_servers_or_shutdown` | Not traced |
  | `tests/integration/test_agent_runtime.py::test_cli_code_task_keeps_failed_final_verification_visible` | Not traced |
  | `tests/integration/test_agent_runtime.py::test_cli_code_task_retries_checkpoint_after_test_failure` | Not traced |
  | `tests/integration/test_agent_runtime.py::test_cli_code_task_workspace_creation_failure_keeps_session_available` | Not traced |

**Standard practice:** tests that need a missing prerequisite (`bwrap`,
`prlimit`, media libraries) should skip, using `pytest.mark.skipif` or
`pytest.importorskip`, not fail.

### A5. Lint, format and typing are not enforced

- With ruff's default rules: 378 lines over 88 characters (`E501`).
- `ruff format --check` would reformat 51 of 74 files.
- With extra rule sets turned on, there are 608 warnings. The main ones:

  | Count | Rule | Issue |
  |---|---|---|
  | 93 | `BLE001` | Blind `except Exception` (103 broad catches in total) |
  | 64 | `I001` | Unsorted imports |
  | 55 | `UP006` | Old-style `typing` annotations (`List`, `Dict`) |
  | 41 | `SIM117` | Nested `with` statements that can be combined |
  | 30 | `UP035` | Deprecated `typing` imports |
  | 19 | `ISC004` | Implicit string concatenation inside collection literals |
  | 7 | `B023` | Loop variable captured in a closure (a real bug class) |
  | 3 | `PLW1510` | `subprocess.run` without `check` |

- There is no `[tool.ruff]` or `[tool.mypy]` config, no pre-commit hooks, and
  no CI workflow under `.github/workflows/`.

### A6. Logging

- There are about 720 `print()` / `console.print` calls but only 8 uses of the
  `logging` module. User output and diagnostics are mixed together, which
  makes problems hard to trace afterwards.

**Standard practice:** use `rich` for user-facing output and the `logging`
module (with module-level loggers) for diagnostics.

### A7. Async mixed with blocking code

- `tools/web.py` (lines 65–66, 129–130) and `tools/location.py` (lines 52–53)
  call `asyncio.run` inside a thread pool to run async code from sync tool
  functions.
- `subprocess.run` is called from async code paths (ruff `ASYNC221`), which
  blocks the event loop.
- `database/memory.py:51` shares one SQLite connection across threads with
  `check_same_thread=False` and no visible lock.

### A8. Smaller issues

- `agent/__init__.py` re-exports every name in `runtime.py`, including private
  ones, so there is no real public API.
- The constant name `VISIBILE_REASONING` (`config.py:212`, `config.py:506`) is
  misspelled. A correct alias exists, but the typo is still used.
- `main.py` edits `sys.path` by hand, which is not needed once the package is
  installed.
- Dependencies are kept in both `requirements.txt` and `pyproject.toml`,
  synced by `scripts/sync_requirements.py`, so there are two sources of truth.
- `.github/copilot-instructions.md` still points to `IMPLEMENTATION_ROADMAP.md`
  and `structured implementation.md`, which are deleted in the working tree.


---

## Part B: Features and Whether They Follow Standard Approaches

### What the program does

Private Agent is a privacy-focused AI agent that runs in the terminal, uses
local models by default, and works on the project folder it is launched from.

| # | Feature | Where it lives |
|---|---|---|
| 1 | **Model providers:** Ollama, LM Studio, llama.cpp, or an online OpenAI-compatible API, with detection of each model's capabilities (tools, vision, thinking) | `agent/providers.py`, `agent/local_runtimes.py` |
| 2 | **Agent loop:** plan, call tools, check results and retry, within a time budget | `agent/runtime.py` |
| 3 | **Permissions:** three session modes (manual, auto, full) and five access levels for coding tasks | `agent/permissions.py` |
| 4 | **Built-in tools:** read, edit and search files; git; patches; shell commands; web search and fetch; location and weather | `tools/` |
| 5 | **Sandboxed commands:** shell and test commands run in Bubblewrap with resource limits | `code_tasks/__init__.py` |
| 6 | **Coding tasks:** isolated project copy, checkpoints, re-running tests until they pass | `code_tasks/` |
| 7 | **Planning:** saved plans with revisions, approval, resume and leases | `tools/planner.py`, `database/memory.py` |
| 8 | **Memory:** chat history, session summaries, learned facts and context compaction in SQLite | `database/memory.py`, `agent/compaction.py` |
| 9 | **Document search (RAG):** ChromaDB vector search plus BM25 keyword search, 12 chunking strategies, citations | `rag/` |
| 10 | **Skills:** Markdown instruction files picked by relevance | `skills/` |
| 11 | **MCP:** external tool servers, local (stdio, sandboxed) or HTTP | `tools/mcp.py` |
| 12 | **Media:** images, video, webcam, microphone, Whisper transcription | `tools/media.py` |
| 13 | **Interactive terminal:** slash commands, auto-completion, `@file` attachments, session statistics | `agent/interactive.py`, `agent/session_stats.py` |

### Parts that follow standard approaches

- **Model access** uses LangChain (`ChatOllama`, `ChatOpenAI`, `bind_tools`,
  `astream` / `ainvoke`).
- **Document search** combines Chroma and BM25, the usual hybrid design.
  Indexing is incremental and tracks which files have changed.
- **MCP** uses the official `langchain-mcp-adapters` client and supports
  `stdio`, `sse` and `streamable_http`.
- **Sandboxing** uses Bubblewrap with `--unshare-all`, read-only system
  folders and `prlimit`, and refuses to run commands when Bubblewrap is
  missing.
- **Workspace boundary:** `sandbox/linux.py` resolves symlinks with `realpath`
  and checks the path is inside the workspace with `is_relative_to`.
- **Memory search** uses SQLite's built-in full-text search (FTS5).
- **Token counting** uses `tiktoken`, falling back to a character estimate.
- **Shell commands** run with `shell=False` and an argument list.

### Parts that deviate from standard approaches

#### B1. The agent loop is written by hand and duplicated

- The tool-calling loop lives inside the 2,500-line `_run_agent_cli_session`.
- `llm.bind_tools(...)` is called in about 12 places in `runtime.py`, so the
  same loop logic is copied for chat, planning, coding and skills.

**Standard practice:** one reusable loop function, or a framework such as
LangGraph (`create_react_agent`). The project already depends on LangChain.

#### B2. Permission checks are spread across several places

- There is a central `tool_needs_permission` in `agent/permissions.py`.
- `tools/shell.py` also asks the user for approval itself with
  `console.input`.
- `runtime.py` has its own `_tool_needs_permission`.

**Standard practice:** a single policy check that every tool call passes
through. Tools never prompt the user directly.

#### B3. Tool state lives in module globals

- The active database, task, skill folder and code task are global variables
  set through `set_active_*()` functions in `tools/__init__.py`,
  `tools/_task_state.py` and `tools/_sqlite_state.py`.
- `code_tasks.ACTIVE_CODE_TASK` is read directly by `tools/shell.py`.

**Standard practice:** pass a context object into each tool, for example
through LangChain's `RunnableConfig` or injected tool arguments.

#### B4. The risky-command check matches text fragments

- `evaluate_shell_command` in `sandbox/linux.py` looks for substrings such as
  `"rm -r"` and `"/usr"`.
- `echo /usr` is flagged as risky, but `rm -fr` is not.
- This only adds a warning to the approval prompt, and Bubblewrap is the real
  protection, so it is not a security hole. It does give a misleading signal.

**Standard practice:** parse the command with `shlex` and check it against an
allow-list or deny-list of programs and flags.

#### B5. The workspace root is decided at import time

- `SandboxManager.root_dir` is set to `Path.cwd()` when the class is first
  loaded.
- `SandboxManager.set_root` creates the folder if it does not exist.

**Standard practice:** set the root once, explicitly, at startup, and fail if
the folder does not exist.

#### B6. Skill matching uses keyword overlap

- `skills/manager.py` picks skills by counting matching words against fixed
  thresholds.
- `skills/loader.py` uses the first characters of the file as the
  description, and the file name as the skill name.

**Standard practice:** YAML front matter with `name` and `description` (the
format used by Anthropic Agent Skills), matched with embeddings or by the
model itself. The embedding model is already available for RAG.

#### B7. Web search scrapes DuckDuckGo's HTML page

- `tools/network.py:593` scrapes `html.duckduckgo.com`. This breaks when the
  page layout changes and may violate DuckDuckGo's terms of service.

**Standard practice:** a search API, or a self-hosted SearXNG instance, which
also suits the privacy goal.

#### B8. Sync and async code are mixed

- Same as A7: `asyncio.run` inside thread pools, and `subprocess.run` called
  from async code.

**Standard practice:** make tools async (`async def` tools with
`asyncio.create_subprocess_exec` and `httpx.AsyncClient`), or keep them fully
sync and run them with `asyncio.to_thread`.

#### B9. The document retriever merges results in a basic way

- `rag/retrieval.py` puts vector results first and then adds BM25 results
  until it reaches `k`, so BM25 rankings are mostly ignored.
- BM25 splits text on whitespace only, so `"error,"` and `"error"` count as
  different words.

**Standard practice:** reciprocal rank fusion (for example LangChain's
`EnsembleRetriever`) and a proper tokenizer (lowercase, strip punctuation).

---

## Part C: Gaps Compared with Claude Code and Codex

This compares Private Agent with agentic coding tools such as Claude Code and
OpenAI Codex CLI. Each "missing" item was checked against this project's code.
The description of Claude Code and Codex is from general knowledge of those
tools, which change often, so check their current docs for exact details.

### What this project already matches

| Capability | Claude Code / Codex | This project |
|---|---|---|
| Tool loop with file read, edit and search, and shell | Yes | Yes |
| Exact-text replacement edits | Yes | Yes (`apply_workspace_patch`, with a diff preview) |
| Several tool calls run at once | Yes | Yes (`asyncio.gather` in `runtime.py`) |
| Permission modes | Yes | Yes: manual, auto, full |
| Plan mode and a to-do list | Yes | Yes: `/plan` and saved plans, with approval |
| Context compaction | Yes | Yes: `/compact` and automatic summarizing |
| MCP servers | Yes | Yes: stdio, SSE and HTTP |
| Skills | Yes | Partly (see C9) |
| Sandboxed commands | Yes | Linux only (see C4) |

### Where this project goes further

- Runs fully locally (Ollama, LM Studio, llama.cpp), and nothing leaves the
  machine by default.
- Keeps memory across sessions: summaries, learned facts, FTS5 search.
- Searches your own documents (RAG) with citations.
- Takes webcam, microphone, image and video input.
- Plans are saved with revisions, approvals and resumable leases, and coding
  requires an approved plan first.

### Missing or different features

#### High impact

**C1. No project instruction file.**
Claude Code reads `CLAUDE.md` and Codex reads `AGENTS.md` to learn a
project's rules, build commands and conventions. Nothing in this project looks
for such a file. This is probably the cheapest high-value feature to add.

**C2. No non-interactive mode.**
Claude Code has `claude -p` and Codex has `codex exec` for scripts and CI.
This project's only flag is `--verbose-startup` (`cli.py`), and it exits when
not run in an interactive terminal. There is no JSON output and no way to pass
a prompt as an argument.

**C3. No session resume and no undo.**
- Both tools can continue a past conversation (`--continue` / `--resume`).
  Here, past sessions only come back as summaries; the conversation itself
  cannot be resumed. Only plans can be resumed.
- Claude Code can rewind code and conversation to a checkpoint. Here, edits
  are copied to `.agent_snapshots/`, but there is no command to restore them.
  Git checkpoints exist only inside coding tasks.

**C4. Commands only run on Linux.**
The sandbox needs Bubblewrap and `prlimit`, so on macOS and Windows the shell,
project tests and stdio MCP are turned off. Codex and Claude Code sandbox
commands on macOS (Seatbelt) and Linux, and Codex also on Windows.

**C5. A limited shell tool.**
- The default timeout is 30 seconds (`config.py:698`), too short for many
  builds and test runs.
- Commands are run with `shell=False` from a single string, so pipes, `&&`
  and redirects probably do not work (not confirmed).
- It cannot run background processes such as dev servers or watchers, and the
  shell does not keep state between calls.

#### Medium impact

**C6. No subagents.**
Claude Code can hand searches or reviews to separate agents with their own
context. There is nothing like that here.

**C7. No hooks.**
Claude Code lets users run their own scripts before or after tool calls, for
example to format code after every edit. There is no hook system here; the
only "hook" mention in the code turns git hooks off
(`code_tasks/checkpoints.py:88`).

**C8. No fine-grained permission rules.**
Claude Code allows rules like "always allow `npm test`" and "never read
`.env`". Here there are only the three modes plus an MCP allow-list, so Auto
mode asks before every command, including harmless repeated ones.

**C9. Basic skills and no custom commands.**
- Skills have no front matter, are matched by keyword overlap and are loaded
  whole (see B6).
- Users cannot add their own slash commands as Markdown files.

**C10. Reading large files.**
`read_local_file` takes only a path, with no offset or line limit
(`tools/schemas.py:15`). Large files are cut off at the byte limit instead of
being read in pages.

**C11. No git write workflow.**
Git access is read-only (`inspect_workspace_git`), and manual git commands are
blocked inside coding tasks. The agent cannot create branches, commits or pull
requests for you, which both other tools do.

**C12. Interrupting a turn.**
Ctrl+C triggers the summarize-and-exit flow. Claude Code uses Esc to stop the
current action and keep the session, and Esc twice to rewind. This project
does not seem to have a clean way to stop one turn; there are only 3
interrupt handlers in `runtime.py`.

#### Lower impact

**C13. Fewer configuration levels.**
There is one user config file (`~/.private_agent.conf`), not user, project and
local settings files that can be committed alongside the code.

**C14. No status line or output styles.**
These are cosmetic but help users see the model, context use and permission
mode at a glance.

---

## Part D: Planning Mode

### How it works today

1. **`/plan`** applies planning mode to the **next message only**
   (`planning_mode_armed` in `agent/runtime.py`).
2. For that message the model gets just 5 tools: `read_table_schema`,
   `create_task_plan`, `inspect_task_plan`, `revise_task_plan` and
   `web_search` (`runtime.py:4892`).
3. The model must call `read_table_schema` first, or saving the plan is
   refused (`tools/planner.py:129`). Only one planner tool call is allowed per
   model round (`runtime.py:5686`).
4. The plan is saved in SQLite as `awaiting_approval`. Each plan has a goal, a
   type (research, coding or other), up to 20 steps, assumptions and
   constraints. Each step has dependencies, validation, proof, risks and edge
   cases.
5. The user opens **`/tasks`** and runs `approve <ID>`. For coding plans they
   also pick an access level and a project folder, then type
   `APPROVE <id>`. The plan's goal is then sent as the next request.
6. While the plan runs, the model reports each step with `update_todo_step`.
   The user then confirms each one with `verify <ID> <STEP> <evidence>`,
   followed by `validate` and `complete`.
7. Coding requests are refused unless an approved coding plan exists. A regex
   (`code_tasks/__init__.py:20`) decides what counts as a coding request.

### Compared with Claude Code's plan mode

The Claude Code column is from general knowledge of the tool, which changes
often, so check its current docs for exact behaviour.

| Aspect | Claude Code | This project |
|---|---|---|
| Entering | Shift+Tab; the mode stays on until you leave it | `/plan` covers one message only |
| Looking at code while planning | Read-only file reading, searching, web fetch, exploration subagents | Cannot read files at all; only a web search that returns titles and links |
| Clarifying questions | The model can ask questions before planning | One shot, with no back-and-forth |
| Plan format | Readable Markdown | JSON with a fixed schema (`tools/planner.py:28` prints it with `json.dumps`) |
| Showing the plan | Shown straight away with an approve prompt | Saved first; the user opens `/tasks` to approve it |
| Approval | One choice: approve with edits allowed, approve with manual edit approval, or keep planning | Type `APPROVE <id>`, pick an access level and folder, take a lease |
| Rejecting with feedback | "Keep planning" plus comments | Approval is declined; start again with `/plan` and a new message |
| Editing the plan yourself | Edit the plan text | Not possible; only the model can revise, and only in `/plan` mode |
| Progress tracking | A to-do list the model updates | SQLite steps, plus manual verification by the user |
| Keeping plans | Mostly within the session | Saved with revisions, history log, leases and resume |
| Audit trail | Little | Versioned plans, approvals tied to exact content, event history |

### What this project does better

- **Approval is tied to an exact plan version.** Each revision has a digest; a
  new revision cancels the old approval, and approving checks both the
  revision number and the digest (`approve_task_plan` in
  `database/memory.py`).
- **Step dependencies are checked.** Unknown dependencies, self-references and
  cycles are rejected when a plan is saved, and a step cannot start until the
  steps it depends on are done (`database/memory.py:820` and `:1631`).
- **The model's word is not treated as proof.** "Reported done" and "verified"
  are separate states, and a plan can only be completed when every step is
  verified or skipped.
- **Citations come from real searches.** Research links are added by the
  runtime from actual search results; the model cannot add its own.
- **Plans survive restarts.** Leases with a heartbeat stop two processes from
  running the same plan, and interrupted plans can be resumed.

### Gaps and issues that hurt effectiveness

#### Critical

**D1. The planner cannot see the code.**
In planning mode the model has no `read_local_file`, `search_workspace_text`
or `list_directory`. Coding plans are written without opening a single project
file, so they rest on guesses about structure, names and conventions. This is
the biggest reason plans will be poor. Read-only tools carry no risk, so the
restriction protects nothing.

**D2. The coding-plan gate gets requests wrong in both directions.**
Results of running `is_code_task_request` on sample messages:

| Request | Detected as coding? | Effect |
|---|---|---|
| "what is the best way to create a class in python?" | True | A plain question is blocked until a plan is approved |
| "How would I fix this bug in my code?" | True | Same |
| "Can you add a comment to README file" | True | A one-line edit needs a full plan |
| "write tests for parser.py" | False | Edits go ahead without a plan |
| "Update the parser to handle tabs" | False | Same |
| "rename utils.py to helpers.py" | False | Same |

So "only coding requires an approved plan" does not hold in practice. The
missed cases still go through the normal permission prompts, so this is not a
security hole, but the gate cannot be relied on.

**D3. Planning lasts for one message.**
The model cannot ask clarifying questions, the user cannot push back on the
result, and the plan cannot be refined in conversation. Anything unclear in
the request ends up as a guessed assumption in the plan.

#### High

**D4. Too many steps for small local models.**
- The model must call `read_table_schema` before it may save a plan, and it
  may make only one planner call per round, so saving one plan takes at least
  2–3 model rounds.
- The step schema is large: up to 20 steps, each with 5 lists.
- Small local models (the project's main target) often produce malformed
  nested tool arguments.
- The schema-reading step exposes database internals to the model; the tool's
  own Pydantic schema already enforces the structure.

**D5. Most of the plan is dropped when it runs.**
The approved plan is injected into the prompt as one line holding only each
step's ID, status and description (`runtime.py:5400`). The validation
criteria, risks, edge cases, dependencies, assumptions and constraints the
model had to write are left out.

**D6. Validation criteria are never checked automatically.**
Nothing runs or checks each step's `validation` / `proof` text; the user types
evidence by hand with `verify <ID> <STEP> <evidence>`. For coding plans the
criteria could be turned into test commands and checked.

**D7. Approving and finishing take too many manual steps.**
- After `/plan`: open `/tasks`, find the ID, run `approve <ID>`, pick an
  access level and folder, and type `APPROVE xxxxxxxx`.
- After the work: run `verify` for each step with your own evidence, then
  `validate`, then `complete`.
- This much effort pushes people to skip planning, which then hits the D2
  gate.

**D8. Rejecting a plan loses the work so far.**
Declining an approval tells the user to "submit the updated request with
/plan", and the earlier planning conversation is gone. There is no way to say
"change step 3", and the user cannot edit the plan directly.

#### Medium

**D9. Research during planning is shallow.**
`web_search` returns titles and links, and `fetch_webpage` is not allowed in
planning mode, so plans cite pages the model never read. Each search also
needs its own manual approval, even in Full mode (`runtime.py:5815`).

**D10. The hint for blocked coding requests is misleading.**
The message says to submit "this exact coding request" and then "repeat the
request" (`runtime.py:4960`). But `_approved_coding_task_scope` compares
against the goal the model wrote, ignoring case, so repeating the original
wording usually will not match. Starting the plan through `/tasks approve`
works because it sends the stored goal.

**D11. Only one plan can be active.**
`ACTIVE_TASK_ID` is a single global variable (see B3).

**D12. A whole plan probably runs within one time budget.**
The approved goal is sent as a single request, which appears to share the
600-second `max_task_seconds`. Long plans will likely hit the limit and need a
manual `resume` (not confirmed).

**D13. Plans are shown as raw JSON.**
Plans are shown to the user and to the model as JSON, which is harder to read
than Markdown.

**D14. Planning needs a tool-capable model.**
It stops with an error otherwise. There is no fallback, such as a text plan
that the runtime turns into structure.

---

## Part E: Coverage by Role

The project aims to be five tools at once. This part checks how well it fills
each role and what is missing for each. The ratings are a judgement from
reading the code, not measurements, and the app was not run with a model.

### Summary

| Role | Rating | Strongest part | Biggest gap |
|---|---|---|---|
| Chatbot | Good, about 70% | Local models, streaming, memory across sessions | Plain-text output, no regenerate or edit, one conversation at a time |
| Chatbot with RAG | Good on indexing, partial on answering, about 60% | Hybrid search, 12 chunking strategies, incremental indexing, citations, benchmark script | Only 2 chunks per request, no relevance cut-off, follow-up questions search on the raw text |
| Coding agent | Partial, about 50% | Sandbox, isolated copies, checkpoints, tests re-run until they pass | Planner cannot read code (D1), regex gate (D2), weak shell (C5), Linux only (C4) |
| Research agent | Weak, about 35% | Network consent, citations only from real results, offline handling | 3 search results, DuckDuckGo scraping, no multi-step research loop, no report output |
| Everyday helper | Weak, about 30% | Weather, location, date and time, webcam, microphone, transcription | No calendar, email, notes, reminders or document creation; command execution is Linux only |

### Chatbot

**What it does:** Ollama, LM Studio, llama.cpp or an online API; streaming;
optional thinking with an effort setting; chat history, session summaries and
learned facts in SQLite, with relevant past sessions brought back into the
prompt; `/compact`, `/summarize`, and prompt history with Ctrl+R search.

**Missing:**

- **E1. No Markdown rendering.** Replies are printed as raw text
  (`runtime.py:2676`), and `rich.markdown` is not used anywhere. Code blocks,
  tables and lists show as raw `**` and `|` characters.
- **E2. No regenerate or edit.** The user cannot retry the last answer or edit
  a sent message.
- **E3. One conversation at a time.** No new chat, thread switching or named
  conversations.
- **E4. No export.** Conversations cannot be saved as Markdown.
- **E5. No personas.** The system prompt is set once in config and cannot be
  switched per session.

Covered elsewhere: no non-interactive mode (C2), no session resume (C3).

### Chatbot with RAG

**What it does:** indexes PDF, text, Markdown, code, JSON, CSV, HTML and
SQLite files; follows symlinks safely; re-indexes only changed files; enforces
size limits; 12 chunking strategies including semantic; Chroma plus BM25
hybrid search; source citations; `/maintenance` reindex and reset; a benchmark
script measuring recall@k, MRR and latency.

**Missing or weak:**

- **E6. Too little context per answer.** `rag_context_results` defaults to 2
  chunks (`config.py:77`), about 2,400 characters with the default chunk size.
  Answers that need several sections of a document will be incomplete.
- **E7. No relevance cut-off.** Every request pulls 2 chunks whether or not
  they are relevant (`runtime.py:5290`). "Hello" still gets 2 random chunks,
  which wastes context and can mislead small models.
- **E8. Follow-up questions break retrieval.** The search uses the raw current
  message, so "What about the second one?" searches for exactly those words.
  The question is not rewritten using the conversation.
- **E9. The model cannot search on its own.** There is no `search_documents`
  tool, so the model cannot search again with better words or look up a
  second topic. Retrieval happens once, before the model sees the request.
- **E10. No re-ranking,** and the result merging is basic (see B9).
- **E11. Missing file formats.** No `.docx`, `.xlsx`, `.pptx`, `.epub`, or OCR
  for scanned PDFs; `.html` is read as raw markup.
- **E12. No "answer only from my documents" mode,** and no warning when the
  documents do not contain the answer.
- **E13. One collection only.** No way to pick a document set per question or
  per project, apart from the project-folder override at startup.
- **E14. Embeddings always use Ollama.** RAG does not work without Ollama,
  even with LM Studio or an online model.
- **E15. A hidden extra search for coding requests.** A second query for
  "coding standards" with fixed wording runs on coding requests
  (`runtime.py:647`), which may pull in irrelevant documents.

### Coding agent

**What it does:** reads, searches, patches, renames and deletes files with a
snapshot before each change; symbol search; read-only git inspection;
Bubblewrap-sandboxed commands; isolated project copies; git checkpoints; tests
re-run until they pass; approved plans required.

**Missing:**

- **E16. No language tooling.** No LSP, type checking or "go to definition"
  beyond the regex-based symbol search.
- **E17. No automatic test or lint after edits.** Validation only happens
  inside formal coding tasks, not after normal chat edits.

Covered elsewhere: planner cannot read code (D1), regex gate (D2), no project
instruction file (C1), weak shell (C5), Linux only (C4), no git commits or PRs
(C11), no undo (C3), no paged file reading (C10).

### Research agent

**What it does:** `web_search`, `fetch_webpage` and `download_web_file`;
consent before each network request; connection check first; private-address
(SSRF) protection; citations only from real search results; honest offline
fallback; research plans record which sources are unverified.

**Missing or weak:**

- **E18. Very few results.** `max_search_results` defaults to 3
  (`config.py:132`), scraped from DuckDuckGo's HTML page (B7). Too few and too
  fragile for real research.
- **E19. No multi-step research loop.** Nothing drives a cycle of searching,
  reading several pages, finding gaps, searching again and writing up. It
  depends on the model choosing to call tools, which small local models rarely
  do well.
- **E20. Shallow page reading.** `fetch_webpage` cuts pages at 12,000
  characters (`tools/web.py:51`) with no paging, so long articles lose their
  second half.
- **E21. Downloaded files are not read.** A file from `download_web_file` is
  not added to RAG or summarized.
- **E22. No report output.** No Markdown report with a sources list, and no
  way to save research findings.
- **E23. No source quality checks.** No dates, domain checks or
  cross-checking between sources.
- **E24. No academic or specialist sources.** No arXiv, Semantic Scholar,
  Wikipedia API, GitHub search or documentation sites.
- **E25. Consent on every request** slows multi-step research to a crawl,
  including in Full mode during planning (D9).

Covered elsewhere: planning cannot read pages (D9).

### Everyday helper

**What it does:** date and time; weather (up to 16 days ahead or historical);
location from IP; webcam, images, video, microphone recording and local
Whisper transcription; workspace file operations; learned facts about the
user.

**Missing:**

- **E26. No personal tools.** No calendar, reminders, timers, email, notes or
  contacts. MCP servers could fill this, but none are bundled or documented as
  examples.
- **E27. No document creation.** No tool to write `.docx`, `.xlsx` or PDF, and
  no conversions such as Markdown to PDF.
- **E28. No system or desktop tools.** No clipboard, opening apps or URLs,
  notifications or screenshots.
- **E29. No calculation tool.** No calculator or Python tool, and the shell
  tool is unavailable outside Linux.
- **E30. No scheduling.** No scheduled or recurring tasks.
- **E31. No voice conversation.** Recorded clips can be transcribed, but there
  is no voice mode (no wake word, no spoken replies).
- **E32. Everything is tied to the workspace folder.** File tools cannot reach
  `~/Documents` or `~/Downloads` unless the agent is started in a parent
  folder.

### Gaps that affect all roles

- **E33. One undivided agent.** Every role shares the same loop, tool set and
  system prompt. Every request carries every tool definition, which costs
  context and confuses small models. The fix is not separate modes but one
  runtime that sends only the tools each request needs (see Part F).
- **E34. Mixed output.** User-facing text and diagnostic messages
  (`[RAG State]`, `[Processing]`) are printed together, which clutters the
  chat (see A6).

---

## Part F: Plan for a Single Combined Runtime with Live Status

Goal: the chatbot, RAG chatbot, coding agent, research agent and helper become
**one runtime**, not separate modes the user switches between. Any request
can use any mix of these abilities in the same conversation, for example
"research the latest pytest-asyncio changes, then update our tests to match".
While the agent works, a **status overlay** shows which tools are running, and
agentic tasks get a **timeline with an estimated finish time**.

This replaces the "separate role profiles" idea in E33 and step 8.1.

### F1. Current state

- **The parts are already in one process,** but they are joined by flags and
  special cases inside `_run_agent_cli_session`: `planning_only`,
  `code_task_requested`, `include_private_context`, `active_code_workspace`,
  and others. Each combination has its own copy of the loop (B1).
- **Every request gets every tool,** except in planning mode, which gets 5
  (E33).
- **Abilities cannot be combined in one request.** Planning cannot read code
  (D1), research cannot write code in the same task, and coding needs a
  separate approved plan first (D2).
- **Status display is minimal.** There are 5 `console.status(...)` spinners
  in `runtime.py`, and `_display_tool_request` prints one line per tool call.
  There is no `rich.live`, no `rich.progress`, and no `prompt_toolkit` bottom
  toolbar, so there is no persistent view of what is running.
- **There is no time estimate.** The only timing shown is elapsed time after
  slow steps, and the remaining `max_task_seconds` budget is not shown.

### F2. Target architecture

```
                         ┌──────────────────────────────┐
  user message ────────▶ │ Turn router                  │
                         │ commands → rules → model     │
                         └──────────────┬───────────────┘
                                        │ chosen capability packs
                                        ▼
┌─────────────────┐      ┌──────────────────────────────┐      ┌─────────────────┐
│ Context manager │◀────▶│ Agent loop (one, shared)     │◀────▶│ PermissionPolicy│
│ history, memory,│      │ model call → tool calls →    │      │ (one, shared)   │
│ RAG, files,     │      │ results → repeat             │      └─────────────────┘
│ research notes  │      └──────┬───────────────┬───────┘
└─────────────────┘             │               │
                                ▼               ▼
                  ┌──────────────────┐   ┌──────────────────────┐
                  │ Task orchestrator│   │ Tool executor        │
                  │ plan → approve → │   │ (uses ToolContext)   │
                  │ steps → verify   │   └──────────┬───────────┘
                  └────────┬─────────┘              │
                           │      events            │
                           ▼                        ▼
                  ┌──────────────────────────────────────────┐
                  │ Event bus                                │
                  └───┬──────────┬───────────┬───────────┬───┘
                      ▼          ▼           ▼           ▼
                Status overlay  Run log  Session stats  Hooks / JSON output
```

#### Capability packs

Each ability becomes a **capability pack**: a bundle of tools, a system-prompt
section, limits, and optional checks. Packs are added together for a turn
instead of being separate modes.

| Pack | Tools | Prompt section | Limits and checks |
|---|---|---|---|
| Core (always on) | Date and time, read-only file reading and search, `search_documents`, `ask_user`, `enable_capability` | Base rules, privacy, permission mode | — |
| Knowledge (RAG) | `search_documents`, collection selection | Cite sources; say when documents do not contain the answer | Relevance cut-off (8.3) |
| Research | `web_search`, `fetch_webpage`, `download_web_file`, report writer | Search, read, check gaps, cite only real sources | Consent once per research task; source limits |
| Coding | Edit, patch, rename, delete, shell, tests, git, checkpoints | Project instructions (`AGENTS.md`), test after edits | Approved plan required for edits; tests as verification |
| Helper | Weather, location, media, MCP personal tools, documents | Device and privacy rules | Per-use approval for media |
| Planning | `create_task_plan`, `revise_task_plan`, `update_todo_step` | How to write a plan | Plan approval |
| MCP | Configured MCP tools, grouped by server | Server descriptions | MCP allow-list |

Only the packs a turn needs are sent to the model. Plain chat sends only the
Core pack, which saves context and helps small models (fixes E33 without
splitting the runtime).

#### Turn router

The router picks the packs for each turn, in this order:

1. **Explicit commands** win: `/code`, `/research`, `/plan`, `/ask` (chat
   only). These become shortcuts that force packs, not separate runtimes.
2. **Rules:** `@file` or `/add` attachments add Coding read tools; a URL adds
   Research; an approved active plan adds the packs its steps need; an image
   adds Helper media tools.
3. **Model classification** for everything else: one short, cheap call (or a
   smaller configured model) returns a list of packs and a complexity level
   (`simple`, `multi_step`, `long`). It must return valid JSON, and falls back
   to Core plus Knowledge if it fails.
4. **Escalation during a turn:** the model can call `enable_capability` to add
   a pack it needs, for example when chat turns into "now fix it". The
   `PermissionPolicy` decides whether that needs the user's approval
   (switching to Coding or Research always asks in Manual and Auto modes).

The chosen packs are shown in the status overlay, so routing is never hidden.

#### Task orchestrator

When the router reports `multi_step` or `long`, or the user asks for a plan,
the orchestrator runs the task as a plan instead of one long turn:

1. **Plan** with read-only tools from every chosen pack (fixes D1). The model
   may ask clarifying questions (D3).
2. **Show the plan** as Markdown with one approval choice (D7, D13).
3. **Run each step** with its own packs, context and time budget (D12). A
   research step can feed a coding step.
4. **Verify** each step automatically where possible: tests for coding,
   sources checked for research (D6).
5. **Report** a summary, changed files, sources and anything left unverified.

Simple requests skip the orchestrator and run as one turn, as today. The
existing SQLite plan storage, revisions, digests and leases stay as they are.

#### Context manager

One component builds the prompt from all sources with a token budget per
source, so no single source crowds out the others:

| Source | Example share of the budget |
|---|---|
| System prompt and pack sections | 15% |
| Recent conversation | 35% |
| Active plan and step | 10% |
| RAG chunks | 15% |
| Memory (episodes, learned facts) | 10% |
| Attached files and research notes | 15% |

Unused shares go to recent conversation. The shares are configurable, and the
overlay shows total context use.

#### Event bus

Every component emits typed events instead of printing directly:

| Event | Fields |
|---|---|
| `TurnStarted` / `TurnFinished` | Turn ID, packs, complexity, duration, tokens |
| `ModelCallStarted` / `ModelCallFinished` | Model, input tokens, output tokens, tokens per second |
| `ToolStarted` | Call ID, tool name, pack, short redacted argument summary, start time |
| `ToolProgress` | Call ID, message (for example "downloaded 2.1 MB", "12 tests passed") |
| `ToolFinished` | Call ID, status (ok, error, denied, timed out), duration, output size |
| `ApprovalRequested` / `ApprovalResolved` | Call ID, kind, answer |
| `PlanStepChanged` | Task ID, step ID, old and new status |
| `EstimateUpdated` | Task ID, per-step and total estimate ranges |
| `BudgetWarning` | Time or token budget nearly used |

The status overlay, run log (A6), session statistics, hooks (6.11) and the
JSON output of non-interactive mode (6.8) all read from this bus. This also
removes most of the scattered `console.print` calls (A6, E34).

### F3. Status overlay for tools in use

A live panel pinned to the bottom of the terminal while the agent works.

```
 You: research the pytest-asyncio 1.x changes and update our async tests
 ─────────────────────────────────────────────────────────────────────────
 Agent: Found the 1.0 changelog. The main change is that the event_loop
 fixture was removed, so ...                          (streamed reply text)

╭─ Working · Research + Coding · step 2/4 ───────────────── 01:42 ───────╮
│ ⠹ fetch_webpage      docs.pytest.org/…/changelog.html          0:04     │
│ ⠹ search_workspace   "event_loop" in tests/                    0:01     │
│ ✓ web_search         "pytest-asyncio 1.0 changes"              2.3s     │
│ ✓ read_local_file    tests/conftest.py                         0.1s     │
│ ─────────────────────────────────────────────────────────────────────── │
│ Model qwen3:8b · 14 tok/s · context 7.9k / 12k (66%) · mode Auto        │
╰ Esc stop turn · Ctrl+T timeline · Ctrl+O tool details ──────────────────╯
```

**What it shows:**

- **Phase:** Thinking, Calling tools, Waiting for approval, Verifying, or
  Writing reply.
- **Active packs and current plan step,** if any.
- **Running tools** with a spinner, a short redacted argument summary and
  elapsed time. Tools running at the same time (the existing `asyncio.gather`)
  each get their own line.
- **The last few finished tools** with a status mark (✓ ok, ✗ error, ⊘ denied,
  ⏱ timed out) and duration. Older ones fold into "+12 earlier".
- **Footer:** model, generation speed (already measured by
  `_display_generation_rate`), context use, permission mode, turn elapsed
  time.
- **Key hints:** Esc to stop the turn (6.4), Ctrl+T for the timeline, Ctrl+O
  for full tool details.

**Behaviour:**

- **During a turn:** a `rich.live.Live` region. Streamed reply text renders
  above the panel (with Markdown, 8.2) and the panel stays at the bottom.
- **Between turns:** a one-line `prompt_toolkit` bottom toolbar with model,
  context use, mode and the active plan's progress.
- **Approval prompts** pause the live region, show the prompt, then resume, so
  input and live updates never fight over the terminal.
- **Errors** stay visible in the panel until the turn ends, then go into the
  normal output.
- **Plain mode** (setting, `NO_COLOR`, or a terminal that cannot redraw) falls
  back to one line per event, as today.
- **Non-interactive mode** writes the same events as JSON lines (6.8).
- **Privacy:** argument summaries use the existing redaction
  (`_redact_tool_arguments`).

### F4. Estimated timeline overlay for agentic tasks

For planned tasks, Ctrl+T expands the overlay into a timeline. It opens
automatically when a plan starts.

```
╭─ Task: Update async tests for pytest-asyncio 1.x ────── 04:12 elapsed ─╮
│                                                                         │
│ ✓ 1  Research changelog and migration guide      1:38   (est. 1–3 min)  │
│ ▶ 2  Find affected tests and fixtures             2:34   (est. 2–4 min)  │
│ ○ 3  Update fixtures and tests                    —      (est. 4–8 min)  │
│ ○ 4  Run tests and fix failures                   —      (est. 3–10 min) │
│                                                                         │
│ ████████████░░░░░░░░░░░░░░░░░░  ~35%                                    │
│ Estimated finish: 10–20 min from now (around 14:35–14:45)               │
│ Time budget: 25 min left of 30 · confidence: medium                     │
╰ Ctrl+T collapse ────────────────────────────────────────────────────────╯
```

#### How the estimate is calculated

LLMs are poor at estimating their own task time, so the model's guess is only
one input, and the estimate improves as the task runs.

1. **Starting estimate per step,** from three sources, weighted by how much
   data exists:
   - **History:** the median duration of similar past steps, grouped by pack,
     main tools and model, from a new step-timing table (below). This becomes
     the main source once there are about 5 matching past steps.
   - **Model speed:** expected output tokens × the measured tokens per second
     for the current model, plus the average duration of the tools the step
     is likely to use.
   - **Model's guess:** an optional `estimated_minutes` field on each plan
     step. It gets low weight, and is used mainly when there is no history.
2. **Recalibration while running:** after each step, compare actual and
   estimated time, and scale the remaining steps by a smoothed ratio. Running
   10 minutes slow on step 1 makes later estimates longer.
3. **Ranges, not single numbers:** always show a range (for example 4–8 min)
   and a confidence label (low, medium, high) based on how much history was
   used. With no data at all, show "estimating…" until the first step ends.
4. **Waiting time is separate:** time spent waiting for the user's approval is
   left out of the estimate and shown as "waiting for you".
5. **Budget warnings:** if the estimated finish goes past the remaining
   `max_task_seconds` budget, the overlay warns early and offers to extend the
   budget, pause after the current step, or simplify the plan.

#### Data needed

A new SQLite table, `agent_step_timings`:

| Column | Purpose |
|---|---|
| `task_id`, `step_id`, `revision` | Which step |
| `packs`, `tools_used` | What kind of work it was |
| `model`, `provider` | Speed differs by model |
| `estimated_low_s`, `estimated_high_s` | What was predicted |
| `started_at`, `finished_at`, `waiting_s` | What actually happened |
| `input_tokens`, `output_tokens`, `tool_calls` | Size of the work |
| `outcome` | Done, failed, skipped |

The existing `agent_task_events` table already stores step status changes with
timestamps, so past tasks can partly fill this table when it is created.

#### Non-plan agentic turns

Simple turns that still call several tools show a lighter view: tool rounds
so far, elapsed time and the remaining time budget, without a per-step
estimate.

### F5. Risks and decisions

- **Router mistakes:** a wrong pack choice can hide a needed tool. Mitigations:
  the Core pack always has read-only tools, `enable_capability` allows
  escalation, the chosen packs are shown, and commands can force packs.
- **Extra model call:** classification adds latency. Use rules first, cache
  classifications for similar follow-up messages, and allow a smaller
  classifier model.
- **Terminal handling:** `rich.live` and `prompt_toolkit` must not draw at the
  same time. Keep a strict rule: `Live` only during turns, toolbar only while
  waiting for input, and pause `Live` for every prompt. A full-screen TUI (for
  example Textual) is a possible later step, not needed now.
- **Estimate trust:** wrong estimates erode trust quickly. Always show ranges
  and confidence, and let the user turn the timeline off.
- **Safety stays the same:** combining abilities must not weaken the existing
  rules. Coding edits still need an approved plan, network calls still need
  consent, and media still needs per-use approval. All of these go through the
  single `PermissionPolicy`.

---

## Part G: User Interface

The UI is a prompt-driven terminal app built with `rich` and `prompt_toolkit`.
The startup screens below come from a real run in a pseudo-terminal with a
temporary home folder. No model was reachable, so the conversation screens
are taken from the code.

### What it looks like

**Startup (captured):**

```
Private Agent
No local runtimes found to proceed with local inference. Online configuration is available.
RAG documents directory (Enter to use configured path '/…/.private_agent/rag', or enter
a project-specific path):
[Warning] Provided KB path '/…/.private_agent/rag' is not a valid directory. Skipping
knowledge base initialization.
⠋ Indexing local knowledge base...
Enter workspace root path for security sandbox (Press Enter for current dir):
[Agent Permission Mode]
  1. Manual — ask before every tool action
  2. Auto — read/search freely; ask before edits and command execution
  3. Full — approve all tool calls for this session
Choose permission mode [1/2/3] (default: auto): 2

Model Provider
  1. Online — enter details manually
Choose provider [1-1] (Default: 1):
OpenAI-compatible API base URL :
Online mode will send requests to api.openai.com. Continue? [y/N]:

Model Provider
  1. Online — enter details manually
Choose provider [1-1] (Default: 1):
```

**During a conversation (from the code):**

```
--- Local Ready! Type 'exit' or 'switch' ---
User: explain tests/conftest.py                 ← bold blue, single line
[Processing] Processing model response...
[RAG State] Retrieved 2 document/episodic chunks.
[Tool Request] read_local_file {'file_path': 'tests/conftest.py'}
╭─ Tool output: read_local_file ─────────────╮ ← blue panel, first 500 chars only
│ import pathlib ...                          │
╰─────────────────────────────────────────────╯
AI (qwen3:8b): This file sets up **pytest** fixtures ...   ← raw text, Markdown not rendered
```

At exit, a **Session Statistics** table is printed, which looks good.

**In numbers:**

| Element | Count in code |
|---|---|
| `console.input` prompts | 59 |
| `[y/N]` confirmations | 12 |
| Typed confirmations | `APPROVE <id>`, `DELETE <id>`, `TAKEOVER <id>`, `RESET` |
| Panels / tables | 10 / 9 |
| `Markdown`, `Syntax`, `Live`, `Progress`, `Rule` | 0 |
| Separate `Console()` objects | 11 |
| Different `[Label]` message prefixes | about 57 |
| Colour tags | yellow 166, cyan 75, red 60, green 35 |

### Strengths worth keeping

- Command and `@file` completion while typing, with "directory" and
  "workspace file" labels, and Ctrl+R history search.
- Clean tables for `/status`, the startup Ready summary and Session
  Statistics.
- Typed confirmations for destructive actions (`DELETE <id>`, `RESET`).
- Redacted tool arguments, and plain-text fallback when not attached to a
  terminal.

### Bugs seen in the real run

**G1. No way out of the online-provider loop.**
With no local model, pressing Enter at `Continue? [y/N]` (default No) returns
to a provider menu whose only option is the same online provider. Only Ctrl+C
exits.

**G2. A warning about its own default folder.**
On first run it warns that its own default RAG folder `~/.private_agent/rag`
is not a valid directory, instead of creating it or skipping quietly.

**G3. A spinner after skipping.**
"Indexing local knowledge base..." appears right after it says indexing was
skipped.

**G4. Questions in an odd order.**
It asks for the RAG folder before the workspace, though the project-local
`rag` folder is found relative to the workspace.

### Essentials missing

**G5. No Markdown rendering or code highlighting.**
Replies are streamed as plain text with markup turned off
(`runtime.py:2676`), so code blocks, tables, lists and bold show as raw `**`,
`|` and backticks. This is the biggest visible gap (same as E1).

**G6. No coloured diffs.**
Edit previews (`preview_workspace_patch`) go through the same `Text()` panel
as all tool output, so it is hard to see what an edit changes before
approving it.

**G7. Too many startup questions.**
There are 4–7 questions before the first message (RAG folder, workspace,
mode, provider, URL, key, confirm), with no "remember my choices" and no
command-line flags.

**G8. No live status while working** (see F3).

**G9. No way to stop one turn** (see C12).

**G10. No multi-line input.**
There are no key bindings, so there is no Alt+Enter or Shift+Enter for a new
line.

**G11. No way to expand tool output.**
Previews stop at 500 characters (`TOOL_RESULT_PREVIEW_CHARS`,
`runtime.py:196`), with no way to see the rest of what the model saw.

**G12. Basic approval prompts.**
Prompts are plain `[y/N]` with no "always allow" option (C8), and tool
arguments are shown as a Python dict.

**G13. No arrow-key menus.**
Menus are numbered or typed, and `/tasks` opens a sub-prompt where commands
such as `approve <ID>` are typed and IDs copied by hand.

**G14. Replies and diagnostics are mixed** (see E34).

### Cosmetics missing

| ID | Missing | What it would look like |
|---|---|---|
| G15 | **Consistent message style.** About 57 different `[Label]` prefixes in mixed styles; yellow used 166 times for both warnings and normal information | A few levels (info, success, warning, error), each with one colour and one symbol |
| G16 | **One shared console and theme.** 11 separate `Console()` objects and no theme | One console with a named theme (`warning`, `tool`, `user`, …) and a light-terminal option |
| G17 | **An informative prompt.** Just `User:` in bold blue | `qwen3:8b · auto · 42% ›`, or the bottom toolbar (F3) |
| G18 | **Separation between turns.** `AI (model):` header with no divider | A thin `Rule` between turns, aligned headers |
| G19 | **A welcome screen.** Only "Private Agent" | Version, model, workspace and a `/help` hint |
| G20 | **Readable tool requests.** Raw Python `repr` | `read_local_file · tests/conftest.py` |
| G21 | **Per-turn details.** Speed, tokens and time only appear in the exit table | A dimmed line after each reply: `1.8k tokens · 14 tok/s · 6.2s` |
| G22 | **Status symbols.** Results are marked only by panel colour | ✓ ✗ ⊘ ⏱ markers |
| G23 | **Grouped `/help`.** One long list | Groups (Chat, Files, Plans, Memory, Model) in a table |
| G24 | **Messages that fit narrow terminals.** Long warnings wrap mid-sentence | Short messages with details on request |

---

## Part H: Platform Support (Linux, WSL, macOS and Termux; no native Windows)

Goal: support **POSIX systems only**, namely Linux (including WSL2), macOS and,
as a partial environment, Termux on Android. Drop native Windows, so it runs
in WSL without any setup trouble.

### H1. What was checked

- **Native-Windows code is almost nonexistent.** The source has no `winreg`,
  `msvcrt`, `os.name == "nt"` or `WindowsPath` code. The Windows traces are:
  - tests that simulate Windows or macOS (`tests/unit/test_providers.py:282`
    patches `sys.platform = "win32"`, `tests/unit/test_code_tasks.py:297–327`,
    `tests/integration/test_agent_runtime.py:210`);
  - README wording about "supported desktop platforms" (lines 32, 139, 350);
  - `isolation_unavailable_reason()` (`code_tasks/__init__.py:77`), which
    already rejects everything except Linux.

  So removing Windows is mostly removing tests and wording, plus a clear
  startup message.
- **The code already relies on POSIX features:**
  - the secure write path uses `O_NOFOLLOW`, `O_DIRECTORY` and `dir_fd`
    (`tools/filesystem.py:216–231`);
  - hard links via `os.link` (`tools/filesystem.py:416`,
    `database/memory.py:2890` and `:2955`);
  - Bubblewrap and `prlimit` for the sandbox.
- **On this machine (WSL2, Fedora 44):**
  - mirrored networking with `hostAddressLoopback=true` is set in
    `.wslconfig`, so `127.0.0.1:11434` can reach a Windows-side Ollama;
  - Ollama is installed on the Windows side but was not answering
    (connection refused), so it was probably not running;
  - unprivileged user namespaces work (`unshare --user` succeeded), and
    `prlimit` is installed, but **`bwrap` is not installed**. That one missing
    package is why shell commands, project tests and stdio MCP are disabled,
    and why 1 of the failing tests fails.

### H2. Target support levels

| Platform | Level | Sandbox | Local models | RAG | Media |
|---|---|---|---|---|---|
| Linux (x86_64, arm64) | Full | Bubblewrap + `prlimit` | Ollama, LM Studio, llama.cpp | Full | Full |
| WSL2 | Full | Bubblewrap + `prlimit` inside WSL | Ollama in WSL, or Windows-side Ollama via mirrored networking | Full | Limited: no webcam or microphone by default |
| macOS (arm64, x86_64) | Full | Seatbelt (`sandbox-exec`) + `setrlimit` (step 6.14) | Ollama, LM Studio, llama.cpp | Full | Full (needs camera and microphone permission) |
| Termux (Android) | Partial | None available; commands off by default | Ollama or llama.cpp running in Termux, or an online provider | BM25 keyword search only, unless the vector store installs | Off |
| Native Windows | Not supported | — | — | — | — |

On native Windows, startup prints a short message pointing to WSL and exits.

### H3. Issues to fix per platform

#### All platforms

**H3a. Hard dependency on heavy packages.**
`runtime.py` imports `ollama`, `langchain_ollama`, the RAG stack (Chroma) and
the media module at the top (`runtime.py:24`, `:32`, `:120`, `:189`). If any
of them fails to install, as Chroma's `onnxruntime` dependency is likely to on
Termux, the whole app fails to start. Optional parts should be imported only
when used, with a clear message when missing.

**H3b. No platform detection or doctor command.**
Nothing reports which features are available on the current system or why
some are missing. The Ready summary shows the OS and whether isolation works,
but not how to fix it.

**H3c. Platform claims are not tested.**
There is no CI matrix, so macOS and arm64 support are untested.

#### WSL

**H3d. Missing Bubblewrap with no hint.**
The message says to install Bubblewrap and `prlimit` but not the command for
the current distribution (`sudo dnf install bubblewrap` on Fedora,
`sudo apt install bubblewrap` on Ubuntu or Debian).

**H3e. Reaching Ollama on the Windows side.**
The default `http://localhost:11434` works only with mirrored networking (as
on this machine) or with Ollama installed inside WSL. With NAT networking (the
WSL default), Windows-side Ollama is at the host's gateway IP and needs
`OLLAMA_HOST=0.0.0.0` on Windows. The loopback-only check in
`agent/local_runtimes.py:28` would also reject the gateway address. None of
this is documented.

**H3f. Working on `/mnt/c` is slow and unreliable.**
Projects on the Windows drive (`/mnt/c/...`) go through a translation layer.
File access is much slower, hard links and `O_NOFOLLOW` behave differently,
and file permissions do not map cleanly. This can break the secure write path
and snapshots. Nothing warns about it.

**H3g. No webcam or microphone by default.**
WSL has no direct access to cameras, and microphone access depends on WSLg's
audio support. The media tools should report this clearly rather than fail.

#### macOS

**H3h. No sandbox.**
Shell, project tests and stdio MCP are disabled because Bubblewrap is
Linux-only (C4). This needs a Seatbelt backend and `resource.setrlimit`
instead of `prlimit` (step 6.14).

**H3i. Camera and microphone permissions.**
The terminal app must be granted access in System Settings, and failures
should say so.

#### Termux (partial environment)

**H3j. No sandbox available.**
Android does not allow unprivileged user namespaces, so Bubblewrap cannot
run, and `proot` is not a security boundary. Under the current fail-closed
rule, commands stay disabled. The plan offers an explicit, off-by-default
choice (see H4); whether to allow it is a decision for the project owner.

**H3k. Packages that may not install.**
Chroma needs `onnxruntime`, which has no Android builds, and some packages
need to be compiled in Termux. RAG needs a fallback that works without them.

**H3l. Shared storage limits.**
`/sdcard` (shared Android storage) does not support hard links or symlinks
the same way, which can break snapshots and the secure write path. The
workspace should live in Termux's home folder.

**H3m. Small screens and limited input.**
Wide tables, long prompts and Ctrl or Esc shortcuts are hard to use on a
phone keyboard. The UI needs a compact layout and keys that exist on the
Termux extra-keys row.

### H4. Approach

1. **POSIX only.** Declare it in `pyproject.toml` (`Operating System ::
   POSIX` classifiers), and add a startup check that exits on native Windows
   with a message pointing to WSL. Remove the tests that simulate Windows.
2. **A platform module.** Add `platform_support.py`, which detects Linux,
   WSL (from `/proc/sys/fs/binfmt_misc/WSLInterop` or a `microsoft` kernel
   name), macOS and Termux (`PREFIX` containing `com.termux`). It reports which
   features are available and why. Everything else asks this module instead
   of checking `platform.system()` itself.
3. **A sandbox interface with backends.** Bubblewrap (Linux and WSL),
   Seatbelt (macOS) and None. With the None backend, commands stay disabled
   unless the user turns on a clearly labelled "unsandboxed, approve every
   command" option for that session. That option must never apply in Full
   mode without per-command approval.
4. **Optional extras.** Split dependencies into a small core plus extras:
   `rag` (Chroma and embeddings), `media`, and `all`. The core must install
   on Termux. Without the `rag` extra, document search falls back to BM25
   only (`rank-bm25` is pure Python) or to SQLite FTS5, which is already used
   for memory.
5. **A `private-agent doctor` command.** It lists every feature as available
   or missing, with the reason and the exact fix command for the detected
   system (for example `sudo dnf install bubblewrap`, `pkg install ollama`,
   or "move the project out of `/mnt/c`").
6. **Setup docs per platform:** WSL (Bubblewrap, Ollama inside WSL vs.
   Windows-side with mirrored networking, keeping projects on the Linux
   filesystem), macOS (Homebrew, permissions) and Termux (packages, storage
   location, limits).
7. **A CI matrix** on Ubuntu x86_64, Ubuntu arm64 and macOS, plus a
   core-only install test that mimics Termux (no `rag` or `media` extras).

---

## Part I: Local Provider Independence (Not Ollama Only)

Goal: the program works with **any installed local provider**, not just
Ollama. If several are installed, it asks which one to use at session start.
When a session or task is resumed and its provider is not available, it falls
back to the same choice prompt.

### I1. How it works today

**Already provider-neutral:**

- **Discovery** (`agent/local_runtimes.py:121`) probes Ollama
  (`/api/tags`) plus up to 7 configured OpenAI-compatible endpoints
  (`/models`) in parallel, with a 1.5-second timeout. The defaults add LM
  Studio (`127.0.0.1:1234/v1`) and llama.cpp (`127.0.0.1:8080/v1`).
- **The provider menu** (`runtime.py:3990`) lists every reachable local
  runtime with its model count, plus the online options, and asks the user to
  choose. So with several running servers, a prompt already appears.
- **Chat** works with OpenAI-compatible local servers through `ChatOpenAI`
  (`_make_local_openai_compatible_chat_model`, `runtime.py:2512`).
- **Session summaries** use whichever chat model the session uses.

**Still tied to Ollama:**

| # | Where | What depends on Ollama | Effect without Ollama |
|---|---|---|---|
| I1a | `rag/indexing.py:15`, `:346` | Embeddings always use `OllamaEmbeddings` at `OLLAMA_BASE_URL` | Document search (RAG) and `semantic` chunking do not work at all (same as E14) |
| I1b | `runtime.py:4149–4160` | Model capability detection (tools, vision, thinking, context window) only runs for Ollama; other runtimes get "unknown" for everything | Vision and thinking are never used; the context window falls back to the configured default; tool support is assumed, not checked |
| I1c | `runtime.py:4161` | Switching to a tool-capable model when the chosen one cannot call tools is Ollama-only | A non-tool model on LM Studio fails at tool calls instead of being redirected |
| I1d | `runtime.py:2416–2428` | `inspect_model_capabilities` uses the global `OLLAMA_BASE_URL`, not the selected runtime's URL | A second Ollama server, or one on another port, gets the wrong capability data |
| I1e | `runtime.py:24`, `:32`, `hardware/status.py:6`, `rag/indexing.py:15` | `ollama` and `langchain_ollama` are imported at the top of modules and are required dependencies | The packages must be installed even when Ollama is never used |
| I1f | `hardware/status.py`, `/hardware-status` | Hardware placement and acceleration options are Ollama-only | Other runtimes show "no standard API" (acceptable, but undocumented per runtime) |
| I1g | `runtime.py:3097` | `fetch_local_chat_models()` talks only to Ollama | Any caller of this helper sees only Ollama models |
| I1h | `scripts/benchmark_rag.py` | The RAG benchmark requires Ollama | The benchmark cannot run with other providers |
| I1i | `config.py:37` | One `ollama_base_url`; Ollama is always probed first and listed first | Cannot list two Ollama servers; Ollama always appears as option 1 |

**Discovery and selection gaps:**

- **I1j. Only running servers are found.** Discovery detects servers that
  answer on their port. An installed provider that is not running (for example
  Ollama installed but `ollama serve` not started) is invisible, and nothing
  suggests starting it.
- **I1k. Few providers are known.** Only Ollama, LM Studio and llama.cpp have
  defaults. Other common local servers that speak the OpenAI-compatible API
  (for example vLLM, Jan, KoboldCpp, GPT4All, LocalAI, llamafile, MLX-LM and
  text-generation-webui) need manual configuration. Their default ports must
  be checked against each project's docs; several share port 8080, so
  probing a port does not identify which server answered.
- **I1l. Embedding models are listed as chat models** on OpenAI-compatible
  servers: the filter that hides embedding models (`local_runtimes.py:87`)
  only runs for Ollama.
- **I1m. Loopback only.** Only `localhost` and `127.0.0.0/8` addresses are
  accepted (`local_runtimes.py:28`). A provider on the WSL host gateway or on
  another machine in the home network cannot be used as "local" (see H3e).
- **I1n. Always prompts, even with one choice,** and there is no
  `--provider` flag or remembered last choice (see G7).
- **I1o. A provider that stops mid-session** is not handled specially; the
  model call fails and the error is reported, but there is no offer to switch
  to another available provider.

**Resume gaps:**

- **I1p. Nothing records which provider was used.** `chat_history` has no
  provider or model column (`database/memory.py:234`), and `agent_tasks` has
  none either. Conversations cannot be resumed at all yet (C3), and task plans
  resumed through `/tasks` simply use whatever provider the current session
  has, with no check that it suits the plan (for example vision or tool
  support).

### I2. Target behaviour

#### Provider registry

Each provider type becomes a **provider adapter** with one interface:

| Method | Purpose |
|---|---|
| `detect_installed()` | Is the provider installed? (binary on `PATH`, known app folder, or configured) |
| `detect_running()` | Does its server answer, and which models does it list? |
| `start()` (optional) | Start the server, only after the user agrees (for example `ollama serve`) |
| `list_models()` | Chat models, with embedding models separated |
| `capabilities(model)` | Tools, vision, thinking, context window; "unknown" when the provider cannot say |
| `chat_model(model, options)` | Build the LangChain chat model |
| `embeddings(model)` (optional) | Build an embeddings client for RAG |
| `hardware_status()` (optional) | GPU and CPU placement, where the provider exposes it |

**Adapters to ship:**

- **Ollama:** native API, all methods.
- **Generic OpenAI-compatible:** chat, `/models` listing, embeddings through
  `/v1/embeddings` when the server supports it.
- **Presets on top of the generic adapter** for LM Studio, llama.cpp
  (`llama-server`), vLLM, Jan, KoboldCpp, GPT4All, LocalAI, llamafile and
  MLX-LM. Each preset has a name, default port, install check and start hint.
  Ports and paths must be verified against each project's docs before
  shipping.

Capabilities for OpenAI-compatible servers come from, in order: what the
server reports (some list a context length), a small built-in table of known
model families, a one-time test call (for example a tiny tool call) cached per
model, and finally user overrides in config.

#### Session start

```
Local providers
  1. Ollama        running   · 6 chat models          (last used)
  2. LM Studio     running   · 2 chat models
  3. llama.cpp     installed · not running            [s] start it
Online
  4. Online — configured in config
Choose provider [1-4] (Default: 1):
```

Rules:

1. **Detect** every known provider: installed, running, or both.
2. **Two or more usable local providers:** always ask, with the last-used
   provider as the default.
3. **One usable local provider:** use it and say so in one line, unless
   `providers.always_ask` is on.
4. **None running but some installed:** offer to start one (with consent), or
   continue online, or show the `doctor` fix.
5. **None installed:** offer online, or show how to install a provider. Never
   loop without an exit (fixes G1).
6. **Flags win:** `--provider lmstudio --model qwen3-8b` skips the prompt;
   if the named provider is missing, fall back to the prompt with a message.
7. **Local first:** online options are listed after local ones and are never
   chosen automatically.

#### Resume with fallback

Each session and each task stores a **provider record**: adapter type, display
name, base URL, model, and the capabilities the work needs (for example tools,
vision).

```
Resuming session "Refactor parser" (last used LM Studio · qwen3-8b)
LM Studio is not running.
  1. Ollama        running · qwen3:8b available
  2. llama.cpp     running · 1 chat model
  3. Start LM Studio
  4. Online — configured in config   (history will be sent online)
Choose provider [1-4]:
```

Rules:

1. **Same provider and model available:** resume without a prompt, and show
   one line saying which provider is used.
2. **Provider available, model missing:** show that provider's model list,
   with the closest name match first.
3. **Provider not available:** show the provider prompt above, with the
   reason, and an option to start it if it is installed.
4. **Capability check:** if the resumed work needs tools or vision and the new
   model lacks them, warn and suggest a model that has them.
5. **Context window check:** if the new model's context is smaller than the
   history, compact the history before continuing (using the existing
   compaction) and say so.
6. **Privacy:** never switch a resumed local session to an online provider
   without an explicit choice, and warn that the conversation history will be
   sent to it.
7. **Record the change:** the session's provider record is updated, and a
   task records the switch as an event in `agent_task_events`.
8. **Mid-session loss:** if the provider stops answering during a session,
   offer the same fallback prompt instead of only reporting an error (I1o).

#### Embeddings without Ollama

- Embeddings get their own setting (`rag.embedding_provider`,
  `rag.embedding_model`), separate from the chat provider.
- Any adapter with an `embeddings()` method can serve it (Ollama, LM Studio,
  llama.cpp with embeddings on, and others).
- The RAG index records which provider and model built it. If they change,
  the index is rebuilt, because vectors from different models cannot be mixed.
- With no embedding provider at all, RAG falls back to keyword search (BM25 or
  SQLite FTS5), as in step 11.6.

---

## Part J: Fix Roadmap

Each step is small enough for its own branch and PR. Steps are ordered so that
each one makes the next one safer: first get a reliable test and lint safety
net, then fix real bugs, then remove global state, then restructure, then
improve individual features, then add the features from Part C (Phase 6),
improve planning mode from Part D (Phase 7), fill the role gaps from Part E
(Phase 8), combine everything into one runtime with live status from
Part F (Phase 9), improve the UI from Part G (Phase 10), support Linux,
WSL, macOS and Termux from Part H (Phase 11), and remove the dependence on
Ollama from Part I (Phase 12). Most of Phases 6–12 build on the shared agent
loop,
`PermissionPolicy` and `ToolContext` created in Phases 3 and 4; the
"Depends on" column shows which steps can be done earlier.

Recommended order of work, following the dependencies:

1. Phase 1 (safety net), then steps 11.1–11.4 (POSIX only, platform
   detection, `doctor`, WSL fixes), so the app runs cleanly in WSL early.
2. Phase 2 (quick fixes), then steps 10.1–10.3 (startup bugs, Markdown
   replies, coloured diffs).
3. Steps 7.1–7.3 and 8.3 (small planning and RAG fixes with no dependencies).
4. Phases 3–4 (remove global state, restructure the core), then step 11.5
   (lazy imports and optional extras) and steps 12.1–12.3 (provider adapters,
   optional Ollama, capability detection for every provider).
5. Steps 9.1–9.3 (event bus, capability packs, turn router), so later
   features are added as packs instead of as special cases.
6. Step 9.7 (live status overlay), which needs only the event bus, and step
   11.7 (sandbox interface).
7. Phase 5, then Phases 6, 7, 8 and the rest of 10, 11 and 12, adding new
   tools to the right pack. Do step 12.9 (resume fallback) right after step
   6.6 (session resume).
8. Steps 9.4–9.6 and 9.8–9.11 (escalation, context manager, orchestrator,
   timeline and estimates) once the planning steps they depend on are done.

Rules for every step:

- The full test suite passes before the PR is merged.
- No behaviour changes in refactoring steps. Behaviour changes go in their
  own steps.
- Update `README.md` and `.github/copilot-instructions.md` when a step
  changes how things work.

### Phase 1: Safety net (tooling and tests)

| Step | Task | Fixes | Done when |
|---|---|---|---|
| 1.1 | Change `[tool.pytest.ini_options] pythonpath` to `["src"]`, install the `dev` extras in `.venv`, and switch `.vscode/settings.json` to pytest | A4 | Plain `pytest -q` runs from the repo root |
| 1.2 | Add `skipif` markers for tests that need `bwrap` / `prlimit`, and `importorskip` for media tests | A4 | Missing tools show up as skips, not failures |
| 1.3 | Investigate the 3 git failures in `test_code_tasks.py` and the 5 failures in `test_agent_runtime.py`. Fix the code, or fix the tests if they are wrong | A4 | 0 failures on a machine without `bwrap` |
| 1.4 | Add a `[tool.ruff]` section (line length, rule sets `E,F,W,I,B,UP,SIM`), run `ruff check --fix` and `ruff format` in a single formatting-only commit | A5 | `ruff check` and `ruff format --check` are clean |
| 1.5 | Add a GitHub Actions workflow running ruff, pytest and `scripts/sync_requirements.py --check` on Python 3.10 and 3.14 | A5, A8 | CI runs on every PR |
| 1.6 | Add pre-commit hooks for ruff | A5 | Hooks run on commit |

### Phase 2: Quick fixes and real bugs

| Step | Task | Fixes | Done when |
|---|---|---|---|
| 2.1 | Fix the 7 `B023` loop-variable-in-closure warnings | A5 | `ruff check --select B023` is clean, with a test for each real bug |
| 2.2 | Replace `VISIBILE_REASONING` with `VISIBLE_REASONING` everywhere, keeping the old config key readable for migration | A8 | The typo appears only in the legacy-key mapping |
| 2.3 | Remove `sys_modules_safe()` from `config.py` and handle that case with a test fixture | A3 | Production code never checks for pytest |
| 2.4 | Update `.github/copilot-instructions.md` so it no longer points to deleted files | A8 | No links to missing files |
| 2.5 | Add `check=` explicitly to every `subprocess.run` call | A5 | `PLW1510` is clean |

### Phase 3: Remove global state

| Step | Task | Fixes | Done when |
|---|---|---|---|
| 3.1 | Create a typed `Settings` model (pydantic) that mirrors the config sections. Load it in `cli.main`, but keep the old constants as thin aliases for now | A2 | `Settings` is loaded once at startup |
| 3.2 | Move each module from the global constants to the `Settings` object, one module per PR, then delete the aliases | A2 | `config.py` does no work at import time |
| 3.3 | Move the `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` setup into startup code | A2 | Importing the package has no side effects |
| 3.4 | Create a `ToolContext` object (database, workspace root, active task, skills, code task, approval state) and pass it to tools through `RunnableConfig` or injected arguments | A2, B3 | `set_active_*()` functions and `global` statements are gone from `tools/` |
| 3.5 | Make the workspace root explicit: set it once at startup, and fail if the folder does not exist | B5 | `SandboxManager` has no import-time default |

### Phase 4: Restructure the core

| Step | Task | Fixes | Done when |
|---|---|---|---|
| 4.1 | Write characterization tests for the current agent loop: tool calls, retries, budgets, empty responses | A1, B1 | The tests cover the loop's main paths |
| 4.2 | Extract a single reusable `run_agent_loop(model, tools, messages, policy, budget)` and replace the ~12 copies in `runtime.py` | A1, B1 | `bind_tools` is called in one place |
| 4.3 | Create a single `PermissionPolicy` that every tool call goes through. Remove `console.input` from `tools/shell.py` and the duplicate `_tool_needs_permission` from `runtime.py` | B2 | No tool prompts the user directly |
| 4.4 | Split slash-command handling (`/tasks`, `/plan`, `/maintenance`, `/skills`, `/think`) out of `runtime.py` into an `agent/commands/` package, one module per command | A1 | `_run_agent_cli_session` is under about 300 lines |
| 4.5 | Split `PersistentMemory` into repositories (chat history, episodes, learned items, tasks, skills) behind a thin facade | A1 | No database module is over about 800 lines |
| 4.6 | Replace the catch-all facade in `agent/__init__.py` with an explicit `__all__`, and update the tests that import private names | A8 | Only public names are exported |
| 4.7 | Use one SQLite connection per thread, or add a lock around the shared one | A7 | Concurrent tool calls cannot corrupt state |

### Phase 5: Improve individual features

| Step | Task | Fixes | Done when |
|---|---|---|---|
| 5.1 | Make network and shell tools async (`httpx.AsyncClient`, `asyncio.create_subprocess_exec`), removing `asyncio.run` from thread pools | A7, B8 | `ASYNC221` is clean, and no `asyncio.run` is left outside `cli.main` |
| 5.2 | Replace print-based diagnostics with module-level `logging` loggers, keeping `rich` for user-facing output | A6, E34 | Diagnostics go to the run log, and user output stays unchanged |
| 5.3 | Narrow the 93 blind `except Exception` blocks to specific exceptions, keeping broad catches only at the outer session boundary | A5 | `BLE001` warnings are only at documented boundaries |
| 5.4 | Rewrite `evaluate_shell_command` using `shlex` parsing and a program/flag list | B4 | `echo /usr` is not flagged and `rm -fr` is, with tests |
| 5.5 | Use reciprocal rank fusion in `HybridRAGRetriever` and a punctuation-aware tokenizer for BM25. Measure before and after with `scripts/benchmark_rag.py` | B9 | Recall@k and MRR are the same or better |
| 5.6 | Support YAML front matter (`name`, `description`) in skills, and match them by embedding similarity, falling back to the current keyword match | B6, C9 | Old skills still load, and new skills use front matter |
| 5.7 | Put web search behind a provider interface, with SearXNG (self-hosted) or an API as the main option and the DuckDuckGo scraper as a fallback | B7, E18 | The search provider is chosen in config |
| 5.8 | Decide whether to keep `requirements.txt`. If it is not needed for deployment, delete it and `scripts/sync_requirements.py` | A8 | One source of truth for dependencies |
| 5.9 | Remove the `sys.path` edit from `main.py`, or drop `main.py` in favour of `python -m private_agent` | A8 | No manual `sys.path` changes |

### Phase 6: Features from Claude Code and Codex

Steps are in priority order: the cheapest, highest-value features first, and
the features that depend on earlier phases later.

| Step | Task | Fixes | Depends on | Done when |
|---|---|---|---|---|
| 6.1 | Load a project instruction file (`AGENTS.md`, with an optional project-specific name set in config) from the workspace root and add it to the system prompt | C1 | 3.1 | Instructions in `AGENTS.md` show up in the model's context, with a test |
| 6.2 | Add `offset` and `limit` (line-based) to `read_local_file`, and report the total line count so the model can page through large files | C10 | — | Large files can be read in chunks |
| 6.3 | Shell improvements: make the timeout configurable per call (with a maximum), support pipes and `&&` by running `bash -c` inside the Bubblewrap sandbox, and add background processes with tools to read output and stop them | C5 | 4.3, 5.1 | `pytest \| tail`, a 5-minute build and a dev server all work inside the sandbox |
| 6.4 | Add an Esc / single Ctrl+C handler that cancels the current turn and keeps the session; keep double Ctrl+C for exit | C12 | 4.2 | Stopping a turn does not end the session |
| 6.5 | Add permission rules in config (`allow` / `deny` lists matching tool names, commands and paths, for example `run_shell_command(pytest *)` or `read_local_file(.env)`) | C8 | 4.3 | Allowed commands run in Auto mode without a prompt; denied ones are refused |
| 6.6 | Save full conversations per session and add `--continue` (last session) and `--resume` (pick a session). Provider fallback on resume is step 12.9 | C3 | 4.2, 4.5 | A resumed session continues with its full message history |
| 6.7 | Add `/rewind` to restore files from `.agent_snapshots/` (and the conversation) to an earlier turn | C3 | 6.6 | An edit can be undone from the prompt, with a test |
| 6.8 | Add a non-interactive mode: `private-agent -p "prompt"` with `--output text\|json`, using the permission rules from 6.5 instead of prompts | C2 | 4.2, 6.5 | The agent can run from a script or CI job without a terminal |
| 6.9 | Add project-level config (`.private_agent/settings.json` in the workspace) and a local override file, merged over the user config | C13 | 3.1 | Project settings apply only inside that project |
| 6.10 | Add custom slash commands loaded from Markdown files in a commands folder (user and project level) | C9 | 4.4 | A new `.md` file becomes a `/command` |
| 6.11 | Add hooks in config: scripts that run before or after tool calls and at session start, able to block a call | C7 | 4.2, 4.3 | A post-edit formatter hook runs after every file edit |
| 6.12 | Add git write tools (create branch, stage, commit) behind permission checks, and optional PR creation through `gh` | C11 | 4.3, 6.5 | The agent can commit its work on a new branch after approval |
| 6.13 | Add subagents: a tool that starts a separate agent loop with its own context and a limited tool set, and returns only its summary | C6 | 4.2, 3.4 | A search can be delegated without filling the main context |
| 6.14 | ~~Add a macOS sandbox using Seatbelt~~ Moved to steps 11.7–11.8 | C4 | 11.8 | See 11.8 |
| 6.15 | ~~Add a status line showing model, context use and permission mode~~ Replaced by step 9.7 | C14 | 9.7 | See 9.7 |

### Phase 7: Planning mode

Steps with no dependency ("—") are small and can be done any time after
Phase 2, before the big restructuring.

| Step | Task | Fixes | Depends on | Done when |
|---|---|---|---|---|
| 7.1 | Give planning mode read-only tools: `read_local_file`, `list_directory`, the `search_workspace_*` tools, `inspect_workspace_git`, and `fetch_webpage` with consent | D1, D9 | — | A coding plan can quote real file names and code, with a test |
| 7.2 | Drop the `read_table_schema` requirement and the one-call-per-round limit for planner tools, and make most step fields optional | D4 | — | A plan can be saved in one model round |
| 7.3 | Include the whole approved plan when running it, as readable text with validation criteria, risks, assumptions and constraints | D5 | — | The execution prompt contains every step field, with a test |
| 7.4 | Replace the regex gate: require an approved plan at the edit and command tools themselves (or through an explicit `/code` command), not from the wording of the message, and fix the blocked-request hint | D2, D10 | 4.3 | Questions are never blocked, and every edit inside a coding task needs an approved plan |
| 7.5 | Make `/plan` a mode that stays on until the plan is approved or `/plan off`, and add an `ask_user` tool for clarifying questions | D3 | 4.2, 4.4 | A plan can be refined over several messages |
| 7.6 | Show the plan as Markdown right after it is saved, with three choices: approve, approve and allow edits, or give feedback and keep planning. Let the user edit the plan text. Keep the typed `APPROVE <id>` only for the riskier access levels | D7, D8, D13 | 7.5 | Approving takes one choice for normal plans |
| 7.7 | Check validation automatically where possible: run the plan's test commands and pre-fill evidence from tool results so the user only confirms | D6 | 6.3 | Coding steps with test commands are verified without typed evidence |
| 7.8 | Give each plan step its own time budget | D12 | 4.2 | A long plan runs to the end without a manual resume |
| 7.9 | Allow more than one active plan, using the `ToolContext` | D11 | 3.4 | Two plans can be active in different sessions |
| 7.10 | Add a text-plan fallback for models without tool support, parsed into the same structure by the runtime | D14 | 7.6 | Planning works with a non-tool model |

### Phase 8: Role coverage

Steps are in priority order within the phase. Do steps 9.2–9.3 (capability
packs and turn router) first, so each step here adds its tools and prompts to
the right pack without affecting the others.

| Step | Task | Fixes | Depends on | Done when |
|---|---|---|---|---|
| 8.1 | ~~Add separate role profiles with a command to switch~~ Replaced by capability packs and the turn router (steps 9.2–9.3), which combine roles in one runtime | E33 | 9.2, 9.3 | See 9.2 and 9.3 |
| 8.2 | Render replies as Markdown with `rich.markdown` (including while streaming; done in step 10.2), and keep diagnostics apart from replies | E1, E34 | 5.2 | Code blocks, tables and lists render correctly |
| 8.3 | Add a relevance cut-off for RAG results and raise the default to 4–6 chunks. Make the extra coding-standards query configurable or remove it | E6, E7, E15 | — | "Hello" gets no chunks, and `scripts/benchmark_rag.py` scores are the same or better |
| 8.4 | Add a `search_documents` tool so the model can search the documents itself | E9 | 3.4 | The model can run a second, better search during a turn |
| 8.5 | Rewrite follow-up questions using the conversation before searching | E8 | 8.3 | "What about the second one?" retrieves the right chunks, with a test |
| 8.6 | Add re-ranking of retrieved chunks (local cross-encoder or the chat model) | E10 | 5.5 | Benchmark MRR improves |
| 8.7 | Add a documents-only answer mode that says when the answer is not in the documents | E12 | 8.3 | The agent says "not found in your documents" instead of guessing |
| 8.8 | Support more formats: `.docx`, `.xlsx`, `.pptx`, `.epub`, text extraction from `.html` (BeautifulSoup is already a dependency), optional OCR for scanned PDFs | E11 | — | Each format is indexed, with a test file per format |
| 8.9 | ~~Put embeddings behind a provider interface~~ Moved to step 12.11 | E14 | 12.11 | See 12.11 |
| 8.10 | Support several document collections and choosing one per question or per project | E13 | 3.1 | A question can be limited to one collection |
| 8.11 | Chat commands: `/new`, `/retry`, `/edit`, `/export`, named conversations, and personas | E2, E3, E4, E5 | 6.6 | Each command works, with tests |
| 8.12 | Raise search results to a configurable 8–10, and add paged `fetch_webpage` (offset and length) | E18, E20 | 5.7 | Long pages can be read to the end |
| 8.13 | Add a research loop to the Research pack (with `/research` as a shortcut): search, read, find gaps and search again, ending in a saved Markdown report with sources. Add downloaded files to RAG automatically, and allow consent once per research task | E19, E21, E22, E25 | 9.3, 8.4, 8.12 | A research question produces a report file with cited sources |
| 8.14 | Add source dates and domain checks, cross-checking between sources, and specialist sources (Wikipedia API, arXiv, GitHub search) as optional providers | E23, E24 | 8.13 | Reports show source dates and flag single-source claims |
| 8.15 | Run tests and lint automatically after edits outside coding tasks (through hooks), and add language tooling through an LSP MCP server or built-in type checking | E16, E17 | 6.11 | A failing test or type error after an edit is reported to the model |
| 8.16 | Document example MCP server configs for calendar, notes, email and reminders instead of building each tool | E26 | — | The README has working example configs |
| 8.17 | Add a calculator / sandboxed Python tool | E29 | 6.3 | Calculations run without the general shell tool |
| 8.18 | Add document creation tools (Markdown to `.docx` and PDF, simple `.xlsx`) | E27 | — | The agent can write a `.docx` and a PDF, with tests |
| 8.19 | Allow extra approved folders outside the workspace (for example `~/Documents`), each with its own permission rule | E32 | 6.5 | The helper can read an approved folder outside the workspace |
| 8.20 | Lower priority: clipboard, notifications and opening URLs; scheduled tasks; voice conversation with spoken replies | E28, E30, E31 | 9.2 | Each feature is behind its own setting and permission |

### Phase 9: Single combined runtime and live status

This phase implements Part F. Steps 9.1–9.6 replace the separate role
profiles of step 8.1, and step 9.7 replaces the simple status line of step
6.15.

| Step | Task | Fixes | Depends on | Done when |
|---|---|---|---|---|
| 9.1 | Add a typed event bus. Emit events from the agent loop, model calls, tool executor, approvals and planner. Move session statistics and the run log onto it | F2, A6, E34 | 4.2 | Session statistics are computed only from events, with tests |
| 9.2 | Group tools, prompt sections and limits into capability packs (Core, Knowledge, Research, Coding, Helper, Planning, MCP) | F2, E33 | 3.4, 4.2 | Each tool belongs to exactly one pack, and a turn can bind any set of packs |
| 9.3 | Add the turn router: commands, then rules, then model classification with a safe fallback. Turn `/code`, `/research`, `/plan` and `/ask` into shortcuts that force packs | F2, D2 | 9.2 | Plain chat binds only Core; "research X then change Y" binds Research and Coding, with tests for each route |
| 9.4 | Add the `enable_capability` tool for escalation during a turn, checked by `PermissionPolicy` | F2 | 9.3, 4.3 | Chat can switch to coding after approval, without starting a new turn |
| 9.5 | Add the context manager with a token budget per source, and show context use | F2, E6 | 9.2 | No source goes over its share unless others are unused, with tests |
| 9.6 | Add the task orchestrator: complexity check, plan with read-only tools from all chosen packs, one-choice approval, per-step packs and budgets, automatic verification and a final report | F2, D1, D3, D6, D7, D12 | 9.3, 7.5, 7.6, 7.8 | A multi-step request runs as a plan whose steps use different packs, end to end |
| 9.7 | Add the live status overlay: `rich.live` panel during turns, `prompt_toolkit` bottom toolbar between turns, paused for approvals, plain-mode fallback | F3, C14 | 9.1 | Running tools, finished tools, packs, model speed and context use are shown live; approvals still work |
| 9.8 | Add the `agent_step_timings` table, record every step, and fill it from existing `agent_task_events` history | F4 | 9.1, 4.5 | Every finished step has a timing row |
| 9.9 | Add the estimator: history, model speed and the model's guess, shown as ranges with confidence, recalibrated after each step | F4 | 9.8 | On a replayed set of past tasks, most actual durations fall inside the estimated range |
| 9.10 | Add the timeline overlay (Ctrl+T) with per-step status, elapsed and estimated times, overall finish range, waiting time and budget warnings | F4 | 9.7, 9.9 | A planned task shows its timeline, and a warning appears before the time budget runs out |
| 9.11 | Write events as JSON lines in non-interactive mode | F2, F3 | 9.1, 6.8 | `private-agent -p … --output json` streams tool and step events |

### Phase 10: User interface

Steps 10.1–10.3 are small and can be done right after Phase 2.

| Step | Task | Fixes | Depends on | Done when |
|---|---|---|---|---|
| 10.1 | Fix the startup bugs: give the online-provider prompt a way back or out, create the default RAG folder (or skip quietly), drop the spinner when indexing is skipped, and ask for the workspace before the RAG folder | G1, G2, G3, G4 | — | First run with no model and no RAG folder ends cleanly, with a test for each case |
| 10.2 | Render replies as Markdown with code highlighting, including while streaming (same as step 8.2) | G5, E1 | — | Code blocks, tables and lists render correctly |
| 10.3 | Show edit previews as coloured unified diffs with `rich.syntax` | G6 | — | Added and removed lines are coloured in approval prompts |
| 10.4 | Cut the startup questions: remember the last choices, add `--model`, `--provider`, `--mode`, `--workspace` and `--rag` flags, and only ask about what is not set | G7 | 3.1 | Starting with saved settings asks no questions |
| 10.5 | Use one shared console with a named theme, a few message levels (info, success, warning, error) with one colour and symbol each, and replace the ~57 ad-hoc `[Label]` prefixes | G15, G16, G22, G24 | 5.2 | All output goes through one console and theme, and `NO_COLOR` is respected |
| 10.6 | Add multi-line input (Alt+Enter for a new line) and key bindings | G10 | — | A multi-line prompt can be typed and edited |
| 10.7 | Show tool requests in readable form, and add Ctrl+O (or `/last`) to see full tool output | G11, G20 | 9.1 | The full output of the last tool can be shown on request |
| 10.8 | Better approval prompts: a readable request, coloured diff and choices for yes, no, and always allow for this session or project | G12 | 6.5 | "Always allow" adds a permission rule |
| 10.9 | Arrow-key menus for provider, model, mode and `/tasks`, with typed input still possible | G13 | 4.4 | Menus work with arrow keys and with typed numbers |
| 10.10 | Cosmetics: welcome screen, informative prompt, divider between turns, per-turn tokens and speed line, grouped `/help` | G17, G18, G19, G21, G23 | 10.5 | Each item is shown, and each can be turned off in config |

### Phase 11: Platform support (POSIX only)

Steps 11.1–11.4 make the app work in WSL without hassle and should be done
early, right after Phase 1.

| Step | Task | Fixes | Depends on | Done when |
|---|---|---|---|---|
| 11.1 | Declare POSIX only: add `Operating System :: POSIX` classifiers, exit on native Windows with a message pointing to WSL, remove the tests that simulate Windows, and update the README wording about "desktop platforms" | H4 | — | Native Windows exits with the WSL message, and no Windows-specific tests remain |
| 11.2 | Add `platform_support.py` to detect Linux, WSL, macOS and Termux and report available features. Replace direct `platform.system()` checks with it | H3b, H4 | — | Each platform is detected correctly, with tests using fake `/proc` files and environment variables |
| 11.3 | Add the `private-agent doctor` command, with the exact fix command per detected distribution and platform (for example `sudo dnf install bubblewrap`) | H3b, H3d | 11.2 | On this WSL machine, doctor reports the missing `bwrap` with `sudo dnf install bubblewrap` |
| 11.4 | WSL support: warn when the workspace or RAG folder is under `/mnt/<drive>`, document Ollama inside WSL vs. Windows-side Ollama with mirrored networking, and allow an explicitly configured WSL host-gateway address as a local runtime | H3e, H3f | 11.2 | Doctor shows how Ollama is reached, and a `/mnt/c` workspace shows a warning |
| 11.5 | Import optional parts lazily (Ollama client, Chroma, media) and split dependencies into a core plus `rag`, `media` and `all` extras | H3a, H3k | 3.3 | `pip install .` without extras starts, and missing features are reported instead of crashing |
| 11.6 | Add a fallback for document search without the `rag` extra: BM25 only, or SQLite FTS5 | H3k | 11.5 | RAG answers with citations on a core-only install |
| 11.7 | Add a sandbox interface with Bubblewrap, Seatbelt and None backends; move the current Bubblewrap code behind it, and use `resource.setrlimit` where `prlimit` is missing | H3h, H4 | 5.1 | The Bubblewrap backend passes the existing sandbox tests unchanged |
| 11.8 | Add the macOS Seatbelt backend (same as step 6.14), and clear messages for camera and microphone permissions | H3h, H3i | 11.7 | Shell, tests and stdio MCP work on macOS |
| 11.9 | Termux support: detect it, keep the workspace in Termux's home folder (warn on `/sdcard`), use a compact layout on narrow screens, and add an explicit off-by-default "unsandboxed, approve every command" session option (project owner to decide whether to ship it) | H3j, H3l, H3m | 11.5, 11.7, 10.5 | The core chat, RAG fallback and online or Termux-local models work on Termux |
| 11.10 | Report missing webcam and microphone access clearly on WSL and Termux | H3g | 11.2 | Media tools explain why they are unavailable instead of failing |
| 11.11 | Add a CI matrix: Ubuntu x86_64, Ubuntu arm64 and macOS, plus a core-only install job that mimics Termux | H3c | 1.5 | All jobs pass on every PR |
| 11.12 | Write setup docs for WSL, macOS and Termux | H4 | 11.3 | Each platform has a tested step-by-step setup in the README |

### Phase 12: Local provider independence

Steps 12.1–12.3 can start after Phase 3. Step 12.9 (resume fallback) needs
session resume (6.6) first. This phase replaces step 8.9 (embedding
providers).

| Step | Task | Fixes | Depends on | Done when |
|---|---|---|---|---|
| 12.1 | Add the provider adapter interface and move the current Ollama and OpenAI-compatible code behind it, with no change in behaviour | I1b, I1d, I1g | 3.1 | Existing provider tests pass through the new adapters |
| 12.2 | Import `ollama` and `langchain_ollama` only inside the Ollama adapter, and make them an optional `ollama` extra | I1e | 12.1, 11.5 | The app starts with no Ollama packages installed |
| 12.3 | Capability detection for OpenAI-compatible servers: server-reported data, known model families, a cached test call, and config overrides. Extend the tool-capable model redirect to all adapters | I1b, I1c | 12.1 | An LM Studio model gets real tool, vision and context values, with tests |
| 12.4 | Add presets (LM Studio, llama.cpp, vLLM, Jan, KoboldCpp, GPT4All, LocalAI, llamafile, MLX-LM), each with verified default port, install check and start hint. Identify the server from its responses, not only its port | I1k | 12.1 | Each preset is detected when running, with recorded test responses |
| 12.5 | Detect installed-but-not-running providers, and offer to start them after consent | I1j | 12.4 | An installed, stopped Ollama is listed as "installed · not running" and can be started from the prompt |
| 12.6 | Filter embedding models from chat lists for every adapter, and allow several servers of the same type (a list of Ollama URLs) | I1i, I1l | 12.1 | Two Ollama servers are both listed, and embedding models are not offered as chat models |
| 12.7 | New startup selection rules: ask when two or more local providers are usable, auto-select a single one, last-used as default, `--provider` and `--model` flags, local before online, no endless loop | I1n, G1, G7 | 12.5, 10.4 | Each rule has a test, including 0, 1 and several providers |
| 12.8 | Store a provider record (adapter, name, URL, model, needed capabilities) with each session and each task | I1p | 4.5 | New sessions and tasks have a provider record, and old rows migrate with "unknown" |
| 12.9 | Resume fallback: same provider and model → resume; model missing → model prompt; provider missing → provider prompt with reason and start option; capability and context checks; explicit consent before going online; record the switch | I1p | 12.7, 12.8, 6.6 | Resuming a session whose provider is stopped shows the fallback prompt, with tests for each case |
| 12.10 | Offer the fallback prompt when the provider stops answering mid-session | I1o | 12.9 | Stopping the server mid-session leads to the prompt instead of a dead session |
| 12.11 | Embeddings from any adapter: separate embedding provider setting, record the provider and model in the RAG index, rebuild when they change, keyword-search fallback | I1a, E14 | 12.1, 11.6 | RAG works with only LM Studio installed, and changing the embedding model triggers a rebuild |
| 12.12 | Allow explicitly configured non-loopback local addresses (WSL host gateway, home network) with a clear warning | I1m, H3e | 12.1, 6.5 | A configured gateway address works after the user confirms it |
| 12.13 | Hardware status through the adapter where available, with a clear "not exposed by this provider" message otherwise; make the RAG benchmark use the embedding adapter | I1f, I1h | 12.1, 12.11 | `/hardware-status` and the benchmark work with every adapter |

### Progress tracker

- [ ] Phase 1: Safety net (1.1–1.6)
- [ ] Phase 2: Quick fixes and real bugs (2.1–2.5)
- [ ] Phase 3: Remove global state (3.1–3.5)
- [ ] Phase 4: Restructure the core (4.1–4.7)
- [ ] Phase 5: Improve individual features (5.1–5.9)
- [ ] Phase 6: Features from Claude Code and Codex (6.1–6.15)
- [ ] Phase 7: Planning mode (7.1–7.10)
- [ ] Phase 8: Role coverage (8.1–8.20)
- [ ] Phase 9: Single combined runtime and live status (9.1–9.11)
- [ ] Phase 10: User interface (10.1–10.10)
- [ ] Phase 11: Platform support, POSIX only (11.1–11.12)
- [ ] Phase 12: Local provider independence (12.1–12.13)
