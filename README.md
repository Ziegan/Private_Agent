# Private Agent

Private Agent is a local-first command-line assistant built on Ollama. It
combines local chat models, SQLite conversation memory, optional local document
retrieval, Markdown skills, built-in tools, and explicitly configured MCP
servers. Multi-step work can use an iterative tool loop with configurable
limits.

## Privacy and execution boundaries

- Chat and embedding requests use the configured Ollama endpoint. Keep
  `ollama_base_url` pointed at a trusted local service for local-only
  inference.
- Web research is enabled by default but asks for approval before sending a
  query. The agent checks connectivity before making a request. If offline, it
  offers a retry or allows the user to continue with clearly incomplete
  offline information. Set `enable_web_research` to `false` to disable it, or
  set `web_research_consent` to `never`.
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
  IP addresses. WebSocket MCP is restricted to explicit IP addresses because
  its transport cannot pin hostname resolution. MCP tool
  calls require interactive approval unless their tool names are explicitly
  listed in `mcp_auto_approve_tools`.
- The `create_skill` tool is available when the user explicitly requests a
  reusable skill. The agent drafts Markdown instructions from that request
  and relevant context, then asks for approval before writing only a new
  `.md` file in the configured skills directory; existing skills are never
  overwritten. Approved skills are available immediately and on later runs.
- Optional local media tools can capture a single webcam frame for a local
  vision-capable Ollama model and record up to 30 seconds from a microphone for
  transcription by a locally installed Whisper model. Every capture prompts
  for per-use approval; camera frames remain in memory and are not saved, and
  microphone audio is not sent to a remote service. Media tools are disabled
  for Online providers. A transcript is ordinary conversation content and may
  appear in the model's reply and saved conversation history. Video-file input
  and direct audio-file/model input are not implemented.
- File tools validate paths against the workspace root. Shell commands require
  interactive approval, then run without shell interpretation in a Bubblewrap
  Linux sandbox with network access disabled, workspace-only writes, dropped
  capabilities, and CPU, memory, process, and file-size limits. Generated
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
- Media input is optional: webcam frames reach only a vision-capable local
  model, and microphone speech reaches a local model as a locally generated
  transcript. Direct audio input to the chat model and video-file input remain
  unsupported.

## Setup

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

The first run creates `.private_agent.conf`. `resources/skills` is the default
Markdown skill directory. Configure `rag_docs_path` to enable local document
indexing; leave it `null` to skip RAG. Conversation history is stored in
`~/.local_ai_memory.db`, and the Chroma index is stored in
`./.local_ai_chroma_db`.

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

Camera and microphone integrations are optional and are not installed with
the base package. Install them from the checkout with
`python -m pip install '.[media]'`. Linux microphone support may also require
the system PortAudio library. Set `PRIVATE_AGENT_WHISPER_MODEL_PATH` to an
existing local faster-whisper model directory before using microphone
transcription; the application will not download a speech model.

Explicit code creation or modification requests use an isolated project under
`~/.private_agent/projects` by default (configurable with `code_output_root`).
For changes to an existing project, the CLI prompts for its source directory
and works on a separate copy; it does not edit the original. Git metadata,
symlinks, likely secrets, and common caches/build outputs are excluded from the
copy. Git is initialized only inside the generated project, with no remote
configured. If Git is unavailable or cannot create the initial checkpoint, the
CLI asks before continuing without Git. Code-task checkpoints run the project
tests and require a unit-test file; final verification runs the tests again.
Python (`pytest` or
`unittest`), npm, Go, and Cargo test projects are recognized. This is not an
OS-level container boundary, but supported Linux project commands run in the
Bubblewrap sandbox described above. Tests cannot read host files outside the
explicit read-only runtime mounts, access the host loopback network, or write
outside the project and temporary filesystem. If the sandbox prerequisites
are missing, code-task verification is blocked rather than run on the host.

RAG indexing is bounded by `rag_max_file_bytes` (5 MiB per file),
`rag_max_corpus_bytes` (25 MiB total), `rag_max_pdf_pages` (250 per PDF), and
`rag_max_documents` (20,000 chunks). The index location is configurable with
`rag_index_path`. If its state file is corrupt or incompatible, the CLI offers
to move the old index intact to a timestamped backup before rebuilding.

Conversation context is limited by `max_context_tokens` (12,000 tokens by
default), preserving the newest messages and user request first.
`conversation_retention_days` defaults to `0` (retention disabled); a positive
value prunes older messages and summaries at startup. Session summaries are
also capped by `max_summary_chars`. All outbound HTTP for web tools, Ollama,
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

Example `.private_agent.conf`:

```json
{
  "default_db_path": "~/.local_ai_memory.db",
  "workspace_root": ".",
  "code_output_root": "~/.private_agent/projects",
  "skills_folder": "resources/skills",
  "rag_docs_path": null,
  "rag_index_path": "./.local_ai_chroma_db",
  "rag_max_file_bytes": 5242880,
  "rag_max_corpus_bytes": 26214400,
  "rag_max_pdf_pages": 250,
  "rag_max_documents": 20000,
  "max_context_tokens": 12000,
  "conversation_retention_days": 0,
  "max_summary_chars": 2000,
  "max_network_concurrency": 4,
  "network_request_timeout": 30,
  "ollama_base_url": "http://localhost:11434",
  "hardware_acceleration": "auto",
  "preferred_model": null,
  "online_base_url": "https://api.openai.com/v1",
  "online_model": null,
  "default_model_temperature": 0.3,
  "embedding_model": "nomic-embed-text",
  "thinking_toggle_default": false,
  "thinking_effort_default": "medium",
  "enable_web_research": true,
  "web_research_consent": "ask",
  "max_tool_iterations": 15,
  "max_tool_calls": 60,
  "max_task_seconds": 600,
  "max_tool_output_chars": 12000,
  "max_history_messages": 20,
  "mcp_auto_approve_tools": [],
  "mcpServers": {}
}
```

The same runtime dependencies are listed in `requirements.txt`; install test
dependencies with `requirements-dev.txt` if not installing the editable
development extra. `python -m pytest -q` runs the test suite.

The default embedding model in the generated configuration may differ from
this example; set it to a model actually installed in your Ollama instance.
Supported thinking effort values are `low`, `medium`, and `high`. The runtime
uses Ollama model metadata to identify declared tool, thinking, vision, and
audio support; unsupported features are not silently assumed.

MCP server entries use the configuration format expected by
`langchain-mcp-adapters`. For example, a local stdio server may be configured
as:

```json
{
  "mcpServers": {
    "example": {
      "transport": "stdio",
      "command": "python",
      "args": ["-m", "example_mcp_server"]
    }
  }
}
```

Only configure servers you trust. Tool names in `mcp_auto_approve_tools`
persistently bypass the per-call interactive approval prompt; leave this list
empty unless the server and the named tools have been reviewed. Stdio servers
must be installed in the application/runtime locations mounted read-only into
the sandbox; arbitrary host scripts and data paths are intentionally
inaccessible.

## CLI controls

- `exit` or `quit`: end the session and save an episodic summary.
- `switch`: return to model selection.
- `/think` or `/think on|off`: toggle supported model thinking.
- `/think-effort low|medium|high`: request a supported reasoning effort.
- `/think-status`: show requested/effective thinking state and model support.
- `/hardware-status`: refresh Ollama's loaded-model hardware placement report.

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
