# Private Agent — Remaining Production Work

This roadmap contains work that is still outstanding. Previously implemented
items have been removed from the pending list. The project is not yet
production-ready: WebSocket MCP cannot use the pinned HTTP transport, sandbox
and lifecycle coverage is incomplete, and the CI/deployment process has not
been verified on all declared environments.

## Completed and removed from pending scope

The current implementation already includes local Ollama and explicit
OpenAI-compatible provider selection; SQLite conversations and bounded recent
history/context summaries; chunked local RAG; model capability inspection and
thinking controls; explicitly approved Markdown skill creation and immediate
activation; a bounded tool loop; configured MCP tool discovery and per-call
approval; isolated code-task workspaces with local Git checkpoints
and test gates; Linux Bubblewrap isolation for approved shell commands and
generated-project tests; bounded and atomically persisted RAG indexing with
fail-closed damaged-state handling and confirmed timestamped backup/rebuild;
session-scoped, bounded summaries and configurable retention; token-budgeted
history/context and separate source citations; SQLite and HTTP client shutdown
cleanup with MCP tool-registry restoration; peer-validated, IP-pinned HTTP
transports for web search/fetch/download, Ollama, remote HTTP MCP, and explicit
Online endpoints; isolated stdio MCP processes on Linux; bounded network
request time/concurrency; multi-runner code-task testing; MCP tool
catalog/schema validation; stricter local-model fallback; hardware status
reporting and CPU policy; the `private-agent` entry point; CI with focused Ruff
correctness checks; and the current unit suite. These items are not
repeated as future implementation tasks below. Their remaining limitations
are listed only where further work is still necessary.

## P0 — Release blockers

### Extend and validate execution isolation

- The current Linux backend uses Bubblewrap namespaces, a dedicated writable
  project mount, restricted read-only runtime mounts, disabled networking,
  dropped capabilities, and `prlimit` resource ceilings. It fails closed when
  Linux, `bwrap`, or `prlimit` is unavailable. Generated-project commands and
  configured stdio MCP processes use this boundary, with separate temporary
  workspaces for MCP. It is not a container boundary.
- Expand platform support only with equivalently enforced filesystem,
  identity, network, and resource restrictions; do not silently fall back to
  host execution.

**Acceptance:** add tests proving credential-path denial, process/resource
limits, sandbox failure modes, and actual isolation behavior in CI. The current
test proves only workspace writes and denial of one host-file and loopback
access. Add equivalent backends only after validating them on supported OSes.

### Enforce outbound network policy outside URL parsing

- Web fetch/download and Online API HTTP clients now resolve and validate
  addresses immediately before connecting, pin the selected address for that
  connection, disable environment proxies, and validate each redirect. Online
  provider mode permits a user-selected loopback or explicit private IP literal.
- Request duration and concurrency are bounded for web fetch/download;
  Online connection pools/timeouts are bounded separately. Download failures
  preserve existing destinations.
- DuckDuckGo search, configured Ollama endpoints, and remote MCP over HTTP now
  use the pinned transport. Ollama/HTTP MCP permit explicitly selected
  loopback/private literal endpoints; hostnames resolving privately remain
  blocked. MCP WebSocket is restricted to IP literals because
  that adapter transport offers no pinned HTTP client factory.
- Synchronous OS DNS resolution cannot be forcibly interrupted by the Python
  request timeout; use an egress proxy/resolver with an enforceable deadline
  for a hard total DNS-plus-connect bound.

**Acceptance:** expand tests for IPv4/IPv6, redirects, oversized and slow
responses, and atomic destination preservation. Current unit tests cover
address pinning, mixed public/private DNS, redirect rejection, timeout, and
failure preservation; exercise these against the actual transport in CI.

### Test the real agent lifecycle

- Add deterministic end-to-end tests for startup, local and Online selection,
  one normal turn, tool call and tool result, tool error recovery, persistence,
  session resume, shutdown, and code-task completion/blocking.
- The suite now exercises a mocked local CLI turn, resume, persistence,
  shutdown, tool result/error handling, and Online model listing/auth redaction.
  These are not yet a complete HTTP-boundary server or a full Online CLI turn.
- Mock Ollama and an OpenAI-compatible server at the HTTP boundary, including
  auth failures, timeouts, malformed responses, tool-call payloads, and
  privacy-sensitive request bodies.
- Test approval and denial flows through the same runtime path used by CLI
  tool calls. Confirm no external call occurs after denial or while offline.
- Cover IPv6 addresses and sensitive argument redaction for every tool origin;
  current MCP prompt/log redaction is tested, but full tool-audit behavior is
  not yet established.
- The generated Python runner clears pytest's cache and bytecode; detected
  Python, npm, Go, and Cargo suites run in the OS sandbox. Tests now cover
  missing Python runners, timeouts, ignored test files, and multiple runners.
  Expand final-checkpoint and failure-flow coverage.

**Acceptance:** all lifecycle tests run without a live Ollama service or
external network; the tests prove request contents, persisted state, and
side-effect boundaries rather than only testing helper functions.

## P1 — Reliability and privacy

### Complete resource-lifecycle coverage

- SQLite and tracked HTTP clients are closed in the agent's `finally`-owned
  runtime lifecycle, and dynamically registered MCP tools are restored. The
  inspected MCP adapter opens a new session per tool call.
- Verify cleanup under keyboard interrupt, startup failures, unexpected
  exceptions, and MCP server process failures; add coverage for repeated CLI
  invocations in one process.

**Acceptance:** lifecycle tests verify each resource closes exactly once on
all exit paths, no server process is leaked, and one failed MCP server does not
prevent other configured servers from shutting down cleanly.

### Make code-task setup and verification explicit

- The CLI asks for an existing source path or explicit new-project choice.
  Test commands are discovered from Python tests and package/go/cargo project
  metadata; detected applicable suites all run, and failures/missing runners
  are reported instead of silently skipped.
- Add explicit metadata-driven test commands, configured source paths for
  unattended use, and coverage for ignored/untracked file preservation.
- Report copy exclusions and checkpoint results, preserve ignored/untracked
  user files, and guarantee test or commit failures are never presented as
  verified completion.

**Acceptance:** tests cover new and copied projects across supported runners,
source preservation, ambiguous/headless selection, excluded secrets,
ignored files, multiple test commands, failed tests, and final status.

### Bound memory by useful context and provide retention controls

- Configurable token budgeting now bounds local history and retrieved/episodic
  context, prioritizes recent messages and the current user request, and keeps
  retrieved sources visible as citations. Retention days prune messages and
  summaries together; clear-history deletes summaries with their session.
- Make the token budget model-aware where metadata is available, bound system
  prompt/output tokens, and test retention/deletion behavior end-to-end.
- Keep memory summaries bounded, session-scoped, and isolated between users
  when multiple operating-system accounts or profiles are supported.

**Acceptance:** long-session tests stay within the configured context budget;
recent instructions and citations remain available; retention and deletion
remove exactly the documented records.

### Make RAG indexing recoverable and bounded

- Configurable file, corpus, PDF-page, and chunk limits are enforced. Index
  state writes are atomic and occur after vector operations; damaged state
  fails closed. The CLI now asks before preserving the old index as a
  timestamped backup and rebuilding.
- Remaining work: test recovery if the Chroma database itself is corrupt,
  fully transactional partial vector updates, deletion/modified-source edge
  cases, and citations through the live CLI.

**Acceptance:** tests cover size limits, corrupted state, partial vector-store
failures, recovery/rebuild, deleted and modified files, and source citations.

### Complete provider and capability verification

- Online model listing and auth failure redaction have mocks; API clients now
  use the pinned outbound transport. Add mock-server coverage for
  rate-limits, retries, timeouts, tool calls, malformed responses, and
  redaction. Verify raw API keys never appear in logs, config, SQLite, or
  failure output.
- Test model fallback only for recognized availability/capability failures;
  preserve privacy mode, permissions, task context, and hardware policy.
- Keep vision/audio inputs explicitly unsupported until implemented. Reject
  such inputs clearly instead of silently treating them as text or sending
  them to another provider.

**Acceptance:** no implicit local-to-remote fallback occurs; Online request
bodies contain only user-approved context; unsupported capabilities produce
an actionable error without leaking input.

## P2 — Operations and maintainability

### Manage MCP permissions and lifecycle consistently

- A tool catalog now reports origin, approval policy, and known/network-capable
  effects; configured MCP transport fields and timeout ranges are validated.
- Stdio MCP children now run in Bubblewrap with no network, an isolated
  temporary writable workspace, read-only runtime mounts, and resource limits.
  Remote SSE/streamable-HTTP uses the pinned async HTTP transport. WebSocket
  cannot pin hostnames and is limited to IP literals.
- Continue enforcing a consistent invocation policy with bounded execution,
  safe output handling, and auditable approvals. Verify each transport,
  isolation, and shutdown path through integration tests.

**Acceptance:** integration tests cover each supported transport, malformed
configuration, schema conflicts, invocation approval, timeout, server
failure, and clean shutdown.

### Establish repeatable release engineering

- Run the new GitHub Actions workflow on the actual supported Python versions
  and fix incompatibilities; add supported operating systems only after the
  dependencies and runtime are verified there.
- A focused Ruff correctness check now runs in CI. Add formatting, broader
  lint/type-check jobs; establish dependency update and vulnerability
  monitoring.
- Wheel/sdist builds and CLI help have passed locally. Add clean-environment
  installation smoke tests and verify packaged resources/default config.
- Document upgrade, backup/restore, supported deployment platforms, logging,
  and recovery from corrupt databases or vector indexes.

**Acceptance:** clean artifact installation and CI pass on every declared
platform/Python combination without network-dependent unit tests or a local
Ollama daemon.

## P3 — Optional improvements

- Model-aware context-window selection and optional summarization benchmarks
  remain.
- Benchmark retrieval and hardware policies on representative datasets and
  machines. Report measured environment and methodology; do not claim a speedup
  from device detection alone.
- Add direct CUDA, ROCm, Vulkan, Metal, or OpenVINO integration only if a
  maintained backend can select and verify it safely. Ollama's current status
  reporting indicates VRAM allocation but does not identify the active vendor
  backend.
- Optional webcam capture and microphone recording/transcription are available
  to local sessions with per-use approval, bounded capture duration/image count,
  and optional dependencies. Webcam frames are sent only to a vision-capable
  local model; microphone audio is transcribed locally with a configured
  on-disk Whisper model. Online providers do not receive media tools.
- Remaining multimodal work:
  1. Add explicit user image-file selection, format/size validation, and the
     same local vision capability/privacy checks used for webcam frames.
  2. Add video-file handling using bounded frame sampling and strict duration,
     frame-count, and decoded-size limits; do not upload or retain source video.
  3. Add direct audio-model support only when an installed local provider
     declares a compatible audio-input API; otherwise continue local
     transcription. Keep remote media disabled unless a separate per-session
     consent flow is implemented.
  4. Add end-to-end tests with mock camera/microphone devices and local model
     HTTP boundaries, proving no capture before approval, no media on Online
     calls, and no raw-media persistence.

## Production readiness exit criteria

Do not describe the application as production-ready until P0 acceptance checks
pass; lifecycle and privacy behavior is covered end-to-end; clean installation
and CI succeed on declared platforms; and security boundaries, data retention,
supported hardware, and operational recovery are documented and verified.
