# Private Agent — Remaining Production Work

This roadmap lists outstanding work and the recommended implementation
sequence. The project is not yet production-ready: sandbox and lifecycle
coverage is incomplete, outbound DNS resolution has no enforceable hard
deadline, and CI/deployment has not been verified on all declared environments.

The source-package reorganization, focused test layout, dependency
source-of-truth, and wheel/sdist smoke workflow are implemented. Do not repeat
those completed tasks here; extend them only where a remaining release task
below requires it.

## Recommended implementation sequence

1. **Prove the current security boundaries.** Baseline tests now exercise
   credential-file denial, workspace-only writes, loopback denial, configured
   resource ceilings, fail-closed sandbox setup, and private DNS rejection
   through the real HTTP transport for IPv4 and IPv6. Continue checking the
   remaining gaps below before changing policy or adding features.
2. **Close release-blocking boundary gaps.** Make DNS-plus-connect deadlines
   enforceable, verify sandbox denial/resource limits, and retain fail-closed
   behavior when a boundary is unavailable.
3. **Exercise real user flows.** Add offline mocked CLI/runtime tests covering
   provider selection, approvals, tool calls and errors, persistence/resume,
   code-task outcomes, and cleanup.
4. **Harden stateful features.** Complete code-task source preservation,
   memory retention/search/deletion, RAG recovery, provider redaction, and MCP
   lifecycle behavior with focused failure-path tests.
5. **Add safe workspace operations.** Implement bounded search, patch
   preview/apply, workspace diff, read-only Git inspection, and approved
   recoverable destructive operations; keep execution isolated and bounded.
6. **Verify release operations.** Run the committed CI matrix on declared
   Python versions, address failures, and document supported platforms,
   backups/upgrades, and recovery.
7. **Defer optional work.** Add integrations, benchmarks, hardware backends,
   or additional media input only after the security, consent, and support
   boundaries are designed and tested.

Each stage should preserve user-data locations and public CLI behavior. Do not
mark a stage complete based on helper tests alone: meet that stage's acceptance
criteria through the actual runtime boundary.

## P0 — Release blockers

### Extend and validate execution isolation

- Expand platform support only with equivalently enforced filesystem,
  identity, network, and resource restrictions; do not silently fall back to
  host execution.

**Acceptance:** add tests proving credential-path denial, process/resource
limits, sandbox failure modes, and actual isolation behavior in CI. Add
equivalent backends only after validating them on supported OSes.

**Verified baseline:** the Linux subprocess integration test now checks that
workspace files remain writable while host files/credentials and host network
listeners are inaccessible, that CPU/address-space/process/file-size limits
are set, and that writes outside the mounted workspace do not reach the host.
Unit tests also verify non-Linux and missing-tool setups fail closed. Other
operating systems remain unsupported until an equivalent backend is validated.

### Enforce outbound network policy outside URL parsing

- Synchronous OS DNS resolution cannot be forcibly interrupted by the Python
  request timeout; use an egress proxy/resolver with an enforceable deadline
  for a hard total DNS-plus-connect bound.

**Acceptance:** expand tests for IPv4/IPv6, redirects, oversized and slow
responses, and atomic destination preservation; exercise these against the
actual transport in CI.

**Verified baseline:** an actual HTTPX transport request is rejected before
reaching a local listener when a hostname resolves to private IPv4 or IPv6.
Redirect, response-size, timeout, and failed-download preservation cases have
focused tests; stronger actual-transport coverage and the hard DNS deadline
remain outstanding.

### Test the real agent lifecycle

- Add deterministic end-to-end tests for startup, local and Online selection,
  one normal turn, tool call and tool result, tool error recovery, persistence,
  session resume, shutdown, and code-task completion/blocking.
- Mock Ollama and an OpenAI-compatible server at the HTTP boundary, including
  auth failures, timeouts, malformed responses, tool-call payloads, and
  privacy-sensitive request bodies.
- Test approval and denial flows through the same runtime path used by CLI
  tool calls. Confirm no external call occurs after denial or while offline.
- Cover IPv6 addresses, sensitive argument redaction for every tool origin,
  and code-task final-checkpoint and failure-flow behavior.

**Acceptance:** all lifecycle tests run without a live Ollama service or
external network; the tests prove request contents, persisted state, and
side-effect boundaries rather than only testing helper functions.

## P1 — Reliability and privacy

### Complete resource-lifecycle coverage

- Verify cleanup under keyboard interrupt, startup failures, unexpected
  exceptions, and MCP server process failures; add coverage for repeated CLI
  invocations in one process.

**Acceptance:** lifecycle tests verify each resource closes exactly once on
all exit paths, no server process is leaked, and one failed MCP server does not
prevent other configured servers from shutting down cleanly.

### Make code-task setup and verification explicit

- Add explicit metadata-driven test commands, configured source paths for
  unattended use, and coverage for ignored/untracked file preservation.
- Report copy exclusions and checkpoint results, preserve ignored/untracked
  user files, and guarantee test or commit failures are never presented as
  verified completion.

**Acceptance:** tests cover new and copied projects across supported runners,
source preservation, ambiguous/headless selection, excluded secrets,
ignored files, multiple test commands, failed tests, and final status.

### Add focused, safer project-inspection and editing tools

- Add dedicated workspace-bounded file-pattern search, text search, and (where
  practical) symbol search so routine exploration does not require shell
  execution.
- Add a patch preview/apply flow and a workspace diff tool. Preview exact
  changes before applying them; preserve the existing path validation and
  snapshot-backup behavior for modifications.
- Add explicit file rename and delete operations only with path validation,
  collision checks, backups or recoverability where applicable, and
  confirmation before destructive changes.
- Add read-only Git status, diff, log, and branch inspection for ordinary
  workspace sessions. Keep code-task checkpoint/finalize operations scoped to
  the active isolated task and retain their test gates.
- Consider a bounded, allowlisted way to run a selected test, lint, or build
  check. Do not expose an unbounded host command path or weaken the existing
  Bubblewrap isolation and approval rules.

**Acceptance:** tests prove search and all file operations stay inside the
workspace (including symlink and traversal cases); patch previews match the
applied diff; destructive operations honor approval and preserve recoverable
state; Git inspection is read-only; and verification tools enforce time,
output, and sandbox limits.

### Bound memory by useful context and provide retention controls

- Add search across older conversation history and summaries, plus selective
  session or memory-item deletion. Keep retrieval scoped to the local SQLite
  store and make deletion targets reviewable before confirmation.
- Make the token budget model-aware where metadata is available, bound system
  prompt/output tokens, and test retention/deletion behavior end-to-end.
- Keep memory summaries bounded, session-scoped, and isolated between users
  when multiple operating-system accounts or profiles are supported.

**Acceptance:** long-session tests stay within the configured context budget;
recent instructions and citations remain available; retention and deletion
remove exactly the documented records; memory search returns bounded,
session-aware results and selective deletion cannot affect unrelated sessions.

### Make RAG indexing recoverable and bounded

- Test recovery if the Chroma database itself is corrupt, fully transactional
  partial vector updates, deletion/modified-source edge cases, and citations
  through the live CLI.

**Acceptance:** tests cover size limits, corrupted state, partial vector-store
failures, recovery/rebuild, deleted and modified files, and source citations.

### Complete provider and capability verification

- Add mock-server coverage for rate-limits, retries, timeouts, tool calls,
  malformed responses, and redaction. Verify raw API keys never appear in
  logs, config, SQLite, or failure output.
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

- Continue enforcing a consistent invocation policy with bounded execution,
  safe output handling, and auditable approvals. Verify each transport,
  isolation, and shutdown path through integration tests.

**Acceptance:** integration tests cover each supported transport, malformed
configuration, schema conflicts, invocation approval, timeout, server
failure, and clean shutdown.

### Add and document optional service integrations

- Add optional integrations such as GitHub or databases as user-configured MCP
  servers or narrowly scoped adapters; document that each integration is
  available only after it has been configured and loaded.
- Defer browser automation and computer-control tools until there is a concrete
  need and a design for domain restrictions, credential isolation, per-action
  approval, and bounded execution. Prefer explicit user consent and visible
  actions over unattended browsing or control.

**Acceptance:** each shipped integration has documented setup, required
permissions, data sent outside the machine, failure behavior, and integration
tests; browser/computer-control features remain disabled until their security
boundary and consent UX are verified.

### Establish repeatable release engineering

- Run the existing GitHub Actions workflow on the declared Python versions
  (3.10 and 3.14), inspect failures, and fix compatibility issues. Treat Linux
  as the only supported OS until equivalent runtime/isolation behavior is
  verified elsewhere.
- Extend automated release checks only for gaps that remain: consider
  formatting/type checks and dependency/vulnerability monitoring after the
  blocking correctness work is covered.
- Document supported platforms, upgrade and backup/restore procedures,
  logging controls, and recovery from corrupt SQLite databases or vector
  indexes.

**Acceptance:** clean artifact installation and CI pass on every declared
Python/platform combination; supported platforms and operational recovery
procedures are documented; CI tests do not require a local Ollama daemon or
live Internet services.

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
