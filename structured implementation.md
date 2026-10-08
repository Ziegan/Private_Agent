l# Structured Implementation Plan

## Purpose and relationship to the roadmap

This document turns [IMPLEMENTATION_ROADMAP.md](./IMPLEMENTATION_ROADMAP.md)
from a target-state vision into a dependency-ordered delivery plan. The
roadmap describes where Private Agent should go; this plan describes how to
restructure the current implementation and deliver that direction in
independently verifiable phases.

The plan is intentionally incremental. It does not assume that the current
code already implements future roadmap capabilities, and it does not call for
a one-time rewrite. Each phase must leave the application runnable, preserve
supported behavior, and expose stable contracts that later phases can build
on. Dates and performance targets are not pre-assigned: capture a baseline
first, then set measurable acceptance limits from representative workloads.

“Without a roadblock” means that dependencies, interfaces, migration paths,
and quality gates are addressed before downstream feature work depends on
them. It cannot guarantee that unforeseen technical or product risks will
never arise; those must be surfaced at phase gates rather than passed silently
to the next phase.

## Delivery rules

1. **Preserve a working product at every phase boundary.** Keep a tested
   compatibility path for existing CLI entry points, configuration,
   databases, skills, RAG indexes, and permission behavior while their
   internals are being extracted or replaced.
2. **Stabilize contracts before parallel feature work.** Define the interfaces
   and ownership boundaries for providers, model capabilities, memory,
   retrieval, tools, policies, workflows, and lifecycle management before
   multiple implementations depend on them.
3. **Migrate in slices.** Introduce a new implementation behind an existing
   facade or adapter, compare its behavior with the old path, migrate one
   concern at a time, and remove the compatibility layer only after its
   consumers and persisted data are accounted for.
4. **Treat privacy and permissions as architecture.** All provider, tool,
   plugin, MCP, research, and computer-use paths must use shared policy
   boundaries. No new route may bypass consent, workspace isolation, or
   explicit user authorization.
5. **Budget resources at subsystem boundaries.** Bound model context,
   persistent and in-memory caches, queues, concurrent work, retries, file
   reads, index sizes, and subprocess execution. Close clients and release
   temporary resources deterministically, including on cancellation and
   failure.
6. **Measure before optimizing.** Use existing tests and benchmarks to record
   a baseline for representative startup, memory, repository indexing,
   retrieval, model/tool execution, and shutdown paths. Verify performance
   changes against the same workload and retain correctness tests.
7. **Make readiness evidence-based.** A phase is complete only when its exit
   criteria, migration/recovery behavior, tests, and documented operational
   behavior are verified. An unimplemented roadmap item is not represented
   as shipped.

## Dependency map

```text
Phase 0: Baseline and compatibility inventory
  -> Phase 1: Modular architecture and migration seams
    -> Phase 2: Configuration, dependency injection, plugin/lifecycle contracts
      -> Phase 3: Shared security, policy, and observability foundations
        +-> Phase 4: Provider platform
        +-> Phase 5: Repository intelligence
        +-> Phase 10: MCP ecosystem
        +-> Phase 11: Multi-agent workflow foundation
        Phase 4 + Phase 5 -> Phase 6: Context engine
        Phase 6 -> Phase 7: Memory evolution
        Phase 5 + Phase 7 (+ Phase 4 embeddings) -> Phase 8: Advanced
          retrieval and knowledge graph
        Phase 4 + Phase 8 -> Phase 9: Deep research
        Phase 5 + Phase 6 + Phase 11 -> Phase 12: Autonomous coding workflows
        Phase 3 + relevant integrations -> Phase 13: Enterprise security and
          observability completion
        Phase 3 + Phase 4 + Phase 13 -> Phase 14: Computer use
        Relevant completed capabilities -> Phase 15: Agent operating system
```

The graph shows required foundations, not a requirement to serialize every
task. After Phases 0–3 have established shared contracts, provider adapters,
repository indexing, and policy/telemetry integrations can proceed as separate
workstreams if they use those contracts and keep the main branch releasable.
Research depends on provider, evidence/provenance, and retrieval contracts.
The multi-agent foundation does not require MCP or research integrations to
ship: those roles plug in when available. Coding workflows use repository
intelligence and context ranking; graph-based impact analysis can improve them
later but is not a prerequisite for a first useful coding workflow. Computer
use can use direct browser/device adapters; MCP and multi-agent support remain
optional integrations, not gates. The agent operating system assembles only
the capabilities required by each workflow.

## Phase 0 — Establish the current-state baseline

**Roadmap alignment:** Preparation for every phase.

**Objective:** Record what exists today and make current behavior measurable
before structural changes begin.

**Phase status: IN PROGRESS.** Architecture, roadmap, compatibility, and
initial validation inventories are recorded below. Benchmark coverage is
incomplete, and reproducibility/failure follow-up remains open.

### Task status

- [x] **1. Map current architecture and resource ownership.** Completed;
  evidence is in the architecture map below.
- [x] **2. Classify roadmap capabilities against current code and tests.**
  Completed; evidence is in the capability inventory below.
- [x] **3. Identify compatibility and migration surfaces.** Completed;
  evidence is in the compatibility inventory below.
- [x] **4. Run documented baseline checks and record environment-specific
  failures.** Completed for this snapshot; results and caveats are recorded
  below. Completion means the checks were run and results classified, not
  that every existing check passes.
- [~] **5. Establish repeatable performance benchmark inputs and resource
  measurements.** Partially complete: one CLI bootstrap measurement was
  captured, but no labeled RAG corpus/query fixture or live Ollama embedding
  service was available for the documented retrieval benchmark.
- [ ] **6. Triage baseline test/lint failures and make test configuration
  isolation reproducible.** Added after observing different results between
  the user's home and an isolated home. Keep this distinct from feature work;
  do not silently fix production behavior as part of the inventory.

### Baseline snapshot

Captured 2026-10-08 in the repository at commit `ca2dd37`, branch `Phase_0`.
The working tree already contained untracked `.github/` and
`structured implementation.md` deliverables; they are not source behavior
included in the baseline. The package contains 45 Python source files and the
test suite contains 17 `test_*.py` files. The runtime used for the checks
reported Python 3.14.8, pytest 9.1.1, and Ruff 0.16.10. README documents Python
3.10–3.14 support (3.14 validated).

### Current architecture map

| Boundary | Current implementation and responsibility |
| --- | --- |
| Entrypoints | `main.py`, `src/private_agent/__main__.py`, and the `private-agent` script in `pyproject.toml` delegate to `private_agent.cli:main` in `src/private_agent/cli.py`. |
| Session orchestration | `src/private_agent/agent/runtime.py` owns interactive startup, provider/model choice, context assembly, turns, tool execution, task flows, and shutdown. `agent/__init__.py` is a compatibility facade. |
| Configuration | `src/private_agent/config.py` owns defaults, JSON load/create/migration, runtime aliases, and packaged prompt loading; configuration is currently centralized rather than split into the future roadmap's separate files. |
| Providers | `src/private_agent/agent/providers.py` holds provider-independent validation and Ollama model construction helpers; provider selection and capability orchestration also live in `agent/runtime.py`. Current choices are Ollama and OpenAI-compatible online endpoints. |
| Tools and policy | `src/private_agent/tools/` contains domain tools and the mutable `AVAILABLE_TOOLS` registry in `tools/__init__.py`. `agent/permissions.py` and runtime dispatch enforce session, network, coding, and device boundaries; MCP tools are loaded into the registry at runtime. |
| Persistent memory and plans | `src/private_agent/database/memory.py` owns SQLite schema/migrations, conversation history, episodic and learned memory, task plans, revisions, leases, and task events/actions. |
| Document retrieval | `src/private_agent/rag/` owns text chunking, Chroma indexing/state, BM25/vector retrieval, and citations. `agent/runtime.py` retrieves context and injects it into requests. |
| Skills | `src/private_agent/skills/` loads and manages Markdown skill profiles; current content format and registry are file-based. |
| Workspace and code tasks | `src/private_agent/sandbox/linux.py` validates workspace paths and shell risk; `src/private_agent/code_tasks/` creates isolated workspaces, checkpoints, and bounded test runs. Linux sandbox features require `bwrap` and `prlimit`, both present during this baseline. |
| Tests | `tests/unit/` and `tests/integration/` exercise modules and the assembled runtime. `tests/conftest.py` supplies temporary SQLite/workspace fixtures; tests mock model/service behavior. |

This map describes current ownership, not a recommendation to preserve every
current coupling. The large runtime and dynamic global tool registry are
notable Phase 1 extraction seams.

### Roadmap capability inventory

Status describes the current implementation relative to each roadmap target:
**partial** means related capability exists but the target scope is not met;
**planned** means no corresponding end-to-end capability was identified.

| Roadmap phase | Status | Current evidence and remaining target gap |
| --- | --- | --- |
| 0 — Foundation stabilization | Partial | Categorized JSON configuration, provider helpers, and separated domain packages exist. Configuration remains centralized, runtime orchestration is heavily coupled, and there is no general dependency-injection/plugin contract. See `config.py`, `agent/runtime.py`, `agent/providers.py`, and `tools/__init__.py`. |
| 1 — Universal provider layer | Partial | Ollama and OpenAI-compatible endpoints, capability handling, and provider tests exist. LM Studio, Anthropic, Gemini, Azure, OpenRouter, and a provider-neutral embeddings/models interface are not implemented as interchangeable adapters. See `agent/providers.py`, `agent/runtime.py`, and `tests/unit/test_providers.py`. |
| 2 — Repository intelligence | Partial | Workspace file/text/symbol search and Git inspection support coding tasks. Persistent AST, symbol, dependency/call graph indexing, and architecture discovery are not present. See `tools/filesystem.py` and `tests/unit/test_tools.py`. |
| 3 — Advanced context engine | Partial | Token accounting, context limits, history compaction, and bounded memory/RAG context exist. Repository-aware ranking across dependencies/tests and layered provenance-preserving context are not a complete shared engine. See `agent/compaction.py`, `agent/prompts.py`, and `tests/unit/test_compaction.py`. |
| 4 — Advanced memory | Partial | SQLite history, episodic summaries, user-confirmed learned items, skills, and task state exist. Explicit episodic/semantic/procedural/repository memory contracts with unified provenance and retrieval are not complete. See `database/memory.py` and `tests/unit/test_database.py`. |
| 5 — Deep research | Partial | Consent-gated web search/fetch and research citations in planning exist. Multi-source claim evidence, confidence/verification state, contradiction handling, and structured research reports are not complete. See `tools/web.py`, `agent/permissions.py`, and `tests/integration/test_agent_runtime.py`. |
| 6 — Multi-agent architecture | Planned | Planner and coding task modes exist, but the specialized planner/explorer/researcher/developer/reviewer/tester/architect workflow with delegated budgets and handoffs was not identified. See `agent/runtime.py` and runtime integration tests. |
| 7 — Autonomous coding workflows | Partial | Approved plans, isolated workspaces, checkpoints, and bounded project tests exist. Repository-intelligence-driven impact analysis, advisory refactoring analysis, and a complete review/test agent workflow are absent. See `code_tasks/`, `agent/permissions.py`, and `tests/unit/test_code_tasks.py`. |
| 8 — Enterprise security | Partial | Workspace boundaries, tool approval, network validation/consent, and Linux sandboxing exist. The roadmap's secret scanning, security scan integration, general policy engine, and configurable audit coverage are not complete. See `sandbox/linux.py`, `agent/permissions.py`, `tools/network.py`, and `tests/unit/test_network.py`. |
| 9 — Enterprise observability | Partial | Debug run logs and session/token/tool statistics exist. OpenTelemetry-compatible traces/metrics and operational export/monitoring are not implemented as a complete observability layer. See `run_logging.py`, `agent/session_stats.py`, and `tests/unit/test_session_stats.py`. |
| 10 — MCP ecosystem | Partial | Configured stdio and remote HTTP MCP tools and a trusted-tool auto-approve list exist. A managed marketplace/distribution lifecycle and complete per-tool Allow/Deny/Prompt administration are not present. See `tools/mcp.py`, `tools/__init__.py`, and `tests/unit/test_tools.py`. |
| 11 — Advanced RAG | Partial | Local document indexing, configurable chunking, Chroma plus BM25 retrieval, citations, and refresh exist. Repository symbols and external knowledge sources are not unified in the RAG pipeline. See `rag/` and `tests/unit/test_rag.py`. |
| 12 — Knowledge graph | Planned | No typed graph index or graph-query service for files, symbols, services, tables, or APIs was identified. |
| 13 — Computer use | Partial | Local image/video, webcam, microphone, and transcription tools exist with approval/capability checks. Browser automation and general desktop control are not implemented. See `tools/media.py` and `tests/unit/test_media.py`. |
| 14 — Agent operating system | Planned | The current application is a CLI assistant with agent, memory, RAG, tools, and coding-task features; the unified workflow platform and enterprise operations surface are not implemented. |

These classifications are an initial inventory, not feature certification;
revisit them as the relevant code and tests evolve.

### Compatibility and migration inventory

- **Invocation/package:** Preserve `python main.py`, `python -m
  private_agent`, the `private-agent` console script, and package discovery
  under `src/`; changing these affects installation and automation.
- **Configuration:** Preserve `~/.private_agent.conf`, categorized settings,
  accepted legacy flat keys, default merging, secure file permissions, and
  the packaged default system prompt. Tests: `tests/unit/test_config.py`.
- **SQLite:** Preserve existing database files, tables, migration behavior,
  transactions, task recovery data, and backup/restore expectations. Do not
  replace or destructively migrate the user's memory store. Tests:
  `tests/unit/test_database.py`.
- **CLI/session:** Preserve slash/session commands, permission prompts,
  provider selection, summary and task workflows, and observable failure
  behavior. Tests: `tests/unit/test_interactive.py` and
  `tests/integration/test_agent_runtime.py`.
- **Tools/policies:** Treat tool names, Pydantic schemas, outputs, permission
  modes, network-consent prompts, local-only device behavior, and MCP
  configuration as compatibility surfaces. Tests:
  `tests/unit/test_tools.py`, `tests/unit/test_network.py`,
  `tests/unit/test_location.py`, and runtime integration tests.
- **Skills:** Preserve Markdown skill files, identifiers, registration, and
  retention semantics. Tests: `tests/unit/test_skills.py`.
- **RAG:** Preserve configured document/index paths, supported source types,
  Chroma state, chunk metadata/citations, maintenance behavior, and refresh
  semantics. Tests: `tests/unit/test_rag.py` and
  `tests/unit/test_chunking.py`.
- **Coding workspace:** Preserve isolated-copy defaults, direct-workspace
  approvals, checkpoint behavior, command/test budgets, and Linux's
  fail-closed `bwrap`/`prlimit` prerequisites. Tests:
  `tests/unit/test_code_tasks.py` and runtime integration tests.
- **Dependencies/platform:** `pyproject.toml` is the runtime dependency source
  of truth; `requirements.txt` is generated by `scripts/sync_requirements.py`.
  Preserve Python 3.10+ metadata and the README's supported 3.10–3.14 range.

### Validation and performance baseline

Checks were run with the project virtual environment and an isolated writable
home where noted:

| Check | Result |
| --- | --- |
| `pytest -q` with the existing home | 451 passed, 5 failed, 1 warning, 32.18 s. Three failures concern online location/microphone availability and disappear with isolated home configuration. |
| `HOME=/tmp/private-agent-phase0-cleanhome ./.venv/bin/python -m pytest -q` | 454 passed, 2 failed, 1 warning, 21.67 s with a writable isolated default config. Reproducible failures: `test_sync_connect_backend_falls_back_to_second_validated_address` and `test_runtime_bounds_assembled_long_session_to_model_context`. The two tests were rerun together under isolated home and both failed again. |
| `ruff check src tests --select F` | Fails on one F401 unused import at `tests/unit/test_network.py:903`. |
| `python scripts/sync_requirements.py --check` | Pass. |
| `python -m private_agent --help` | Pass. One process-level sample: 0.09 s elapsed, 18,904 KB maximum RSS. This measures CLI help/bootstrap only, not an agent session. |
| `python scripts/benchmark_rag.py --help` | Pass. The actual benchmark did not run: no labeled benchmark corpus/query fixture was identified and the local Ollama endpoint at `127.0.0.1:11434` was unavailable. |

The home-dependent results mean tests are not yet hermetic with respect to
per-user configuration: three failures occurred only with the existing home
configuration, while two persisted with a clean default config. The two
isolated-home failures and the Ruff finding are recorded baseline issues; they
have not been changed or attributed to future implementation work. The CLI
measurement is a single sample and is not a performance target. No meaningful
RAG, repository-indexing, model/tool, or memory-throughput benchmark has been
established yet.

### Exit gate

Close Phase 0 only after task 5 has repeatable inputs and measurements for
available representative paths, task 6 has a documented triage/isolation
outcome, and the baseline can be rerun consistently. Existing failures need
an explicit classification and regression reference; they need not be
silently rewritten as passing.

## Phase 1 — Restructure around explicit architectural boundaries

**Roadmap alignment:** Roadmap Phase 0, “Foundation Stabilization.”

**Objective:** Reduce runtime coupling while preserving existing user-facing
behavior and avoiding a disruptive rewrite.

**Phase status: IN PROGRESS.** The current boundaries are documented and the
first provider-construction slice has been extracted. Broader orchestration,
contract, and lifecycle work remains.

### Task status

- [x] **1. Define current subsystem ownership boundaries.** Completed as an
  initial map in Phase 0. The map highlights `agent/runtime.py` and the
  mutable tool registry as coupling seams; ownership still needs refinement
  as extractions proceed.
- [~] **2. Extract cohesive behavior from the runtime.** In progress. Local
  model construction was already delegated through `agent/providers.py`;
  this change also moved OpenAI-compatible chat-model construction there.
  Runtime retains `_make_online_chat_model` as a compatibility/configuration
  facade. Runtime still owns provider selection, capability policy and most
  session orchestration.
- [~] **3. Add typed subsystem contracts.** Partially complete. Provider
  construction now takes explicit typed factory/configuration parameters
  rather than constructing policy clients inline in the session runtime.
  Shared typed request/result/capability/error contracts have not yet been
  established; model objects still use backend SDK types.
- [~] **4. Clarify resource ownership and cleanup.** Partially complete.
  The extracted builder receives client factories explicitly; HTTP clients
  are registered with the network subsystem and closed by runtime session
  shutdown. A unified lifecycle owner and complete startup/cancellation/
  partial-initialization guarantees across all subsystems remain future work.
- [~] **5. Preserve public imports and entry points during migration.**
  Partially complete. The existing runtime helper remains callable and
  delegates to the provider boundary; targeted online-session tests and
  `python -m private_agent --help` pass. The full compatibility surface still
  needs systematic coverage as more modules are extracted.

### Implemented slice: OpenAI-compatible model construction

`agent/providers.py` now owns construction of OpenAI-compatible chat models
through `create_openai_compatible_chat_model`. The runtime supplies the model
factory, secure synchronous/asynchronous HTTP client factories, and normalized
configuration. This keeps provider-specific model construction together with
local provider helpers, makes its collaborators test-injectable, and leaves
runtime as the existing call facade. The change preserves model name,
endpoint, session key, sampling settings, request/retry/token limits, and
loopback-aware policy transports. The existing network subsystem remains
responsible for tracking and closing outbound clients.

This is one bounded extraction, not a claim that the entire runtime has become
provider-agnostic or that provider/session resource lifecycle is complete.

### Validation record — 2026-10-08

- Focused provider and online-session validation: **6 passed** across
  `tests/unit/test_providers.py`, online model construction/model listing, and
  online microphone-tool gating.
- Full suite with a clean isolated home: **455 passed, 2 failed, 1 warning**
  in 21.62 seconds. The same two failures were present in the Phase 0 baseline:
  `test_sync_connect_backend_falls_back_to_second_validated_address` and
  `test_runtime_bounds_assembled_long_session_to_model_context`.
- Ruff `--select F` on the changed Python files: passed.
- `python -m private_agent --help`: passed, preserving the module entry point.
- `python scripts/sync_requirements.py --check`: passed; no dependencies
  changed.
- Generated provider-builder syntax check for Python 3.10: passed.

**Exit gate:** Existing CLI flows and supported imports continue to work;
responsibilities are documented at module boundaries; tests cover the
extracted behavior; lifecycle cleanup is deterministic on success, error,
interrupt, and cancellation.

## Phase 2 — Configuration, dependency injection, and extension contracts

**Roadmap alignment:** Remaining Roadmap Phase 0 foundation work.

**Objective:** Make implementations replaceable without scattering provider,
storage, or policy conditionals through the runtime.

**Phase status: IN PROGRESS — NOT COMPLETE.** SQLite version tracking and
provider-neutral/plugin contracts are implemented and tested. Validated
configuration, configuration schema migration, application-level dependency
construction, and adoption of shared contracts across existing subsystems
remain unfinished.

### Task status

- [ ] **1. Define validated configuration models and a normalized runtime
  configuration interface.** Not started. Current `config.py` keeps
  categorized JSON defaults and legacy-key migration, but runtime still
  consumes module-level aliases and permissively falls back for invalid
  values. Keep current file format and mode bits as compatibility requirements
  for this work.
- [~] **2. Establish versioned, recoverable configuration and SQLite
  migrations.** Partially complete. `PersistentMemory` now records SQLite
  schema version 1 in `PRAGMA user_version`, applies the existing
  introspective/idempotent schema migration before setting the version, and
  refuses to open a database with a newer version. Tests cover legacy upgrade,
  future-version refusal without mutation, and failed migration rollback.
  Configuration has no explicit schema version yet, and SQLite migrations
  have not been refactored into individually named version-to-version steps.
- [~] **3. Introduce application/session dependency construction.**
  Partially complete. Phase 1's provider builders accept explicit model and
  policy-client factories; `PluginContext` defines explicit config/services/
  logger injection for future extensions. The application does not yet
  compose and inject provider, memory, retrieval, tool-registry, policy,
  clock, and telemetry dependencies, and mutable module globals remain.
- [x] **4. Specify plugin contracts.** Contract definitions are in
  `src/private_agent/plugins.py`: validated name/version/kind, declared
  capabilities and permissions, resource-budget fields, config validation,
  and async initialization/shutdown hooks. Discovery/loading is deliberately
  not added; the contract states that hosts must enforce budgets/permissions
  and isolate a failed plugin rather than enabling arbitrary code by default.
- [x] **5. Define stable shared boundary types.** Provider-neutral types are
  in `src/private_agent/contracts.py` for model messages/requests/responses,
  capability discovery, token usage, tool calls/results, evidence, memory
  records, and structured execution errors. Current provider/runtime/storage
  paths have not yet migrated to consume these types.

### Implemented slices

- **SQLite schema version guard:** `database/memory.py` accepts legacy
  version-0 databases, performs the existing transactional schema
  reconciliation, and sets version 1 only after successful initialization.
  Newer versions fail explicitly before application tables are changed.
- **Contract foundation:** Added typed immutable shared value objects and
  declarative plugin manifest/context/protocol types. The plugin module
  intentionally provides no discovery, import, or execution mechanism.

### Validation record — 2026-10-08

- Focused contract and database tests: **61 passed, 1 dependency deprecation
  warning**.
- Full suite under a clean isolated home: **460 passed, 2 failed, 1 warning**
  in 21.56 seconds. Both failures match the Phase 0 baseline:
  `test_sync_connect_backend_falls_back_to_second_validated_address` and
  `test_runtime_bounds_assembled_long_session_to_model_context`.
- Ruff `--select F` on the changed implementation and test files: passed.
- Repository-wide Ruff check still reports the pre-existing F401 at
  `tests/unit/test_network.py:903`; it is outside this phase's changes.
- Python syntax checks passed for the new contract/plugin modules, database
  code, and associated tests. Pylance diagnostics for `contracts.py` and
  `plugins.py` are clear; its reported optional-usage issue in the new test
  was corrected with an explicit non-None assertion.
- Requirements synchronization and CLI entry-point checks passed in Phase 1;
  Phase 2 changes add no dependencies or entry-point changes.

**Completion gate:** Do not mark Phase 2 complete until task 1 has a validated
configuration model used through a normalized interface; task 2 versions
configuration and provides explicit, restart-safe database migration steps;
task 3 injects the required services at the application/session boundary and
tests with fakes without patching internal globals; and plugin
failure/budget/permission behavior is enforced by a host. Shared contracts
must have a migration path into the implementations that consume them. The
application must remain runnable, preserve legacy config/database behavior,
and isolate extension failures without disabling unrelated built-ins.

## Phase 3 — Shared security, policy, and observability foundations

**Roadmap alignment:** Foundational subset of Roadmap Phases 8 and 9, moved
earlier so later capabilities cannot bypass these controls.

**Objective:** Make authorization and operational evidence consistent before
adding more providers, plugins, agents, or external integrations.

**Work sequence:**

1. Define a central policy decision contract for read, write, execute,
   network, device, data-sharing, and MCP operations. Preserve the current
   session permission modes and coding workspace levels as explicit policy
   inputs, not interchangeable shortcuts.
2. Define policy enforcement points at tool dispatch, provider egress,
   repository access, MCP calls, and device/browser actions. Return structured
   allow/deny/needs-approval decisions with a user-readable reason.
3. Establish secret-safe structured events for session, model, tool, approval,
   task, retrieval, and failure lifecycle events. Do not log prompts, document
   contents, credentials, or sensitive tool arguments by default.
4. Add bounded metrics and correlation identifiers for latency, token usage
   when available, tool outcomes, retries, task completion, and resource
   pressure. Telemetry is local and opt-in/configurable where it could leave
   the machine.
5. Add common timeout, cancellation, retry, concurrency, and output-size
   policy primitives. Retries must be bounded and restricted to operations
   whose repetition is safe.

### Implemented observability and recovery slice

- The private run log now records structured JSONL events with timestamps,
  run identifiers, source context, and traceback details while avoiding raw
  prompts, credentials, and tool content.
- Provider discovery, model initialization, turn failures, tool outcomes,
  task leases, and shutdown failures emit traceable events.
- Empty streaming or non-streaming model responses receive one bounded retry
  within the existing turn deadline; persistent failure is reported in-session.

Phase 3 remains **in progress**: central policy decisions, common enforcement
points, and consistent resource/metrics contracts remain part of the exit gate.

**Exit gate:** Existing permission behavior has regression coverage; every
side-effecting capability reaches a policy decision; events can trace a task
without exposing private content; timeouts, cancellation, and resource bounds
are consistently observable.

## Phase 4 — Universal provider platform

**Roadmap alignment:** Roadmap Phase 1.

**Dependencies:** Phases 1–3.

**Objective:** Add providers through adapters rather than provider-specific
branches in orchestration.

**Work sequence:**

1. Finalize contracts for chat/streaming, embeddings, model listing,
   capability discovery, health/errors, timeouts, cancellation, and usage
   metadata. Mark unsupported operations explicitly.
2. Move current Ollama and OpenAI-compatible behavior behind adapters first;
   treat preserving current behavior as the compatibility acceptance test.
3. Normalize capability data for tool calling, structured output, vision,
   audio, reasoning, and computer use. Distinguish declared, detected, and
   unknown capabilities; do not infer support from a model name alone.
4. Implement provider selection and credentials through validated
   configuration and secure runtime input. External endpoints must pass URL,
   transport, and egress policy checks.
5. Add new providers in separate adapter slices (LM Studio and hosted
   providers such as OpenAI, Anthropic, Gemini, Azure, and OpenRouter), each
   with contract tests, capability tests, error mapping, and documented
   privacy/data-flow behavior.
6. Add conformance tests so provider switching does not require changes to
   agent orchestration.

### Implemented local discovery slice

- Linux/macOS startup probes configured loopback Ollama and OpenAI-compatible
  endpoints with bounded concurrency and per-endpoint timeouts. It reports
  only reachable runtimes and their advertised chat models.
- Windows skips local discovery and defaults to online mode. When no local
  runtime is found elsewhere, online mode remains available by default.
- Online startup explicitly offers saved configuration or manual URL/key entry;
  local-compatible runtimes use the existing policy-controlled client factory.

This is a partial provider-platform slice, not adapter conformance: provider
contracts, normalized adapter lifecycle, and hosted-provider integrations
remain open.

**Exit gate:** Existing Ollama and OpenAI-compatible sessions retain supported
behavior; adapter conformance tests pass; unsupported provider capabilities
degrade explicitly; credentials do not leak into logs or persistent config
unless the user explicitly chooses supported storage.

## Phase 5 — Repository intelligence

**Roadmap alignment:** Roadmap Phase 2.

**Dependencies:** Phases 1–3; provider embeddings are optional and not
required for deterministic symbol indexing.

**Objective:** Build a bounded, incremental repository model so later context,
retrieval, coding, and graph features can reuse the same facts.

**Work sequence:**

1. Define a repository index interface and metadata schema for repository
   identity, file fingerprints, parser/version, symbols, imports, references,
   and diagnostics.
2. Build safe file discovery with ignore rules, symlink policy, file-size and
   corpus limits, encoding handling, and workspace-root enforcement.
3. Add language-aware AST/syntax adapters incrementally. Start with the
   repository's primary languages; use explicit partial/unsupported results
   for other files rather than pretending parsing succeeded.
4. Index files, symbols, imports, dependencies, and supported call/reference
   edges. Keep unresolved and dynamic relationships marked as uncertain.
5. Update incrementally from fingerprints or filesystem events with a
   full-rebuild path, corruption detection, index versioning, and cancellation.
6. Expose architecture summaries and symbol/dependency queries through a
   stable service API and tools that respect existing workspace policy.

**Exit gate:** Repeated queries reuse the index rather than rereading the
repository; changed/deleted files are reflected correctly; full rebuild and
incremental results agree on test repositories; indexing is bounded and
reports parser gaps and timing.

## Phase 6 — Advanced context engine

**Roadmap alignment:** Roadmap Phase 3.

**Dependencies:** Phases 3–5; memory summaries from Phase 7 may later be
included through the same context-item contract.

**Objective:** Select relevant evidence under a deterministic model-context
budget and reduce redundant rereads.

**Work sequence:**

1. Define typed context items with source, timestamp/version, token estimate,
   trust level, relevance, and citation or repository identity.
2. Build a ranking pipeline for task-related files, similar implementations,
   dependencies, tests, user request, repository architecture, and optional
   memory/retrieval results.
3. Allocate explicit budgets across global, project, task, and file context.
   Preserve recent/raw evidence alongside summaries when correctness requires
   exact text.
4. Make compression bounded and attributable: summaries must identify
   provenance, truncation, and uncertainty; retain access to the source for
   verification.
5. Cache only by stable inputs and invalidate on file, config, model/tokenizer,
   index, or task changes. Bound cache size and lifetime.
6. Instrument context selection, estimated/actual token counts, dropped items,
   compression cost, and cache behavior without logging sensitive content.

**Exit gate:** Context construction respects configured/model limits,
prioritizes verified relevant evidence, can recover source text for checks,
and outperforms the baseline on repeated repository questions without
regressing answer grounding.

## Phase 7 — Advanced memory system

**Roadmap alignment:** Roadmap Phase 4.

**Dependencies:** Phases 1–3 and the context-item contract from Phase 6.

**Objective:** Evolve existing SQLite history, episodic summaries, learned
preferences, task records, and skills into explicit memory types with bounded
retrieval and user control.

**Work sequence:**

1. Inventory existing SQLite records and migrations; define stable schemas for
   episodic, semantic, procedural, and repository memory, including source,
   scope, creation/update time, expiry, confidence, and review/deletion state.
2. Keep SQLite as the authoritative transactional store for structured
   records. Add vector storage only for retrieval use cases that need it; keep
   identifiers and lifecycle state reconcilable with the authoritative store.
3. Define memory write policy: what may be stored automatically, what needs
   user confirmation, what must never be stored, and how corrections,
   revocation, expiry, export, and deletion work.
4. Build retrieval with scope filters, relevance limits, token budgets, and
   provenance. Do not inject unverified remembered facts as current truth.
5. Migrate existing history/learning/task data without loss; support backup,
   restore, rollback, and interrupted-migration recovery.
6. Add retention, compaction, and maintenance jobs with bounded work and
   measurable database/index growth.

**Exit gate:** Existing records survive migration; users can inspect and
correct/delete supported memories; retrieval returns scoped, attributable,
bounded items; retention and restore are tested.

## Phase 8 — Advanced retrieval and knowledge graph

**Roadmap alignment:** Roadmap Phases 11 and 12, ordered together because
repository/entity relationships improve retrieval and change-impact queries.

**Dependencies:** Phases 5–7; provider embeddings from Phase 4 for vector
retrieval. Lexical and structured retrieval remain usable if embeddings are
unavailable.

**Objective:** Extend retrieval beyond document chunks to repository symbols,
services, APIs, tables, and typed relationships.

**Work sequence:**

1. Generalize ingestion contracts across local documents, repository files,
   wikis, tickets, database metadata, and approved APIs. Each connector must
   define identity, permissions, update/delete behavior, provenance, and
   resource limits.
2. Normalize records and citations so every result can be traced to a source,
   version, and location.
3. Add hybrid retrieval strategies (vector, BM25/lexical, structured symbol,
   and graph traversal) behind a common query interface. Keep ranking signals
   inspectable and tune with labeled evaluation queries.
4. Model graph nodes and edges for files, symbols, tests, services, tables,
   APIs, and dependencies. Preserve edge source and confidence; rebuild or
   incrementally update deterministically.
5. Add bounded graph queries for callers, dependencies, ownership, and likely
   impact. Enforce workspace and source permissions during both indexing and
   retrieval.
6. Compare retrieval quality and latency to existing RAG benchmarks; document
   connector-specific privacy and retention behavior.

**Exit gate:** Hybrid results include provenance and citations; the system
continues with supported non-vector retrieval when embeddings fail; graph
answers expose uncertainty and bounded traversal; updates and deletions
propagate across derived indexes.

## Phase 9 — Deep research engine

**Roadmap alignment:** Roadmap Phase 5.

**Dependencies:** Phases 3–4 and 8.

**Objective:** Produce evidence-led reports across local and user-approved
external sources.

**Work sequence:**

1. Define a research task model: question, scope, sources allowed, budget,
   deadline, evidence requirements, and completion status.
2. Add source connectors one at a time (web, documentation, GitHub,
   StackOverflow, local documents) using shared network consent, provider
   egress, rate/size limits, and cancellation.
3. Store claims with source, retrieval time, quoted/located evidence,
   confidence rationale, and verification state. Separate facts, inference,
   and unresolved questions.
4. Add cross-source verification and contradiction detection; do not mark a
   fact verified merely because multiple results repeat the same source.
5. Render bounded reports with executive summary, findings, risks,
   recommendations, and citations that can be checked by the user.
6. Test offline behavior, consent denial, stale sources, malformed content,
   and partial retrieval explicitly.

**Exit gate:** Reports distinguish evidence from inference; every externally
derived claim has inspectable provenance; offline and denied-consent paths
remain useful and honest; source and task budgets are enforced.

## Phase 10 — MCP ecosystem

**Roadmap alignment:** Roadmap Phase 10.

**Dependencies:** Phases 2–3, especially the shared egress and authorization
contracts. It can be implemented in parallel with provider adapters.

**Objective:** Expand current MCP support into a managed integration system
without turning integrations into a permission bypass.

**Work sequence:**

1. Formalize server manifests, transport configuration, versioning,
   health/status, startup/shutdown, and tool schema validation.
2. Enforce per-server and per-tool Allow/Deny/Prompt policies, with explicit
   data-egress summaries and auditable approval decisions. Preserve the
   existing trusted-tool allow-list as a migration input, not a default
   blanket trust rule.
3. Isolate server failures and resource budgets; bound tool results, calls,
   concurrency, timeouts, and retries.
4. Add signed/verified or otherwise trusted distribution and update controls
   before adding a marketplace or one-click installation.
5. Integrate GitHub, Jira, Slack, cloud, and database servers in separate
   opt-in connector slices with least privilege and secret-safe configuration.

**Exit gate:** Every MCP call is attributable to a server and tool, authorized
by policy, bounded, and visible to the user; server failure does not corrupt
the session or authorize another server implicitly.

## Phase 11 — Multi-agent architecture

**Roadmap alignment:** Roadmap Phase 6.

**Dependencies:** Phases 2–6, especially shared task budgets, policy checks,
cancellation, and observability. Research and MCP capabilities from Phases 9
and 10 are optional role integrations, not prerequisites for the workflow
foundation.

**Objective:** Introduce specialized agents as bounded workflow components,
not as independent processes with ungoverned access.

**Work sequence:**

1. Define a typed agent/task contract for input, output, delegated scope,
   evidence, tool allow-list, budget, deadline, cancellation, and handoff.
2. Implement planner, explorer, researcher, developer, reviewer, tester, and
   architect roles incrementally; begin with sequential orchestration and
   explicit handoffs.
3. Pass only the minimum required context and permissions to each role.
   Delegated agents inherit the parent task's policy ceiling and cannot grant
   themselves broader access.
4. Bound recursion, fan-out, active agents, model/tool calls, memory/context,
   and total wall-clock time. Cancel children and close resources when the
   parent task ends.
5. Persist resumable workflow state with idempotent step boundaries and
   reconciliation for uncertain side effects.
6. Compare task quality, latency, and resource use against the single-agent
   baseline; enable parallel execution only where measurements and isolation
   support it.

**Exit gate:** A failed or interrupted child can be reported and resumed or
cancelled safely; permissions and budgets are enforceable across delegation;
workflow outputs include evidence and per-agent status.

## Phase 12 — Autonomous coding workflows

**Roadmap alignment:** Roadmap Phase 7.

**Dependencies:** Phases 5–6, 11, and the security foundation in Phase 3.
Graph impact queries from Phase 8 are a later enhancement, not a blocker for
architecture-aware coding based on repository symbols and dependencies.

**Objective:** Make coding tasks architecture-aware, impact-aware, reviewable,
and verifiable before expanding autonomy.

**Work sequence:**

1. Before editing, inspect task-relevant architecture, affected symbols,
   dependencies, and tests using repository intelligence; record a compact
   impact set and its evidence.
2. Build a change plan that names files, expected behavior, risks, tests, and
   rollback/checkpoint strategy. Preserve approved-plan gating and workspace
   access selection.
3. Implement changes in an isolated copy by default where supported. Keep
   real-workspace modes explicit and policy-gated; never silently downgrade
   isolation.
4. Add refactoring analysis for dead code, duplication, and architecture
   violations as advisory findings first. Require evidence and avoid unsafe
   automatic rewrites.
5. Run configured project tests and validation commands through the sandbox
   with time, output, and resource limits; feed failures into bounded
   repair/retest cycles.
6. Require reviewer/tester stages and a final summary of changed files,
   validation evidence, unresolved risks, and checkpoint/restore options.

**Exit gate:** The agent can explain the impact scope before editing, preserve
approval and isolation policies, verify changes with bounded commands, and
report precisely what did and did not pass.

## Phase 13 — Enterprise security and observability completion

**Roadmap alignment:** Completion of Roadmap Phases 8 and 9.

**Dependencies:** Phase 3 foundations and all integrations intended for
production use.

**Objective:** Harden the accumulated system against sensitive-data exposure,
policy drift, operational blind spots, and unsafe extension behavior.

**Work sequence:**

1. Add secret detection/redaction for supported inputs, outputs, logs, and
   repository operations; test common key/token/password formats and false
   positives without storing detected secrets.
2. Add security scanning (for example Bandit and Semgrep) and define
   repository-specific rules, triage ownership, and documented suppressions.
3. Validate policy coverage across providers, tools, plugins, MCP, research,
   multi-agent delegation, and computer-use adapters. Test deny-by-default
   behavior for unknown capabilities.
4. Complete audit records for prompts/approvals/tool use/changes only under
   explicit retention and privacy settings; separate operational metadata
   from content and support safe deletion.
5. Add OpenTelemetry-compatible instrumentation and optional Prometheus/
   Grafana export only behind explicit configuration and data-flow disclosure.
6. Conduct threat modeling and failure-injection tests for provider compromise,
   malicious tool output, corrupted indexes/databases, interrupted
   migrations, exhausted budgets, and partial shutdown.

**Exit gate:** Security controls for the integrations present have
negative-path tests; audit and metric export do not leak content by default;
findings have an agreed disposition; the system's operational health and
resource use can be diagnosed locally. Repeat the relevant checks whenever a
later phase adds a new integration or execution surface.

## Phase 14 — Computer use

**Roadmap alignment:** Roadmap Phase 13.

**Dependencies:** Phases 3, 4, and 13. MCP and multi-agent orchestration are
optional ways to compose computer-use capabilities, not prerequisites for a
bounded browser/device adapter.

**Objective:** Add browser and desktop automation only after policy,
capability, audit, and cancellation contracts are established.

**Work sequence:**

1. Specify supported platforms and supported automation actions before
   selecting libraries such as Playwright or desktop-control tooling.
2. Separate browser sessions/profiles and device resources from user sessions;
   define data retention, download paths, network boundaries, and cleanup.
3. Require explicit per-task approval for navigation, form submission,
   downloads, and irreversible actions; distinguish observation from action.
4. Bind actions to a bounded task plan and verify visible outcomes rather than
   assuming clicks or keystrokes succeeded.
5. Add deterministic tests with local fixtures and mocked devices; never
   require real accounts or uncontrolled external websites in routine CI.

**Exit gate:** Device/browser permissions are per action or task as defined,
browser state and downloads are contained, cancellation closes resources, and
the user can inspect action history and outcomes.

## Phase 15 — Agent operating system

**Roadmap alignment:** Roadmap Phase 14.

**Dependencies:** All preceding phases needed by each advertised workflow.

**Objective:** Present coding, research, knowledge, automation, and operations
as coherent workflows over shared, mature platform services.

**Work sequence:**

1. Define workflow manifests for coding, research, knowledge, automation, and
   operations: required capabilities, permissions, inputs/outputs, budgets,
   and recovery behavior.
2. Compose existing agents, providers, memory, repository intelligence,
   retrieval, MCP, and computer-use integrations through the stabilized
   contracts rather than adding workflow-specific copies.
3. Provide capability discovery and graceful explanations when a provider,
   tool, or platform feature is unavailable.
4. Define enterprise deployment and administration only after local privacy,
   identity, policy, retention, and audit models are explicit.
5. Validate end-to-end workflows against representative tasks, resource
   budgets, failure scenarios, and platform support matrices.

**Exit gate:** Each advertised workflow has an explicit capability and
permission contract, measurable reliability/resource behavior, recovery
path, and end-to-end verification. The platform does not claim features that
are unavailable in the configured environment.

## Cross-phase quality gates

Apply these checks to each phase and to every implementation slice:

- **Compatibility:** Entry points, supported command behavior, config/data
  migrations, and documented privacy behavior remain valid or have an
  explicitly reviewed migration path.
- **Correctness:** Targeted unit/integration tests pass; the broader suite and
  lint checks are run when the change's risk or touched boundaries warrant it.
  Tests cover both success and denial/failure/cancellation paths.
- **Security:** Authorization is enforced at the execution boundary; path,
  network, device, and data-egress scopes are explicit; errors and logs do not
  disclose secrets or private content.
- **Performance:** Benchmark the relevant baseline and changed path with the
  same representative inputs. Report latency, throughput, memory/storage
  growth, and concurrency effects where relevant. Do not trade correctness,
  privacy, or bounded resource use for speed.
- **Resource lifecycle:** Work has explicit limits and deterministic cleanup;
  no unbounded queue, retry, cache, recursive delegation, or background
  process is introduced.
- **Recovery:** Persistent changes are versioned and recoverable; interrupted
  work can resume, roll back, or report an uncertain result without repeating
  unsafe side effects.
- **Documentation:** Update README, configuration reference, and operational
  guidance when supported behavior, setup, data flows, or platform
  prerequisites change.

## Current validation commands

The project's current documented commands are:

```sh
python -m pip install -e ".[dev]"
pytest -q
pytest tests/unit -q
pytest tests/integration -q
pytest tests/unit/test_database.py -q
pytest tests/unit/test_database.py::test_persistent_memory -q
ruff check src tests --select F
python scripts/sync_requirements.py --check
```

Use `IMPLEMENTATION_ROADMAP.md` for the full target feature list and README.md
for current user-visible behavior and operational setup. Keep this plan
current when phase dependencies, migration strategy, or completion criteria
change.
