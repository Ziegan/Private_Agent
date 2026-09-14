# Private Agent 🔐

A secure, privacy-first local AI agent built with LangChain and Ollama. Run powerful agentic AI entirely on your machine—no cloud, no API keys, no telemetry.

## ✨ Key Features

- **100% Local & Offline** — Runs entirely on your hardware using Ollama
- **Agentic Tool Loop** — LLM automatically calls tools (file I/O, shell, web search)
- **Persistent Memory** — SQLite stores sessions, episodic summaries, chat history
- **RAG Integration** — Hybrid vector (Chroma) + keyword (BM25) document retrieval
- **Sandboxed Execution** — Path validation, dangerous command detection, HITL snapshots
- **Skill Profiles** — Load custom markdown skill prompts to guide agent behavior
- **Model Context Protocol (MCP)** — Extensible plugin system for custom tools
- **GPU Acceleration (Optional)** — OpenVINO runtime for faster embedding/inference

## ⚡ Quick Start (5 Minutes)

### Prerequisites

- **Ollama** running locally ([https://ollama.ai](https://ollama.ai))
- Python 3.14+
- 4GB+ RAM (8GB+ recommended for larger models)

### Installation

```bash
# Clone the repository
git clone https://github.com/Ziegan/Private_Agent.git
cd Private_Agent

# Install dependencies
pip install -r requirements.txt
```

### Running the Agent

1. **Start Ollama** (in a separate terminal):
   ```bash
   ollama serve
   ```

2. **Run the agent**:
   ```bash
   python main.py
   ```

3. **On first run**, you'll be prompted for:
   - Knowledge base directory (optional, for RAG docs)
   - Skills folder path (optional, for skill profiles)
   - Workspace root (required, for file/command sandbox)
   - Model selection (pick from your locally installed models)

### First Chat

Select a model like `llama2` or `qwen2.5`, then try:

```
User: Search for FastAPI best practices and summarize them.

Agent: [Calls web_search] → [Calls fetch_webpage] → [Generates response with citations]
```

## 🏗️ Architecture Overview

### Components

| Module | Role |
|--------|------|
| `src/agent.py` | Async REPL loop, model selection, tool iteration, context window tracking |
| `src/tools.py` | 9 sandboxed tools: file I/O, shell, web search, SQLite access, MCP loading |
| `src/database.py` | SQLite persistent memory: chat history, session management, episodic summaries |
| `src/rag.py` | Hybrid RAG retriever: Chroma vector store + BM25 keyword search |
| `src/sandbox.py` | Path validation, HITL snapshots, dangerous command detection |
| `src/config.py` | Configuration loading (.private_agent.conf), offline environment enforcement |
| `src/skills.py` | Markdown skill profile matching and system prompt injection |

### Execution Flow

1. **Initialize** — Load config, spawn workspace sandbox, connect to Ollama, load RAG vectorstore
2. **Enrich Query** — Append episodic memory & RAG context to user input
3. **Invoke Model** — Send enriched prompt + tools to LLM
4. **Tool Loop (Async)** — Model returns tool calls → execute concurrently → feed results back to model
5. **Persist** — Save to SQLite; on exit, summarize session

## 🔒 Security & Sandboxing

All tools run in a restricted workspace with multiple layers of protection:

### Path Validation
- Tools can only access files within `workspace_root`
- Symlinks are resolved and validated to prevent escape attempts

### Dangerous Command Detection
- Blocks elevated privileges: `sudo`, `su`, `pkexec`, `doas`, `chmod 777`, `chown`
- Blocks destructive operations: `rm -rf`, `mkfs`, `dd if=`, fork bombs
- Blocks access to system paths: `/etc`, `/var`, `/usr`, `/bin`, `/sbin`, `/lib`, `/root`, `/sys`, `/proc`

### Human-in-the-Loop (HITL) Snapshots
- Before overwriting a file, a backup is created in `.agent_snapshots/`
- Timestamped backups enable rollback if the agent modifies critical files

### Non-Interactive Safety
- High-risk commands prompt for user approval in interactive mode
- Headless mode (non-TTY) automatically blocks dangerous commands

### Available Tools

| Tool | Purpose | Limits |
|------|---------|--------|
| `read_local_file` | Read workspace files | 1MB max, binary detection |
| `edit_local_file` | Create/overwrite files | HITL snapshots enabled |
| `run_shell_command` | Execute shell commands | 30s timeout, command checks |
| `list_directory` | List directory contents | Hides dotfiles by default |
| `web_search` | Search the web (DDGS) | 3 results, no API key |
| `fetch_webpage` | Extract text from URLs | 12KB truncation, async |
| `download_web_file` | Download files to workspace | Blocks executables (.sh, .exe, .bat) |
| `read_chat_history_from_sqlite` | Access past conversations | Configurable limit |
| `delete_chat_history_from_sqlite` | Manage conversation data | Per-session or full wipe |

## 💾 Memory System

### Episodic Memory (SQLite)

All conversations are persistently stored:

- **Session-based** — Each conversation is tagged with a unique session ID
- **Message history** — Stores role (human/ai), content, and timestamp
- **Episodic summaries** — At session end, the agent auto-generates a 2-sentence summary
- **Queryable** — Use `read_chat_history_from_sqlite` tool to retrieve past conversations

Example usage within a chat:
```
User: Read my last 50 messages
Agent: [Calls read_chat_history_from_sqlite with limit=50] → [Retrieves and displays history]
```

Or query directly:
```bash
sqlite3 ~/.local_ai_memory.db "SELECT * FROM chat_history ORDER BY id DESC LIMIT 20;"
```

### Knowledge Base (RAG)

Build a grounded knowledge base from local documents:

**Supported formats**: `.pdf`, `.txt`, `.md`, `.py`, `.json`, `.csv`, `.rs`, `.js`, `.ts`, `.html`

**Hybrid retrieval strategy**:
- **Chroma vector search** — Semantic similarity based on embeddings
- **BM25 keyword search** — Lexical keyword matching
- **Merged results** — Combined ranking to reduce redundancy

**Incremental indexing**:
- First run: Full vectorization of all files
- Subsequent runs: Only new/modified files are re-embedded
- State tracking: `.local_ai_chroma_db/.index_state.json` tracks file mtimes

Example flow:
```
User: How do I configure FastAPI with dependency injection?
Agent: 
  [Searches KB for "FastAPI dependency"] 
  → [Chroma returns 2 semantic results, BM25 returns 2 keyword results]
  → [Merges and deduplicates]
  → [Injects top results into LLM context]
  → [Grounds answer in retrieved docs]
```

## 🎯 Custom Skill Profiles

Specialize the agent for specific domains by loading custom skill profiles.

### Skill File Format

Create a markdown file in your `skills_folder` (e.g., `resources/skills/python_developer.md`):

```markdown
# Skill: Python Developer

- **Max Tool Iterations**: 20
- **System Prompt**:
  You are an expert Python developer with deep knowledge of async patterns, 
  type hints, and modern best practices. When the user asks about Python:
  - Provide async-first solutions when applicable
  - Use type hints in all examples
  - Reference PEP guidelines when relevant
  - Suggest testing strategies (pytest, fixtures)
```

### How It Works

1. Agent matches user query to the most relevant skill (semantic relevance matching)
2. Injects skill's custom system prompt into the LLM context
3. Respects skill's max tool iteration budget (default: 15)
4. Falls back to base agent mode if no skill is loaded

### Example Skills to Create

- **Web Researcher** (`web_researcher.md`) — Focus on web search, long tool budget
- **Code Debugger** (`code_debugger.md`) — Debugging focus, shell-aware, error analysis
- **Data Analyst** (`data_analyst.md`) — CSV/JSON focus, SQL-ready, statistical reasoning

## 🔌 Model Context Protocol (MCP) Integration

Extend the agent with pluggable tools and resources via the Model Context Protocol.

### Configuration

Edit or create `.private_agent.conf`:

```json
{
  "default_db_path": "~/.local_ai_memory.db",
  "workspace_root": ".",
  "skills_folder": "resources/skills",
  "rag_docs_path": "./docs",
  "default_model_temperature": 0.3,
  "embedding_model": "qwen2.5:7b",
  "thinking_toggle_default": false,
  "mcpServers": {
    "filesystem": {
      "command": "uv",
      "args": ["run", "mcp-server-filesystem"]
    }
  }
}
```

### How MCP Tools Are Loaded

1. On startup, agent connects to configured MCP servers via stdio
2. Retrieves available tools from each server
3. Automatically adds them to `AVAILABLE_TOOLS`
4. MCP tools are executed alongside native tools in the async loop

All MCP tools are gracefully integrated with full error handling and timeout protection.

## ⚙️ Configuration Guide

### .private_agent.conf

Create or edit `.private_agent.conf` in your project root:

```json
{
  "default_db_path": "~/.local_ai_memory.db",
  "workspace_root": ".",
  "skills_folder": "resources/skills",
  "rag_docs_path": null,
  "default_model_temperature": 0.3,
  "embedding_model": "qwen2.5:7b",
  "thinking_toggle_default": false,
  "mcpServers": {}
}
```

| Key | Default | Notes |
|-----|---------|-------|
| `default_db_path` | `~/.local_ai_memory.db` | SQLite storage location for chat history |
| `workspace_root` | `.` | Sandbox boundary for all file operations |
| `skills_folder` | `resources/skills` | Directory containing markdown skill profiles |
| `rag_docs_path` | `null` | Path to knowledge base files (e.g., `./docs/`, `./pdfs/`) |
| `default_model_temperature` | `0.3` | LLM creativity (0.0=deterministic, 1.0=creative) |
| `embedding_model` | `qwen2.5:7b` | Local embedding model via Ollama |
| `thinking_toggle_default` | `false` | Enable reasoning mode by default |
| `mcpServers` | `{}` | MCP server configurations |

### Environment Variables

The following are automatically set for offline operation:

```bash
HF_HUB_OFFLINE=1           # Force offline mode (no HuggingFace downloads)
TRANSFORMERS_OFFLINE=1     # No transformer library downloads
```

## 📊 Advanced Usage

### Model Selection & Tool Support

The agent intelligently detects tool support and falls back when needed:

| Model | Tool Support | Speed | Best For |
|-------|--------------|-------|----------|
| `qwen2.5` | ✅ Excellent | ⚡ Fast | General-purpose, default |
| `llama2` | ✅ Good | ⚡ Fast | Broad compatibility |
| `neural-chat` | ✅ Good | ⚡ Fast | Conversational |
| `deepseek-r1` | ✅ Excellent | 🟡 Slower | Reasoning, chain-of-thought |
| (Others) | ❓ Auto-fallback | Varies | May revert to qwen2.5 |

**Switch models mid-session**: Type `switch` at the prompt to return to model selection.

### Accessing Session History Directly

Query the SQLite database:

```bash
# All messages from last 30 days
sqlite3 ~/.local_ai_memory.db \
  "SELECT timestamp, role, content FROM chat_history 
   WHERE datetime(timestamp) > datetime('now', '-30 days') 
   ORDER BY id DESC;"

# All summaries
sqlite3 ~/.local_ai_memory.db \
  "SELECT session_id, content FROM chat_history WHERE role = 'summary';"
```

### Performance Tuning

| Scenario | Recommendation |
|----------|-----------------|
| Large RAG corpus (>1000 files) | Incremental indexing is automatic; only new docs are embedded |
| Slow on first RAG query | Embedding time depends on model size; `qwen2.5:7b` faster than larger variants |
| Context window fills quickly | Agent tracks usage and warns at >80%; increase temperature for exploration, not coverage |
| Need more tool iterations | Increase skill's `max_iterations` or `MAX_TOOL_ITERATIONS` in agent.py (line 213) |

### Debugging Tool Execution

To inspect tool outputs and execution:

1. **Check SQLite history**:
   ```bash
   sqlite3 ~/.local_ai_memory.db \
     "SELECT role, content FROM chat_history ORDER BY id DESC LIMIT 5;"
   ```

2. **Enable rich console output** (already enabled by default):
   - Tool calls are logged: `[Tool Execution] Calling 'name' with validated args: ...`
   - Errors are captured and fed back to the model for self-correction

3. **Security violations** raise `PermissionError`:
   - Check console output for `[Security Violation]` messages
   - Indicates an attempted path escape or sandbox breach

## 📁 Project Structure

```
Private_Agent/
├── main.py                      # Entry point
├── pyproject.toml               # UV build config, dependencies
├── requirements.txt             # pip-installable dependencies
├── README.md                    # This file
│
├── src/
│   ├── __init__.py
│   ├── agent.py                 # Async REPL, model loop, tool iteration
│   ├── config.py                # Configuration & defaults
│   ├── database.py              # SQLite persistent memory
│   ├── tools.py                 # Tool implementations & schemas
│   ├── sandbox.py               # Path validation & security
│   ├── rag.py                   # Hybrid RAG retriever (Chroma + BM25)
│   └── skills.py                # Skill profile loading & matching
│
├── tests/
│   └── test_private_agent.py    # pytest test suite
│
├── resources/
│   └── skills/                  # (Optional) Custom skill markdown files
│
├── .local_ai_chroma_db/         # (Auto-created) Vector store directory
│   └── .index_state.json        # Tracks file modification times
│
└── .agent_snapshots/            # (Auto-created) File edit backups (HITL)
```

## 🐛 Troubleshooting

### "Could not connect to Ollama"

**Cause**: Ollama server is not running or unreachable

**Solution**:
1. Ensure Ollama is running:
   ```bash
   ollama serve
   ```
2. Verify connectivity:
   ```bash
   curl http://localhost:11434/api/tags
   ```
3. Check the default URL in `src/agent.py` line 166 (should be `http://localhost:11434`)

### "No active chat models found via Ollama"

**Cause**: No models are installed locally

**Solution**:
```bash
# Pull a model
ollama pull llama2
ollama pull qwen2.5

# List installed models
ollama list
```

### "Model does not support tools"

**Cause**: Selected model doesn't implement tool calling

**Solution**:
- Agent auto-fallback: Switches to `qwen2.5` automatically
- Manual switch: Type `switch` at the prompt, select a tool-compatible model
- Tool-compatible models: `qwen2.5`, `llama2`, `neural-chat`, `deepseek-r1`

### "File not found" or "Security Violation"

**Cause**: Attempted to access file outside `workspace_root` or via symlink escape

**Solution**:
1. Verify file path is inside `workspace_root`:
   ```bash
   realpath /path/to/file
   # Should resolve within workspace_root
   ```
2. Check `.private_agent.conf` for `workspace_root` setting
3. Symlinks are followed; ensure targets are within workspace

### "High-risk command blocked"

**Cause**: Command contains dangerous patterns (e.g., `sudo`, `rm -rf`)

**Solution**:
- **Interactive mode**: Agent prompts for approval; type `y` to allow
- **Headless mode**: Commands are auto-blocked for safety
- **Disable check** (if you trust it): Edit `src/sandbox.py` line 44-52, but **not recommended**

### "SQLite database is locked"

**Cause**: Multiple agent instances accessing the same DB

**Solution**:
- Use separate `default_db_path` values in `.private_agent.conf` for different instances
- Or, ensure only one instance is running

### "Slow embedding on first RAG query"

**Cause**: First run downloads and initializes the embedding model

**Solution**:
- Subsequent queries are cached; only first run is slow
- Use a smaller embedding model:
  ```json
  {
    "embedding_model": "all-minilm:22m"
  }
  ```

### "Vector DB initialization error"

**Cause**: Issues with Chroma or embedding model

**Solution**:
1. Ensure embedding model is installed:
   ```bash
   ollama pull qwen2.5:7b
   ```
2. Check `.local_ai_chroma_db/` has write permissions
3. Delete and reinitialize:
   ```bash
   rm -rf .local_ai_chroma_db/
   python main.py  # Recreates DB
   ```

## 📝 Contributing

We welcome contributions! Areas of interest:

- **Tool expansion** — Add new sandboxed tools (e.g., database clients, image processing)
- **MCP integrations** — Test and optimize MCP server compatibility
- **Performance** — Optimize RAG indexing, context window usage
- **Security** — Report vulnerabilities responsibly via GitHub Security Advisory
- **Testing** — Expand test coverage, add integration tests

### Development Setup

```bash
pip install -r requirements.txt
pytest tests/  # Run test suite
```

## 📄 License

MIT License — See LICENSE file for details

## 🚀 What's Included

This project currently features:

- ✅ Async tool-calling agent with LangChain orchestration
- ✅ Hybrid RAG (Chroma vector + BM25 keyword search)
- ✅ SQLite persistent memory with episodic summaries
- ✅ Sandboxed file, shell, and web tools
- ✅ Skill profile system with semantic matching
- ✅ MCP integration framework
- ✅ GPU acceleration support (OpenVINO)
- ✅ Comprehensive pytest test suite

## 💬 Support

- **Issues & Bugs**: [GitHub Issues](https://github.com/Ziegan/Private_Agent/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Ziegan/Private_Agent/discussions)
- **Security**: Report via [GitHub Security Advisory](https://github.com/Ziegan/Private_Agent/security/advisories)

---

**Made with ❤️ by the Private Agent community**
