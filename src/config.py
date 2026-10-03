import os
import pathlib
import json
import warnings

from .hardware import normalize_acceleration_mode

# --- STRICT OFFLINE ENVIRONMENT ENFORCEMENT ---
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

CONFIG_FILE_PATH = pathlib.Path(".private_agent.conf")

DEFAULT_CONFIG = {
    "default_db_path": "~/.local_ai_memory.db",
    "workspace_root": ".",
    "code_output_root": "~/.private_agent/projects",
    "skills_folder": "resources/skills",
    "rag_docs_path": None,
    "ollama_base_url": "http://localhost:11434",
    "hardware_acceleration": "auto",
    "preferred_model": None,
    "online_base_url": "https://api.openai.com/v1",
    "online_model": None,
    "default_model_temperature": 0.3,
    "embedding_model": "nomic-embed-text",
    "rag_index_path": "./.local_ai_chroma_db",
    "rag_max_file_bytes": 5242880,
    "rag_max_corpus_bytes": 26214400,
    "rag_max_pdf_pages": 250,
    "rag_max_documents": 20000,
    "thinking_toggle_default": False,
    "thinking_effort_default": "medium",
    "enable_web_research": True,
    "web_research_consent": "ask",
    "mcp_auto_approve_tools": [],
    "internet_check_host": "1.1.1.1",
    "internet_check_port": 443,
    "internet_check_timeout": 3.0,
    "max_tool_iterations": 15,
    "max_tool_calls": 60,
    "max_task_seconds": 600,
    "max_tool_output_chars": 12000,
    "max_history_messages": 20,
    "max_context_tokens": 12000,
    "conversation_retention_days": 0,
    "max_summary_chars": 2000,
    "max_network_concurrency": 4,
    "network_request_timeout": 30,
    "mcpServers": {}
}

def load_or_create_config() -> dict:
    """Loads settings from .private_agent.conf or creates it with defaults if missing, handling empty, non-dict, directory, or malformed files gracefully."""
    if not CONFIG_FILE_PATH.exists():
        try:
            CONFIG_FILE_PATH.write_text(json.dumps(DEFAULT_CONFIG, indent=4), encoding="utf-8")
        except Exception as e:
            warnings.warn(f"Failed to create default configuration file at {CONFIG_FILE_PATH}: {e}", RuntimeWarning)
        return DEFAULT_CONFIG.copy()
    
    try:
        if CONFIG_FILE_PATH.is_dir():
            raise IsADirectoryError(f"Config path {CONFIG_FILE_PATH} is a directory, not a file.")

        content = CONFIG_FILE_PATH.read_text(encoding="utf-8").strip()
        if not content:
            return DEFAULT_CONFIG.copy()
            
        config = json.loads(content)
        if not isinstance(config, dict):
            raise ValueError(f"Configuration root must be a JSON object (dict), got {type(config).__name__}")
            
        merged = DEFAULT_CONFIG.copy()
        merged.update(config)
        return merged
    except (json.JSONDecodeError, ValueError, IsADirectoryError, PermissionError, OSError) as jde:
        # Suppress or issue warning cleanly for malformed/invalid config scenarios
        if not sys_modules_safe():
            warnings.warn(f"Malformed or invalid configuration encountered in {CONFIG_FILE_PATH}: {jde}. Falling back to default configuration.", RuntimeWarning)
        return DEFAULT_CONFIG.copy()
    except Exception as e:
        warnings.warn(f"Unexpected error reading configuration file {CONFIG_FILE_PATH}: {e}. Falling back to default configuration.", RuntimeWarning)
        return DEFAULT_CONFIG.copy()

def sys_modules_safe() -> bool:
    try:
        import sys
        return "pytest" in sys.modules
    except Exception:
        return False

def load_config() -> dict:
    """Alias for load_or_create_config for compatibility with tool/agent modules."""
    return load_or_create_config()

# Load global configuration immediately with safe parsing/type conversions
APP_CONFIG = load_or_create_config()

DEFAULT_DB_PATH = os.path.expanduser(str(APP_CONFIG.get("default_db_path", "~/.local_ai_memory.db")))
WORKSPACE_ROOT_DEFAULT = str(APP_CONFIG.get("workspace_root", "."))
CODE_OUTPUT_ROOT = str(APP_CONFIG.get("code_output_root", "~/.private_agent/projects"))
SKILLS_FOLDER_DEFAULT = str(APP_CONFIG.get("skills_folder", "resources/skills"))

_raw_rag_path = APP_CONFIG.get("rag_docs_path", None)
RAG_DOCS_DEFAULT = str(_raw_rag_path) if _raw_rag_path else None

try:
    MODEL_TEMPERATURE = float(APP_CONFIG.get("default_model_temperature", 0.3))
except (TypeError, ValueError):
    MODEL_TEMPERATURE = 0.3

EMBEDDING_MODEL = str(APP_CONFIG.get("embedding_model", "nomic-embed-text"))
RAG_INDEX_PATH = os.path.expanduser(
    str(APP_CONFIG.get("rag_index_path", "./.local_ai_chroma_db"))
)
OLLAMA_BASE_URL = str(APP_CONFIG.get("ollama_base_url", "http://localhost:11434"))
HARDWARE_ACCELERATION_MODE = normalize_acceleration_mode(
    APP_CONFIG.get("hardware_acceleration", "auto")
)
PREFERRED_MODEL = APP_CONFIG.get("preferred_model")

def _positive_int(key: str, default: int) -> int:
    try:
        value = int(APP_CONFIG.get(key, default))
        return value if value > 0 else default
    except (TypeError, ValueError):
        return default

MAX_TOOL_ITERATIONS = _positive_int("max_tool_iterations", 15)
MAX_TOOL_CALLS = _positive_int("max_tool_calls", 60)
MAX_TASK_SECONDS = _positive_int("max_task_seconds", 600)
MAX_TOOL_OUTPUT_CHARS = _positive_int("max_tool_output_chars", 12000)
MAX_HISTORY_MESSAGES = _positive_int("max_history_messages", 20)
MAX_CONTEXT_TOKENS = _positive_int("max_context_tokens", 12000)
CONVERSATION_RETENTION_DAYS = max(
    0, _positive_int("conversation_retention_days", 0)
) if APP_CONFIG.get("conversation_retention_days", 0) else 0
MAX_SUMMARY_CHARS = _positive_int("max_summary_chars", 2000)
MAX_NETWORK_CONCURRENCY = _positive_int("max_network_concurrency", 4)
NETWORK_REQUEST_TIMEOUT = _positive_int("network_request_timeout", 30)
RAG_MAX_FILE_BYTES = _positive_int("rag_max_file_bytes", 5 * 1024 * 1024)
RAG_MAX_CORPUS_BYTES = _positive_int("rag_max_corpus_bytes", 25 * 1024 * 1024)
RAG_MAX_PDF_PAGES = _positive_int("rag_max_pdf_pages", 250)
RAG_MAX_DOCUMENTS = _positive_int("rag_max_documents", 20000)
_raw_web_research = APP_CONFIG.get("enable_web_research", True)
if isinstance(_raw_web_research, str):
    ENABLE_WEB_RESEARCH = _raw_web_research.lower() in ("true", "1", "yes", "on")
else:
    ENABLE_WEB_RESEARCH = bool(_raw_web_research)
WEB_RESEARCH_CONSENT = str(APP_CONFIG.get("web_research_consent", "ask")).lower()
if WEB_RESEARCH_CONSENT not in {"ask", "session", "never"}:
    WEB_RESEARCH_CONSENT = "ask"
INTERNET_CHECK_HOST = str(APP_CONFIG.get("internet_check_host", "1.1.1.1"))
INTERNET_CHECK_PORT = _positive_int("internet_check_port", 443)
try:
    INTERNET_CHECK_TIMEOUT = max(0.1, float(APP_CONFIG.get("internet_check_timeout", 3.0)))
except (TypeError, ValueError):
    INTERNET_CHECK_TIMEOUT = 3.0

_raw_thinking = APP_CONFIG.get("thinking_toggle_default", False)
if isinstance(_raw_thinking, str):
    THINKING_TOGGLE_DEFAULT = _raw_thinking.lower() in ("true", "1", "yes", "on")
else:
    THINKING_TOGGLE_DEFAULT = bool(_raw_thinking)

THINKING_EFFORT_DEFAULT = str(APP_CONFIG.get("thinking_effort_default", "medium")).lower()
if THINKING_EFFORT_DEFAULT not in {"low", "medium", "high"}:
    THINKING_EFFORT_DEFAULT = "medium"

_raw_mcp = APP_CONFIG.get("mcpServers", {})
MCP_SERVERS = _raw_mcp if isinstance(_raw_mcp, dict) else {}
_raw_mcp_approvals = APP_CONFIG.get("mcp_auto_approve_tools", [])
MCP_AUTO_APPROVE_TOOLS = (
    {str(name) for name in _raw_mcp_approvals}
    if isinstance(_raw_mcp_approvals, list)
    else set()
)
