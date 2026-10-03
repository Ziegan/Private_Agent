import os
import pathlib
import json
import tempfile
import warnings

from .hardware import normalize_acceleration_mode

# --- STRICT OFFLINE ENVIRONMENT ENFORCEMENT ---
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

CONFIG_FILE_PATH = pathlib.Path.home() / ".private_agent.conf"

DEFAULT_CONFIG = {
    "paths": {
        "database": "~/.local_ai_memory.db",
        "workspace": ".",
        "code_output": "~/.private_agent/projects",
        "skills": "resources/skills",
        "rag_documents": None,
        "rag_index": "./.local_ai_chroma_db",
    },
    "models": {
        "ollama_base_url": "http://localhost:11434",
        "hardware_acceleration": "auto",
        "preferred_model": None,
        "online_base_url": "https://api.openai.com/v1",
        "online_model": None,
        "temperature": 0.1,
        "embedding_model": "nomic-embed-text",
        "thinking_enabled_by_default": False,
        "thinking_effort": "medium",
        "online_request_timeout_seconds": 60,
        "online_model_list_timeout_seconds": 10,
        "online_max_retries": 1,
        "online_model_list_limit": 30,
    },
    "agent": {
        "permission_mode": "auto",
        "max_tool_iterations": 15,
        "max_tool_calls": 60,
        "max_task_seconds": 600,
        "max_tool_output_chars": 12000,
        "max_history_messages": 20,
        "max_context_tokens": 12000,
        "max_model_capability_cache_entries": 128,
        "summary_prompt_sentences": 2,
        "rag_context_results": 2,
    },
    "memory": {
        "conversation_retention_days": 0,
        "max_summary_chars": 2000,
        "default_history_messages": 20,
        "default_episodic_summaries": 5,
    },
    "rag": {
        "max_file_bytes": 5242880,
        "max_corpus_bytes": 26214400,
        "max_pdf_pages": 250,
        "max_documents": 20000,
        "chunk_size_chars": 1200,
        "chunk_overlap_chars": 200,
        "similarity_results": 4,
    },
    "network": {
        "web_research_enabled": True,
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
        "max_fetched_webpage_chars": 12000,
    },
    "tools": {
        "max_read_file_bytes": 1048576,
        "file_read_chunk_bytes": 65536,
        "binary_probe_bytes": 2048,
        "shell_command_timeout_seconds": 30,
        "default_history_read_limit": 20,
    },
    "media": {
        "max_captured_image_bytes": 2097152,
        "max_images_per_turn": 3,
        "max_microphone_seconds": 30,
        "default_microphone_seconds": 5.0,
        "microphone_sample_rate": 16000,
        "default_camera_device_index": 0,
        "max_camera_device_index": 32,
        "max_microphone_device_index": 128,
        "max_image_width": 1280,
        "max_image_height": 720,
        "jpeg_quality": 85,
    },
    "skills": {
        "max_name_chars": 64,
        "max_description_chars": 500,
        "max_instruction_chars": 20000,
        "max_iterations": 15,
        "description_preview_chars": 100,
        "relevance_threshold": 1.5,
        "relevance_exact_match_score": 3.0,
        "relevance_prefix_match_score": 2.0,
        "relevance_word_match_score": 1.0,
    },
    "code_tasks": {
        "git_command_timeout_seconds": 30,
        "test_command_timeout_seconds": 180,
        "successful_test_output_chars": 6000,
        "failed_test_output_chars": 3000,
    },
    "logging": {
        "debug_enabled": 0,
    },
    "mcp": {
        "auto_approve_tools": [],
        "servers": {},
    },
}

_LEGACY_CONFIG_KEYS = {
    "default_db_path": ("paths", "database"),
    "workspace_root": ("paths", "workspace"),
    "code_output_root": ("paths", "code_output"),
    "skills_folder": ("paths", "skills"),
    "rag_docs_path": ("paths", "rag_documents"),
    "rag_index_path": ("paths", "rag_index"),
    "ollama_base_url": ("models", "ollama_base_url"),
    "hardware_acceleration": ("models", "hardware_acceleration"),
    "preferred_model": ("models", "preferred_model"),
    "online_base_url": ("models", "online_base_url"),
    "online_model": ("models", "online_model"),
    "default_model_temperature": ("models", "temperature"),
    "embedding_model": ("models", "embedding_model"),
    "thinking_toggle_default": ("models", "thinking_enabled_by_default"),
    "thinking_effort_default": ("models", "thinking_effort"),
    "agent_permission_mode": ("agent", "permission_mode"),
    "max_tool_iterations": ("agent", "max_tool_iterations"),
    "max_tool_calls": ("agent", "max_tool_calls"),
    "max_task_seconds": ("agent", "max_task_seconds"),
    "max_tool_output_chars": ("agent", "max_tool_output_chars"),
    "max_history_messages": ("agent", "max_history_messages"),
    "max_context_tokens": ("agent", "max_context_tokens"),
    "max_model_capability_cache_entries": (
        "agent", "max_model_capability_cache_entries"
    ),
    "conversation_retention_days": ("memory", "conversation_retention_days"),
    "max_summary_chars": ("memory", "max_summary_chars"),
    "rag_max_file_bytes": ("rag", "max_file_bytes"),
    "rag_max_corpus_bytes": ("rag", "max_corpus_bytes"),
    "rag_max_pdf_pages": ("rag", "max_pdf_pages"),
    "rag_max_documents": ("rag", "max_documents"),
    "enable_web_research": ("network", "web_research_enabled"),
    "web_research_consent": ("network", "web_research_consent"),
    "internet_check_host": ("network", "internet_check_host"),
    "internet_check_port": ("network", "internet_check_port"),
    "internet_check_timeout": ("network", "internet_check_timeout_seconds"),
    "max_network_concurrency": ("network", "max_concurrent_requests"),
    "network_request_timeout": ("network", "request_timeout_seconds"),
    "max_webpage_bytes": ("network", "max_webpage_bytes"),
    "max_download_bytes": ("network", "max_download_bytes"),
    "max_http_redirects": ("network", "max_http_redirects"),
    "max_search_query_chars": ("network", "max_search_query_chars"),
    "max_search_results": ("network", "max_search_results"),
    "max_fetched_webpage_chars": ("network", "max_fetched_webpage_chars"),
    "max_read_file_bytes": ("tools", "max_read_file_bytes"),
    "file_read_chunk_bytes": ("tools", "file_read_chunk_bytes"),
    "binary_probe_bytes": ("tools", "binary_probe_bytes"),
    "shell_command_timeout_seconds": ("tools", "shell_command_timeout_seconds"),
    "default_history_read_limit": ("tools", "default_history_read_limit"),
    "rag_context_results": ("agent", "rag_context_results"),
    "summary_prompt_sentences": ("agent", "summary_prompt_sentences"),
    "DEBUG_LOG_ENABLED": ("logging", "debug_enabled"),
    "mcp_auto_approve_tools": ("mcp", "auto_approve_tools"),
    "mcpServers": ("mcp", "servers"),
}


def _merge_config(config: dict) -> dict:
    """Merge categorized settings and legacy flat settings into runtime aliases."""
    merged = {
        section: values.copy()
        for section, values in DEFAULT_CONFIG.items()
    }
    for section, values in config.items():
        if section in merged and isinstance(values, dict):
            merged[section].update(values)
    for old_key, (section, key) in _LEGACY_CONFIG_KEYS.items():
        if old_key in config and not (
            isinstance(config.get(section), dict) and key in config[section]
        ):
            merged[section][key] = config[old_key]
    for section, value in config.items():
        if section not in merged and section not in _LEGACY_CONFIG_KEYS:
            merged[section] = value
    for old_key, (section, key) in _LEGACY_CONFIG_KEYS.items():
        merged[old_key] = merged[section][key]
    return merged


def _categorized_config_for_file(config: dict, merged: dict) -> dict:
    """Return normalized sectioned settings while preserving unknown extensions."""
    categorized = {
        section: merged[section]
        for section in DEFAULT_CONFIG
    }
    categorized.update({
        key: value
        for key, value in config.items()
        if key not in DEFAULT_CONFIG and key not in _LEGACY_CONFIG_KEYS
    })
    return categorized


def _persist_categorized_config(config: dict) -> None:
    """Atomically migrate old flat settings and materialize new categorized defaults."""
    if CONFIG_FILE_PATH.is_symlink():
        warnings.warn(
            f"Not migrating configuration through symbolic link {CONFIG_FILE_PATH}.",
            RuntimeWarning,
        )
        return
    normalized = json.dumps(config, indent=4)
    try:
        if CONFIG_FILE_PATH.read_text(encoding="utf-8") == normalized:
            return
    except OSError:
        pass

    temporary_path = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{CONFIG_FILE_PATH.name}.",
            suffix=".tmp",
            dir=CONFIG_FILE_PATH.parent,
        )
        temporary_path = pathlib.Path(temporary_name)
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as config_file:
            config_file.write(normalized)
            config_file.flush()
            os.fsync(config_file.fileno())
        os.replace(temporary_path, CONFIG_FILE_PATH)
        os.chmod(CONFIG_FILE_PATH, 0o600)
    except OSError as exc:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        warnings.warn(
            f"Could not update categorized configuration at "
            f"{CONFIG_FILE_PATH}: {exc}",
            RuntimeWarning,
        )


def load_or_create_config() -> dict:
    """Load the per-user config, creating it with defaults when missing."""
    if not CONFIG_FILE_PATH.exists():
        try:
            descriptor = os.open(
                CONFIG_FILE_PATH,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as config_file:
                config_file.write(json.dumps(DEFAULT_CONFIG, indent=4))
        except FileExistsError:
            pass
        except Exception as e:
            warnings.warn(f"Failed to create default configuration file at {CONFIG_FILE_PATH}: {e}", RuntimeWarning)
            return _merge_config(DEFAULT_CONFIG)

    try:
        if CONFIG_FILE_PATH.is_dir():
            raise IsADirectoryError(f"Config path {CONFIG_FILE_PATH} is a directory, not a file.")

        content = CONFIG_FILE_PATH.read_text(encoding="utf-8").strip()
        if not content:
            return _merge_config(DEFAULT_CONFIG)

        config = json.loads(content)
        if not isinstance(config, dict):
            raise ValueError(f"Configuration root must be a JSON object (dict), got {type(config).__name__}")

        merged = _merge_config(config)
        _persist_categorized_config(_categorized_config_for_file(config, merged))
        return merged
    except (json.JSONDecodeError, ValueError, IsADirectoryError, PermissionError, OSError) as jde:
        # Suppress or issue warning cleanly for malformed/invalid config scenarios
        if not sys_modules_safe():
            warnings.warn(f"Malformed or invalid configuration encountered in {CONFIG_FILE_PATH}: {jde}. Falling back to default configuration.", RuntimeWarning)
        return _merge_config(DEFAULT_CONFIG)
    except Exception as e:
        warnings.warn(f"Unexpected error reading configuration file {CONFIG_FILE_PATH}: {e}. Falling back to default configuration.", RuntimeWarning)
        return _merge_config(DEFAULT_CONFIG)

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

def _positive_int(key: str, default: int) -> int:
    try:
        value = int(APP_CONFIG.get(key, default))
        return value if value > 0 else default
    except (TypeError, ValueError):
        return default


def _positive_float(key: str, default: float) -> float:
    try:
        value = float(APP_CONFIG.get(key, default))
        return value if value > 0 else default
    except (TypeError, ValueError):
        return default


def _configured_int(section: str, key: str, default: int) -> int:
    try:
        value = int(APP_CONFIG.get(section, {}).get(key, default))
        return value if value > 0 else default
    except (AttributeError, TypeError, ValueError):
        return default


def _configured_nonnegative_int(section: str, key: str, default: int) -> int:
    try:
        value = int(APP_CONFIG.get(section, {}).get(key, default))
        return value if value >= 0 else default
    except (AttributeError, TypeError, ValueError):
        return default


def _configured_int_range(
    section: str,
    key: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    value = _configured_int(section, key, default)
    return value if minimum <= value <= maximum else default


def _configured_float(section: str, key: str, default: float) -> float:
    try:
        value = float(APP_CONFIG.get(section, {}).get(key, default))
        return value if value > 0 else default
    except (AttributeError, TypeError, ValueError):
        return default


MODEL_TEMPERATURE = _configured_float("models", "temperature", 0.1)

EMBEDDING_MODEL = str(APP_CONFIG.get("embedding_model", "nomic-embed-text"))
RAG_INDEX_PATH = os.path.expanduser(
    str(APP_CONFIG.get("rag_index_path", "./.local_ai_chroma_db"))
)
OLLAMA_BASE_URL = str(APP_CONFIG.get("ollama_base_url", "http://localhost:11434"))
HARDWARE_ACCELERATION_MODE = normalize_acceleration_mode(
    APP_CONFIG.get("hardware_acceleration", "auto")
)
PREFERRED_MODEL = APP_CONFIG.get("preferred_model")

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
DEFAULT_HISTORY_MESSAGES = _configured_int(
    "memory", "default_history_messages", 20
)
DEFAULT_EPISODIC_SUMMARIES = _configured_int(
    "memory", "default_episodic_summaries", 5
)
MAX_NETWORK_CONCURRENCY = _positive_int("max_network_concurrency", 4)
NETWORK_REQUEST_TIMEOUT = _positive_int("network_request_timeout", 30)
RAG_MAX_FILE_BYTES = _positive_int("rag_max_file_bytes", 5 * 1024 * 1024)
RAG_MAX_CORPUS_BYTES = _positive_int("rag_max_corpus_bytes", 25 * 1024 * 1024)
RAG_MAX_PDF_PAGES = _positive_int("rag_max_pdf_pages", 250)
RAG_MAX_DOCUMENTS = _positive_int("rag_max_documents", 20000)
RAG_CHUNK_SIZE_CHARS = _configured_int("rag", "chunk_size_chars", 1200)
RAG_CHUNK_OVERLAP_CHARS = _configured_int("rag", "chunk_overlap_chars", 200)
RAG_SIMILARITY_RESULTS = _configured_int("rag", "similarity_results", 4)
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
INTERNET_CHECK_TIMEOUT = _positive_float("internet_check_timeout", 3.0)

_raw_thinking = APP_CONFIG.get("thinking_toggle_default", False)
if isinstance(_raw_thinking, str):
    THINKING_TOGGLE_DEFAULT = _raw_thinking.lower() in ("true", "1", "yes", "on")
else:
    THINKING_TOGGLE_DEFAULT = bool(_raw_thinking)

THINKING_EFFORT_DEFAULT = str(APP_CONFIG.get("thinking_effort_default", "medium")).lower()
if THINKING_EFFORT_DEFAULT not in {"low", "medium", "high"}:
    THINKING_EFFORT_DEFAULT = "medium"

AGENT_PERMISSION_MODE = str(APP_CONFIG.get("agent_permission_mode", "auto")).lower()
if AGENT_PERMISSION_MODE not in {"manual", "auto", "full"}:
    AGENT_PERMISSION_MODE = "auto"

_debug_log_enabled = APP_CONFIG.get("DEBUG_LOG_ENABLED", 0)
if isinstance(_debug_log_enabled, str):
    DEBUG_LOG_ENABLED = _debug_log_enabled.strip() == "1"
else:
    DEBUG_LOG_ENABLED = _debug_log_enabled == 1 or _debug_log_enabled is True

_raw_mcp = APP_CONFIG.get("mcpServers", {})
MCP_SERVERS = _raw_mcp if isinstance(_raw_mcp, dict) else {}
_raw_mcp_approvals = APP_CONFIG.get("mcp_auto_approve_tools", [])
MCP_AUTO_APPROVE_TOOLS = (
    {str(name) for name in _raw_mcp_approvals}
    if isinstance(_raw_mcp_approvals, list)
    else set()
)

ONLINE_REQUEST_TIMEOUT = _configured_int("models", "online_request_timeout_seconds", 60)
ONLINE_MODEL_LIST_TIMEOUT = _configured_int(
    "models", "online_model_list_timeout_seconds", 10
)
ONLINE_MAX_RETRIES = _configured_nonnegative_int("models", "online_max_retries", 1)
ONLINE_MODEL_LIST_LIMIT = _configured_int("models", "online_model_list_limit", 30)
MAX_MODEL_CAPABILITY_CACHE_ENTRIES = _configured_int(
    "agent", "max_model_capability_cache_entries", 128
)
MAX_WEBPAGE_BYTES = _configured_int("network", "max_webpage_bytes", 2 * 1024 * 1024)
MAX_DOWNLOAD_BYTES = _configured_int(
    "network", "max_download_bytes", 25 * 1024 * 1024
)
MAX_HTTP_REDIRECTS = _configured_nonnegative_int("network", "max_http_redirects", 5)
MAX_SEARCH_QUERY_CHARS = _configured_int("network", "max_search_query_chars", 2000)
MAX_SEARCH_RESULTS = _configured_int("network", "max_search_results", 3)
MAX_FETCHED_WEBPAGE_CHARS = _configured_int(
    "network", "max_fetched_webpage_chars", 12000
)
MAX_READ_FILE_BYTES = _configured_int("tools", "max_read_file_bytes", 1024 * 1024)
FILE_READ_CHUNK_BYTES = _configured_int("tools", "file_read_chunk_bytes", 65536)
BINARY_PROBE_BYTES = _configured_int("tools", "binary_probe_bytes", 2048)
SHELL_COMMAND_TIMEOUT = _configured_int("tools", "shell_command_timeout_seconds", 30)
DEFAULT_HISTORY_READ_LIMIT = _configured_int("tools", "default_history_read_limit", 20)
MAX_SKILL_NAME_CHARS = _configured_int("skills", "max_name_chars", 64)
MAX_STREAM_TIMEOUT = _configured_int(
    "network", "max_stream_timeout_seconds", 600
)
MAX_CAPTURED_IMAGE_BYTES = _configured_int(
    "media", "max_captured_image_bytes", 2 * 1024 * 1024
)
MAX_IMAGES_PER_TURN = _configured_int("media", "max_images_per_turn", 3)
MAX_MICROPHONE_SECONDS = _configured_int("media", "max_microphone_seconds", 30)
DEFAULT_MICROPHONE_SECONDS = _configured_float(
    "media", "default_microphone_seconds", 5.0
)
MICROPHONE_SAMPLE_RATE = _configured_int("media", "microphone_sample_rate", 16000)
DEFAULT_CAMERA_DEVICE_INDEX = _configured_nonnegative_int(
    "media", "default_camera_device_index", 0
)
MAX_CAMERA_DEVICE_INDEX = _configured_int("media", "max_camera_device_index", 32)
MAX_MICROPHONE_DEVICE_INDEX = _configured_int(
    "media", "max_microphone_device_index", 128
)
MAX_IMAGE_WIDTH = _configured_int("media", "max_image_width", 1280)
MAX_IMAGE_HEIGHT = _configured_int("media", "max_image_height", 720)
JPEG_QUALITY = _configured_int_range("media", "jpeg_quality", 85, 1, 100)
MAX_SKILL_DESCRIPTION_CHARS = _configured_int("skills", "max_description_chars", 500)
MAX_SKILL_INSTRUCTION_CHARS = _configured_int("skills", "max_instruction_chars", 20000)
SKILL_MAX_ITERATIONS = _configured_int("skills", "max_iterations", 15)
SKILL_DESCRIPTION_PREVIEW_CHARS = _configured_int(
    "skills", "description_preview_chars", 100
)
SKILL_RELEVANCE_THRESHOLD = _configured_float("skills", "relevance_threshold", 1.5)
SKILL_RELEVANCE_EXACT_MATCH_SCORE = _configured_float(
    "skills", "relevance_exact_match_score", 3.0
)
SKILL_RELEVANCE_PREFIX_MATCH_SCORE = _configured_float(
    "skills", "relevance_prefix_match_score", 2.0
)
SKILL_RELEVANCE_WORD_MATCH_SCORE = _configured_float(
    "skills", "relevance_word_match_score", 1.0
)
RAG_CONTEXT_RESULTS = _configured_int("agent", "rag_context_results", 2)
SUMMARY_PROMPT_SENTENCES = _configured_int("agent", "summary_prompt_sentences", 2)
CODE_TASK_GIT_TIMEOUT = _configured_int(
    "code_tasks", "git_command_timeout_seconds", 30
)
CODE_TASK_TEST_TIMEOUT = _configured_int(
    "code_tasks", "test_command_timeout_seconds", 180
)
CODE_TASK_SUCCESS_OUTPUT_CHARS = _configured_int(
    "code_tasks", "successful_test_output_chars", 6000
)
CODE_TASK_FAILURE_OUTPUT_CHARS = _configured_int(
    "code_tasks", "failed_test_output_chars", 3000
)
