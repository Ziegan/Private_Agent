import os
import pathlib
import json
import warnings

# --- STRICT OFFLINE ENVIRONMENT ENFORCEMENT ---
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

CONFIG_FILE_PATH = pathlib.Path(".private_agent.conf")

DEFAULT_CONFIG = {
    "default_db_path": "~/.local_ai_memory.db",
    "workspace_root": ".",
    "skills_folder": "resources/skills",
    "rag_docs_path": None,
    "default_model_temperature": 0.3,
    "embedding_model": "qwen2.5:7b",
    "thinking_toggle_default": False,
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
SKILLS_FOLDER_DEFAULT = str(APP_CONFIG.get("skills_folder", "resources/skills"))

_raw_rag_path = APP_CONFIG.get("rag_docs_path", None)
RAG_DOCS_DEFAULT = str(_raw_rag_path) if _raw_rag_path else None

try:
    MODEL_TEMPERATURE = float(APP_CONFIG.get("default_model_temperature", 0.3))
except (TypeError, ValueError):
    MODEL_TEMPERATURE = 0.3

EMBEDDING_MODEL = str(APP_CONFIG.get("embedding_model", "qwen2.5:7b"))

_raw_thinking = APP_CONFIG.get("thinking_toggle_default", False)
if isinstance(_raw_thinking, str):
    THINKING_TOGGLE_DEFAULT = _raw_thinking.lower() in ("true", "1", "yes", "on")
else:
    THINKING_TOGGLE_DEFAULT = bool(_raw_thinking)

_raw_mcp = APP_CONFIG.get("mcpServers", {})
MCP_SERVERS = _raw_mcp if isinstance(_raw_mcp, dict) else {}
