import json
import pathlib
import stat
from unittest.mock import MagicMock
import pytest

from private_agent.config import load_or_create_config


def test_malformed_config_handling(tmp_path, monkeypatch):
    config_path = tmp_path / ".private_agent.conf"
    config_path.write_text("{ malformed_json: ", encoding="utf-8")
    monkeypatch.setattr("private_agent.config.CONFIG_FILE_PATH", config_path)

    config = load_or_create_config()
    assert isinstance(config, dict)
    assert "default_model_temperature" in config


def test_debug_log_is_created_with_owner_only_permissions(tmp_path, monkeypatch):
    import private_agent.run_logging as run_logging

    monkeypatch.setattr(run_logging, "DEBUG_LOG_ENABLED", True)
    monkeypatch.setattr(
        pathlib.Path,
        "home",
        classmethod(lambda cls: tmp_path),
    )
    try:
        log_path = run_logging.start_debug_logging()
        assert log_path is not None
        run_logging.RUN_LOGGER.info("trace marker")
        assert stat.S_IMODE(log_path.stat().st_mode) == 0o600
        assert stat.S_IMODE(log_path.parent.stat().st_mode) == 0o700
        run_logging.stop_debug_logging()
        assert "trace marker" in log_path.read_text(encoding="utf-8")
    finally:
        run_logging.stop_debug_logging()


def test_config_uses_per_user_home_path():
    import private_agent.config as config

    assert config.CONFIG_FILE_PATH == pathlib.Path.home() / ".private_agent.conf"


def test_config_creates_private_default_file(tmp_path, monkeypatch):
    import private_agent.config as config

    config_path = tmp_path / ".private_agent.conf"
    monkeypatch.setattr(config, "CONFIG_FILE_PATH", config_path)

    settings = load_or_create_config()

    assert settings["paths"] == config.DEFAULT_CONFIG["paths"]
    assert settings["paths"]["rag_documents"] == "~/.private_agent/rag/"
    assert settings["paths"]["skills"] == "~/.private_agent/resources/skills"
    assert settings["paths"]["rag_index"] == "~/.private_agent/rag_index"
    assert settings["agent"] == config.DEFAULT_CONFIG["agent"]
    assert settings["agent"]["streaming_output"] is True
    assert settings["agent"]["visible_reasoning"] is False
    assert settings["tools"]["override_tool_list"] == []
    assert settings["DEBUG_LOG_ENABLED"] == 0
    assert json.loads(config_path.read_text(encoding="utf-8")) == config.DEFAULT_CONFIG
    assert stat.S_IMODE(config_path.stat().st_mode) == 0o600


def test_default_system_prompt_is_loaded_from_packaged_resource_and_configured(
    tmp_path, monkeypatch
):
    import private_agent.config as config

    config_path = tmp_path / ".private_agent.conf"
    monkeypatch.setattr(config, "CONFIG_FILE_PATH", config_path)

    settings = load_or_create_config()

    assert settings["agent"]["system_prompt"] == config.DEFAULT_SYSTEM_PROMPT
    assert "privacy-first AI assistant" in settings["agent"]["system_prompt"]
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved["agent"]["system_prompt"] == config.DEFAULT_SYSTEM_PROMPT


def test_custom_system_prompt_is_preserved_in_user_config(tmp_path, monkeypatch):
    import private_agent.config as config

    config_path = tmp_path / ".private_agent.conf"
    config_path.write_text(
        json.dumps({"agent": {"system_prompt": "Follow our project conventions."}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "CONFIG_FILE_PATH", config_path)

    settings = load_or_create_config()

    assert settings["agent"]["system_prompt"] == "Follow our project conventions."
    assert json.loads(config_path.read_text(encoding="utf-8"))["agent"][
        "system_prompt"
    ] == "Follow our project conventions."


def test_empty_system_prompt_is_replaced_with_packaged_default():
    import private_agent.config as config

    settings = config._merge_config({"agent": {"system_prompt": "  "}})

    assert settings["agent"]["system_prompt"] == config.DEFAULT_SYSTEM_PROMPT


def test_categorized_config_preserves_legacy_keys_and_prefers_category_values():
    import private_agent.config as config

    merged = config._merge_config({
        "max_tool_output_chars": 4096,
        "agent": {"max_tool_output_chars": 2048},
        "max_read_file_bytes": 4096,
        "tools": {"max_read_file_bytes": 8192},
    })

    assert merged["agent"]["max_tool_output_chars"] == 2048
    assert merged["max_tool_output_chars"] == 2048
    assert merged["tools"]["max_read_file_bytes"] == 8192
    assert merged["max_read_file_bytes"] == 8192


def test_categorized_config_maps_legacy_flat_settings():
    import private_agent.config as config

    merged = config._merge_config({
        "max_tool_output_chars": 4096,
        "rag_max_file_bytes": 123456,
        "DEBUG_LOG_ENABLED": 1,
    })

    assert merged["agent"]["max_tool_output_chars"] == 4096
    assert merged["rag"]["max_file_bytes"] == 123456
    assert merged["logging"]["debug_enabled"] == 1


def test_merge_config_migrates_old_default_rag_and_skills_paths():
    import private_agent.config as config

    merged = config._merge_config({
        "paths": {
            "skills": "resources/skills",
            "rag_documents": None,
            "rag_index": "./.local_ai_chroma_db",
        }
    })

    assert merged["paths"] == {
        "skills": "~/.private_agent/resources/skills",
        "rag_documents": "~/.private_agent/rag/",
        "rag_index": "~/.private_agent/rag_index",
        "database": "~/.local_ai_memory.db",
        "workspace": ".",
        "code_output": "~/.private_agent/projects",
    }


def test_override_tool_list_is_loaded_as_tool_names():
    import private_agent.config as config

    assert config.OVERRIDE_TOOL_LIST == frozenset(
        config.APP_CONFIG["tools"]["override_tool_list"]
    )


def test_config_paths_are_home_anchored_and_tilde_expanded(tmp_path, monkeypatch):
    import private_agent.config as config

    monkeypatch.setenv("HOME", str(tmp_path))

    assert config._configured_path(
        "~/.private_agent/rag/",
        config.DEFAULT_CONFIG["paths"]["rag_documents"],
    ) == str(tmp_path / ".private_agent" / "rag")
    assert config._configured_path(
        "project/skills",
        config.DEFAULT_CONFIG["paths"]["skills"],
    ) == str(tmp_path / "project" / "skills")


def test_runtime_limits_read_categorized_config_and_validate_ranges(monkeypatch):
    import private_agent.config as config

    monkeypatch.setattr(
        config,
        "APP_CONFIG",
        {"tools": {"max_read_file_bytes": 4096}, "media": {"jpeg_quality": 150}},
    )

    assert config._configured_int("tools", "max_read_file_bytes", 1024) == 4096
    assert config._configured_int_range("media", "jpeg_quality", 85, 1, 100) == 85


@pytest.mark.parametrize(
    ("value", "expected"),
    [(True, True), (False, False), ("off", False), ("yes", True), ("unknown", True)],
)
def test_streaming_output_configuration_parses_boolean_values(
    monkeypatch, value, expected
):
    import private_agent.config as config

    monkeypatch.setattr(config, "APP_CONFIG", {"agent": {"streaming_output": value}})

    assert config._configured_bool("agent", "streaming_output", True) is expected


def test_visible_reasoning_configuration_defaults_off_and_accepts_legacy_name():
    import private_agent.config as config

    assert config.DEFAULT_CONFIG["agent"]["visible_reasoning"] is False
    assert config._merge_config({"VISIBILE_REASONING": True})[
        "agent"
    ]["visible_reasoning"] is True


def test_loading_legacy_config_migrates_it_to_categorized_sections(
    tmp_path, monkeypatch
):
    config_path = tmp_path / ".private_agent.conf"
    config_path.write_text(
        json.dumps({
            "max_tool_output_chars": 4096,
            "rag_max_file_bytes": 123456,
            "DEBUG_LOG_ENABLED": 1,
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr("private_agent.config.CONFIG_FILE_PATH", config_path)

    runtime_config = load_or_create_config()
    saved_config = json.loads(config_path.read_text(encoding="utf-8"))

    assert saved_config["agent"]["max_tool_output_chars"] == 4096
    assert saved_config["rag"]["max_file_bytes"] == 123456
    assert saved_config["logging"]["debug_enabled"] == 1
    assert "max_tool_output_chars" not in saved_config
    assert runtime_config["max_tool_output_chars"] == 4096
    assert stat.S_IMODE(config_path.stat().st_mode) == 0o600


def test_debug_log_config_defaults_off(tmp_path, monkeypatch):
    import private_agent.config as config

    monkeypatch.setattr(config, "CONFIG_FILE_PATH", tmp_path / ".private_agent.conf")
    assert load_or_create_config()["DEBUG_LOG_ENABLED"] == 0


def test_logging_console_adapter_installs_and_restores_module_consoles():
    import private_agent.code_tasks as code_tasks
    import private_agent.rag.indexing as rag_indexing
    import private_agent.rag.retrieval as rag_retrieval
    import private_agent.run_logging as run_logging
    import private_agent.skills.loader as skill_loader
    import private_agent.tools as tools
    import private_agent.tools.media as media
    import private_agent.tools.mcp as mcp
    import private_agent.tools.network as network
    import private_agent.tools.shell as shell
    from private_agent.agent import runtime

    modules = (
        runtime,
        code_tasks,
        rag_indexing,
        rag_retrieval,
        skill_loader,
        tools,
        media,
        mcp,
        network,
        shell,
    )
    original_consoles = {module: module.console for module in modules}
    adapter = run_logging.LoggingConsoleAdapter()

    saved_consoles = run_logging.install_logging_console_adapter(adapter)
    try:
        assert saved_consoles == list(original_consoles.items())
        assert all(module.console is adapter for module in modules)
    finally:
        run_logging.restore_logging_consoles(saved_consoles)

    assert all(
        module.console is original_consoles[module]
        for module in modules
    )


def test_debug_log_lines_have_timestamp_level_and_source_file(tmp_path, monkeypatch):
    import logging
    import re

    import private_agent.run_logging as run_logging

    monkeypatch.setattr(run_logging, "DEBUG_LOG_ENABLED", True)
    monkeypatch.setattr(run_logging, "DEBUG_LOG_LEVEL", "DEBUG")
    monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: tmp_path))
    try:
        log_path = run_logging.start_debug_logging()
        logging.getLogger("private_agent.database.memory").warning("module marker")
        run_logging.RUN_LOGGER.debug("debug marker")
        adapter = run_logging.LoggingConsoleAdapter(MagicMock())
        adapter.print("shown to user")
        run_logging.stop_debug_logging()
        text = log_path.read_text(encoding="utf-8")
    finally:
        run_logging.stop_debug_logging()
    assert re.search(
        r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d{3} \| WARNING  \| "
        r"private_agent\.database\.memory \| test_config\.py:\d+ \| .* \| module marker",
        text,
    )
    assert "DEBUG    | private_agent.run | test_config.py" in text
    assert "OUTPUT\nshown to user" in text
    adapter._base.print.assert_called_once_with("shown to user")


def test_debug_log_level_filters_and_terminal_stays_clean(tmp_path, monkeypatch, capsys):
    import logging

    import private_agent.run_logging as run_logging

    monkeypatch.setattr(run_logging, "DEBUG_LOG_ENABLED", True)
    monkeypatch.setattr(run_logging, "DEBUG_LOG_LEVEL", "WARNING")
    monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: tmp_path))
    try:
        log_path = run_logging.start_debug_logging()
        run_logging.RUN_LOGGER.info("hidden info")
        logging.getLogger("private_agent.x").error("kept error")
        run_logging.stop_debug_logging()
        text = log_path.read_text(encoding="utf-8")
    finally:
        run_logging.stop_debug_logging()
    assert "hidden info" not in text and "kept error" in text
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""
