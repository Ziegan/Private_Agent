from unittest.mock import MagicMock
from rich.console import Console

import pytest

from private_agent.tools import (
    create_skill,
    set_active_skill_runtime,
)
try:
    from private_agent.agent import get_robust_chat_model
except ImportError:
    # Fallback definition if not explicitly exposed in private_agent.agent
    def get_robust_chat_model(primary_model_name, fallback_model_name, tools=None):
        from langchain_ollama import ChatOllama
        try:
            model = ChatOllama(model=primary_model_name)
            if tools:
                model = model.bind_tools(tools)
            return model
        except Exception:
            model = ChatOllama(model=fallback_model_name)
            if tools:
                model = model.bind_tools(tools)
            return model

from private_agent.skills import load_skills_from_folder, match_skill_by_relevancy

console = Console()

def test_markdown_skill_loading(tmp_path):
    skill_dir = tmp_path / "skills"
    skill_dir.mkdir()
    
    skill_file = skill_dir / "python_expert.md"
    skill_file.write_text("# Python Expert\nWrite secure and robust python code.", encoding="utf-8")

    empty_file = skill_dir / "empty_skill.md"
    empty_file.write_text("", encoding="utf-8")

    skills = load_skills_from_folder(str(skill_dir))
    assert "python_expert" in skills
    assert "empty_skill" in skills

    matched = match_skill_by_relevancy("I need help with python coding", skills)
    assert matched is not None
    assert matched.name == "Python Expert"

def test_create_skill_tool_writes_markdown_and_updates_active_registry(tmp_path):
    registry = {}
    set_active_skill_runtime(str(tmp_path / "skills"), registry)
    result = create_skill.invoke({
        "name": "Python Testing",
        "description": "Help design focused Python tests.",
        "instructions": "Prefer pytest and cover important edge cases.",
    })
    skill_file = tmp_path / "skills" / "python_testing.md"
    assert "Created skill 'Python Testing'" in result
    assert skill_file.is_file()
    assert "# Python Testing" in skill_file.read_text(encoding="utf-8")
    assert "python_testing" in registry
    assert registry["python_testing"].name == "Python Testing"
    assert match_skill_by_relevancy(
        "I need help writing Python tests", registry
    ) is registry["python_testing"]

def test_create_skill_tool_rejects_path_traversal_and_existing_files(tmp_path):
    skills_dir = tmp_path / "skills"
    set_active_skill_runtime(str(skills_dir), {})
    args = {
        "name": "../outside",
        "description": "Unsafe path test.",
        "instructions": "Must not create outside the configured directory.",
    }
    assert "Error creating skill" in create_skill.invoke(args)
    assert not (tmp_path / "outside.md").exists()

    args["name"] = "safe_skill"
    assert "Created skill" in create_skill.invoke(args)
    original_content = (skills_dir / "safe_skill.md").read_text(encoding="utf-8")
    assert "already exists" in create_skill.invoke(args)
    assert (skills_dir / "safe_skill.md").read_text(encoding="utf-8") == original_content

@pytest.mark.asyncio
async def test_skill_creation_requires_interactive_approval(monkeypatch, tmp_path):
    import private_agent.agent.runtime as agent
    import private_agent.tools as tools

    registry = {}
    set_active_skill_runtime(str(tmp_path / "skills"), registry)
    monkeypatch.setattr(agent, "AVAILABLE_TOOLS", {"create_skill": create_skill})
    monkeypatch.setattr(agent.sys, "stdin", MagicMock(isatty=lambda: True))
    monkeypatch.setattr(agent.console, "input", lambda prompt: "n")
    args = {
        "name": "requested_skill",
        "description": "A user-requested skill.",
        "instructions": "Apply the requested reusable guidance.",
    }
    result = await agent.execute_tool_call({
        "name": "create_skill",
        "args": args,
        "id": "create-declined",
    })
    assert "declined" in result.content
    assert not (tmp_path / "skills" / "requested_skill.md").exists()

    monkeypatch.setattr(agent.console, "input", lambda prompt: "y")
    result = await agent.execute_tool_call({
        "name": "create_skill",
        "args": args,
        "id": "create-approved",
    })
    assert "Created skill 'requested_skill'" in result.content
    assert "requested_skill" in registry
    skill_catalog = next(
        entry for entry in tools.describe_tool_catalog()
        if entry["name"] == "create_skill"
    )
    assert "interactive approval" in skill_catalog["permission"]

@pytest.mark.asyncio
async def test_skill_creation_is_blocked_without_interactive_terminal(monkeypatch, tmp_path):
    import private_agent.agent.runtime as agent

    set_active_skill_runtime(str(tmp_path / "skills"), {})
    monkeypatch.setattr(agent, "AVAILABLE_TOOLS", {"create_skill": create_skill})
    monkeypatch.setattr(agent.sys, "stdin", MagicMock(isatty=lambda: False))
    result = await agent.execute_tool_call({
        "name": "create_skill",
        "args": {
            "name": "noninteractive",
            "description": "Not approved.",
            "instructions": "Must not be written.",
        },
        "id": "create-noninteractive",
    })
    assert "requires interactive approval" in result.content
    assert not (tmp_path / "skills" / "noninteractive.md").exists()
