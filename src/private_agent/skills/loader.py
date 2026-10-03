"""Load Markdown skill profiles into the runtime registry."""

import pathlib
from typing import Dict

from ..config import SKILL_DESCRIPTION_PREVIEW_CHARS, SKILL_MAX_ITERATIONS
from rich.console import Console

console = Console()


class AgentSkill:
    def __init__(
        self,
        name: str,
        description: str,
        system_prompt: str,
        max_iterations: int = SKILL_MAX_ITERATIONS,
    ):
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations


def load_skills_from_folder(folder_path: str) -> Dict[str, AgentSkill]:
    """Load Markdown skill profiles from the designated skills directory."""
    skills = {}
    path = pathlib.Path(folder_path)
    if not path.exists() or not path.is_dir():
        return skills

    for markdown_file in path.glob("*.md"):
        try:
            content = markdown_file.read_text(encoding="utf-8")
            skill_name = markdown_file.stem.replace("_", " ").title()
            skills[markdown_file.stem] = AgentSkill(
                name=skill_name,
                description=content[:SKILL_DESCRIPTION_PREVIEW_CHARS] + "...",
                system_prompt=f"[Skill Markdown Profile: {skill_name}]\n{content}",
                max_iterations=SKILL_MAX_ITERATIONS,
            )
        except Exception as exc:
            console.print(
                f"[red][Warning] Failed loading markdown skill file "
                f"{markdown_file.name}: {exc}[/red]"
            )
    return skills
