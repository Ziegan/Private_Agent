"""Create validated skills and select relevant profiles."""

import pathlib
import re
from typing import Dict, Optional

from ..config import (
    MAX_SKILL_DESCRIPTION_CHARS,
    MAX_SKILL_INSTRUCTION_CHARS,
    MAX_SKILL_NAME_CHARS,
    SKILL_RELEVANCE_EXACT_MATCH_SCORE,
    SKILL_RELEVANCE_PREFIX_MATCH_SCORE,
    SKILL_RELEVANCE_THRESHOLD,
    SKILL_RELEVANCE_WORD_MATCH_SCORE,
)
from .loader import AgentSkill


def create_skill_file(
    folder_path: str,
    name: str,
    description: str,
    instructions: str,
) -> pathlib.Path:
    """Create one validated Markdown skill without replacing an existing file."""
    normalized_name = re.sub(r"[\s-]+", "_", name.strip().lower())
    if (
        not normalized_name
        or len(normalized_name) > MAX_SKILL_NAME_CHARS
        or not re.fullmatch(r"[a-z0-9_]+", normalized_name)
    ):
        raise ValueError(
            f"Skill name must be 1-{MAX_SKILL_NAME_CHARS} characters using letters, numbers, "
            "spaces, hyphens, or underscores."
        )
    description = description.strip()
    instructions = instructions.strip()
    if not description or len(description) > MAX_SKILL_DESCRIPTION_CHARS:
        raise ValueError(
            f"Skill description must contain 1-{MAX_SKILL_DESCRIPTION_CHARS} characters."
        )
    if not instructions or len(instructions) > MAX_SKILL_INSTRUCTION_CHARS:
        raise ValueError(
            f"Skill instructions must contain 1-{MAX_SKILL_INSTRUCTION_CHARS} characters."
        )

    directory = pathlib.Path(folder_path).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{normalized_name}.md"
    content = f"# {name.strip()}\n\n{description}\n\n{instructions}\n"
    try:
        with target.open("x", encoding="utf-8") as skill_file:
            skill_file.write(content)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Skill '{normalized_name}' already exists; no file was changed."
        ) from exc
    return target


def match_skill_by_relevancy(
    user_input: str,
    loaded_skills: Dict[str, AgentSkill],
) -> Optional[AgentSkill]:
    """Select a relevant skill using weighted keyword-token overlap."""
    if not loaded_skills:
        return None

    query_tokens = set(user_input.lower().split())
    if not query_tokens:
        return None

    best_skill = None
    highest_score = 0.0
    for key, skill in loaded_skills.items():
        key_tokens = set(key.lower().replace("_", " ").split())
        name_tokens = set(skill.name.lower().split())
        description_tokens = set(skill.description.lower().split())

        score = 0.0
        for token in query_tokens:
            if token in key_tokens:
                score += SKILL_RELEVANCE_EXACT_MATCH_SCORE
            if token in name_tokens:
                score += SKILL_RELEVANCE_PREFIX_MATCH_SCORE
            if token in description_tokens:
                score += SKILL_RELEVANCE_WORD_MATCH_SCORE

        if score > highest_score:
            highest_score = score
            best_skill = skill

    return best_skill if highest_score >= SKILL_RELEVANCE_THRESHOLD else None
