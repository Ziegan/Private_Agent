import pathlib
import re
from typing import Optional, Dict
from rich.console import Console
from ..config import (
    MAX_SKILL_DESCRIPTION_CHARS,
    MAX_SKILL_INSTRUCTION_CHARS,
    MAX_SKILL_NAME_CHARS,
    SKILL_DESCRIPTION_PREVIEW_CHARS,
    SKILL_MAX_ITERATIONS,
    SKILL_RELEVANCE_EXACT_MATCH_SCORE,
    SKILL_RELEVANCE_PREFIX_MATCH_SCORE,
    SKILL_RELEVANCE_THRESHOLD,
    SKILL_RELEVANCE_WORD_MATCH_SCORE,
)

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
    """Loads markdown skill profiles from the designated skills directory."""
    skills = {}
    p = pathlib.Path(folder_path)
    if not p.exists() or not p.is_dir():
        return skills

    for md_file in p.glob("*.md"):
        try:
            content = md_file.read_text(encoding="utf-8")
            skill_name = md_file.stem.replace("_", " ").title()
            skills[md_file.stem] = AgentSkill(
                name=skill_name,
                description=content[:SKILL_DESCRIPTION_PREVIEW_CHARS] + "...",
                system_prompt=f"[Skill Markdown Profile: {skill_name}]\n{content}",
                max_iterations=SKILL_MAX_ITERATIONS,
            )
        except Exception as e:
            console.print(f"[red][Warning] Failed loading markdown skill file {md_file.name}: {e}[/red]")
    return skills


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


def match_skill_by_relevancy(user_input: str, loaded_skills: Dict[str, AgentSkill]) -> Optional[AgentSkill]:
    """Matches the most relevant loaded skill using weighted keyword-token overlap to prevent false-positive activations."""
    if not loaded_skills:
        return None
    
    query_tokens = set(user_input.lower().split())
    if not query_tokens:
        return None

    best_skill = None
    highest_score = 0.0

    for key, skill in loaded_skills.items():
        # Build token set from key, name, and description/system prompt
        key_tokens = set(key.lower().replace("_", " ").split())
        name_tokens = set(skill.name.lower().split())
        desc_tokens = set(skill.description.lower().split())
        
        # Assign weights
        score = 0.0
        for token in query_tokens:
            if token in key_tokens:
                score += SKILL_RELEVANCE_EXACT_MATCH_SCORE
            if token in name_tokens:
                score += SKILL_RELEVANCE_PREFIX_MATCH_SCORE
            if token in desc_tokens:
                score += SKILL_RELEVANCE_WORD_MATCH_SCORE

        # Normalize or check threshold to prevent false positives
        if score > highest_score:
            highest_score = score
            best_skill = skill

    # Require a minimum overlap score threshold to prevent weak/false-positive matches
    if highest_score >= SKILL_RELEVANCE_THRESHOLD:
        return best_skill
    
    return None
