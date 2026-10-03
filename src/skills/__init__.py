import pathlib
import re
from typing import Optional, Dict
from rich.console import Console

console = Console()
MAX_SKILL_DESCRIPTION_CHARS = 500
MAX_SKILL_INSTRUCTION_CHARS = 20_000

class AgentSkill:
    def __init__(self, name: str, description: str, system_prompt: str, max_iterations: int = 15):
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
                description=content[:100] + "...",
                system_prompt=f"[Skill Markdown Profile: {skill_name}]\n{content}",
                max_iterations=15
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
        or len(normalized_name) > 64
        or not re.fullmatch(r"[a-z0-9_]+", normalized_name)
    ):
        raise ValueError(
            "Skill name must be 1-64 characters using letters, numbers, "
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
                score += 3.0
            if token in name_tokens:
                score += 2.0
            if token in desc_tokens:
                score += 1.0

        # Normalize or check threshold to prevent false positives
        if score > highest_score:
            highest_score = score
            best_skill = skill

    # Require a minimum overlap score threshold to prevent weak/false-positive matches
    threshold = 1.5
    if highest_score >= threshold:
        return best_skill
    
    return None
