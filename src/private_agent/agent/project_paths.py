"""Discover project-local workspace and knowledge resource directories."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectResources:
    """Project folders discovered relative to the CLI invocation directory."""

    workspace: Path | None = None
    rag: Path | None = None
    skills: Path | None = None


def discover_project_resources(project_root: Path) -> ProjectResources:
    """Find conventional project folders in the supported locations.

    Search order for each folder is the project root, ``resources/``, then
    ``private_agent/resources/``. Symbolic links and paths resolving outside
    the project root are ignored so discovery cannot silently expand scope.
    """
    root = project_root.expanduser().resolve()
    resource_parents = (
        Path("."),
        Path("resources"),
        Path("private_agent") / "resources",
    )
    found: dict[str, Path] = {}

    for folder_name in ("workspace", "rag", "skills"):
        for parent_relative in resource_parents:
            parent = root / parent_relative
            candidate = parent / folder_name
            try:
                if any(
                    (root / Path(*parent_relative.parts[:index])).is_symlink()
                    for index in range(1, len(parent_relative.parts) + 1)
                ):
                    continue
                resolved_parent = parent.resolve(strict=True)
                resolved_candidate = candidate.resolve(strict=True)
            except (OSError, RuntimeError):
                continue
            if (
                not parent.is_dir()
                or not candidate.is_dir()
                or candidate.is_symlink()
                or not resolved_parent.is_relative_to(root)
                or not resolved_candidate.is_relative_to(root)
            ):
                continue
            found[folder_name] = resolved_candidate
            break

    return ProjectResources(
        workspace=found.get("workspace"),
        rag=found.get("rag"),
        skills=found.get("skills"),
    )
