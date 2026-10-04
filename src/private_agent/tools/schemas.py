"""Argument schemas for built-in tool calls."""

from typing import Literal, Optional

from pydantic import BaseModel, Field

from ..config import (
    DEFAULT_HISTORY_READ_LIMIT,
    MAX_SKILL_DESCRIPTION_CHARS,
    MAX_SKILL_INSTRUCTION_CHARS,
    MAX_SKILL_NAME_CHARS,
)


class ReadFileInput(BaseModel):
    file_path: str = Field(..., description="Path to the file to read relative to workspace.")


class EditFileInput(BaseModel):
    file_path: str = Field(..., description="Path to the file to create or overwrite.")
    content: str = Field(..., description="Full text content to write into the file.")


class RunShellInput(BaseModel):
    command: str = Field(..., description="Shell command string to execute.")


class WebSearchInput(BaseModel):
    query: str = Field(..., description="Search query string.")


class GetWeatherInput(BaseModel):
    location: Optional[str] = Field(
        default=None,
        max_length=200,
        description=(
            "Place name such as 'Paris' or 'Chennai, India'. Omit to use the "
            "current location (Local sessions only)."
        ),
    )
    date: Optional[str] = Field(
        default=None,
        description=(
            "First day as YYYY-MM-DD; defaults to today. Past dates return "
            "history; up to 16 days ahead returns a forecast."
        ),
    )
    time: Optional[str] = Field(
        default=None,
        description="Optional 24-hour HH:MM to also report that hour's conditions.",
    )
    days: int = Field(
        default=1,
        ge=1,
        le=16,
        description="Number of consecutive days from date (1-16); use 7 for a week.",
    )


class ListDirInput(BaseModel):
    dir_path: str = Field(default=".", description="Directory path to list files from.")


class SearchWorkspaceFilesInput(BaseModel):
    pattern: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Workspace-relative glob pattern such as '**/*.py' or '*.toml'.",
    )
    directory: str = Field(default=".", max_length=1024)
    limit: int = Field(default=100, ge=1, le=200)


class SearchWorkspaceTextInput(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    directory: str = Field(default=".", max_length=1024)
    pattern: str = Field(default="*", min_length=1, max_length=256)
    limit: int = Field(default=50, ge=1, le=100)


class SearchWorkspaceSymbolsInput(BaseModel):
    symbol: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Exact case-insensitive symbol name to locate in source declarations.",
    )
    directory: str = Field(default=".", max_length=1024)
    pattern: str = Field(
        default="*",
        min_length=1,
        max_length=256,
        description="Workspace-relative glob, for example '**/*.py'.",
    )
    limit: int = Field(default=50, ge=1, le=200)


class PreviewWorkspacePatchInput(BaseModel):
    file_path: str = Field(..., max_length=1024)
    old_text: str = Field(..., min_length=1, max_length=50_000)
    new_text: str = Field(..., max_length=50_000)


class ApplyWorkspacePatchInput(PreviewWorkspacePatchInput):
    pass


class RenameWorkspaceFileInput(BaseModel):
    source_path: str = Field(..., min_length=1, max_length=1024)
    destination_path: str = Field(..., min_length=1, max_length=1024)


class DeleteWorkspaceFileInput(BaseModel):
    file_path: str = Field(..., min_length=1, max_length=1024)


class InspectWorkspaceGitInput(BaseModel):
    action: Literal["status", "diff", "log", "branches"]
    directory: str = Field(default=".", max_length=1024)


class FetchWebpageInput(BaseModel):
    url: str = Field(..., description="HTTP or HTTPS URL to fetch content from.")


class DownloadWebFileInput(BaseModel):
    url: str = Field(..., description="URL of file to download.")
    save_path: str = Field(..., description="Destination file path inside the workspace.")


class ReadSqliteHistoryInput(BaseModel):
    limit: int = Field(
        default=DEFAULT_HISTORY_READ_LIMIT,
        ge=1,
        le=500,
        description="Number of recent chat history messages to read (1-500).",
    )


class DeleteSqliteHistoryInput(BaseModel):
    session_id: Optional[str] = Field(
        default=None,
        description="Specific session ID to delete, or leave empty/all to wipe.",
    )


class SearchSqliteHistoryInput(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Literal text to search in locally stored messages and summaries.",
    )
    session_id: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Optionally restrict results to one session.",
    )
    limit: int = Field(default=10, ge=1, le=50)


class DeleteSqliteHistoryEntryInput(BaseModel):
    entry_id: int = Field(..., ge=1, description="Entry ID returned by local history search.")
    session_id: str = Field(..., min_length=1, max_length=200)


class TaskPlanStepInput(BaseModel):
    step_id: str = Field(..., min_length=1, max_length=64)
    description: str = Field(..., min_length=1, max_length=1000)
    dependencies: list[str] = Field(default_factory=list, max_length=20)
    validation: list[str] = Field(default_factory=list, max_length=10)
    proof: list[str] = Field(default_factory=list, max_length=10)
    risks: list[str] = Field(default_factory=list, max_length=10)
    edge_cases: list[str] = Field(default_factory=list, max_length=10)


class CreateTaskPlanInput(BaseModel):
    goal: str = Field(..., min_length=1, max_length=2000)
    task_type: Literal["research", "coding", "other"]
    steps: list[TaskPlanStepInput] = Field(..., min_length=1, max_length=20)
    assumptions: list[str] = Field(default_factory=list, max_length=10)
    constraints: list[str] = Field(default_factory=list, max_length=10)
    research_questions: list[str] = Field(default_factory=list, max_length=10)
    research_references: list[str] = Field(
        default_factory=list,
        max_length=10,
        description="Runtime-injected citations from consented planning searches; do not invent.",
    )


class ReadTableSchemaInput(BaseModel):
    table_name: Literal[
        "agent_tasks",
        "agent_plan_revisions",
        "agent_todo_steps",
        "agent_task_events",
        "chat_history",
        "agent_learned_items",
        "agent_learning_settings",
    ]


class InspectTaskPlanInput(BaseModel):
    task_id: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=64,
        description="Omit to inspect the active task plan.",
    )


class UpdateTodoStepInput(BaseModel):
    step_id: str = Field(..., min_length=1, max_length=64)
    status: Literal["pending", "in_progress", "blocked", "reported_done"]
    evidence: str = Field(default="", max_length=2000)


class ReviseTaskPlanInput(BaseModel):
    goal: str = Field(..., min_length=1, max_length=2000)
    task_type: Literal["research", "coding", "other"]
    expected_revision: int = Field(..., ge=1)
    steps: list[TaskPlanStepInput] = Field(..., min_length=1, max_length=20)
    assumptions: list[str] = Field(default_factory=list, max_length=10)
    constraints: list[str] = Field(default_factory=list, max_length=10)
    research_questions: list[str] = Field(default_factory=list, max_length=10)
    research_references: list[str] = Field(
        default_factory=list,
        max_length=10,
        description="Runtime-injected citations from consented planning searches; do not invent.",
    )


class CreateSkillInput(BaseModel):
    name: str = Field(
        ...,
        min_length=1,
        max_length=MAX_SKILL_NAME_CHARS,
        description="Short skill title or slug, for example 'Python testing'.",
    )
    description: str = Field(
        ...,
        min_length=1,
        max_length=MAX_SKILL_DESCRIPTION_CHARS,
        description="When this skill should be used and what it specializes in.",
    )
    instructions: str = Field(
        ...,
        min_length=1,
        max_length=MAX_SKILL_INSTRUCTION_CHARS,
        description="Markdown instructions for the skill, based only on the user's request and relevant context.",
    )
