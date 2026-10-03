"""Argument schemas for built-in tool calls."""

from typing import Optional

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


class ListDirInput(BaseModel):
    dir_path: str = Field(default=".", description="Directory path to list files from.")


class FetchWebpageInput(BaseModel):
    url: str = Field(..., description="HTTP or HTTPS URL to fetch content from.")


class DownloadWebFileInput(BaseModel):
    url: str = Field(..., description="URL of file to download.")
    save_path: str = Field(..., description="Destination file path inside the workspace.")


class ReadSqliteHistoryInput(BaseModel):
    limit: int = Field(
        default=DEFAULT_HISTORY_READ_LIMIT,
        description="Number of recent chat history messages to read.",
    )


class DeleteSqliteHistoryInput(BaseModel):
    session_id: Optional[str] = Field(
        default=None,
        description="Specific session ID to delete, or leave empty/all to wipe.",
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
