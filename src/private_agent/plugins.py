"""Explicit plugin contracts; this module does not discover or import plugins."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Protocol


_PLUGIN_NAME = re.compile(r"^[a-z][a-z0-9_-]{1,62}$")
_VERSION = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")


@dataclass(frozen=True)
class ResourceBudget:
    """Declared upper bounds a host can enforce around plugin operations."""

    timeout_seconds: float = 30.0
    max_calls: int = 100
    max_output_bytes: int = 1_048_576

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("Plugin timeout_seconds must be positive.")
        if self.max_calls < 1:
            raise ValueError("Plugin max_calls must be positive.")
        if self.max_output_bytes < 1:
            raise ValueError("Plugin max_output_bytes must be positive.")


@dataclass(frozen=True)
class PluginManifest:
    """Validated identity, capabilities, permissions, and resource limits.

    Hosts isolate initialization failures by disabling the failed plugin while
    keeping built-in capabilities and unrelated plugins available.
    """

    name: str
    version: str
    kind: Literal["provider", "tool", "memory", "retriever"]
    capabilities: tuple[str, ...] = ()
    permissions: tuple[str, ...] = ()
    resource_budget: ResourceBudget = field(default_factory=ResourceBudget)

    def __post_init__(self) -> None:
        if not _PLUGIN_NAME.fullmatch(self.name):
            raise ValueError(
                "Plugin name must be 2-63 lowercase letters, digits, '-' or '_', "
                "and start with a letter."
            )
        if not _VERSION.fullmatch(self.version):
            raise ValueError("Plugin version must use MAJOR.MINOR.PATCH format.")
        if self.kind not in {"provider", "tool", "memory", "retriever"}:
            raise ValueError(f"Unsupported plugin kind: {self.kind!r}.")
        if any(not item.strip() for item in (*self.capabilities, *self.permissions)):
            raise ValueError("Plugin capabilities and permissions must not be empty.")


@dataclass(frozen=True)
class PluginContext:
    """Host-supplied, explicitly scoped dependencies for plugin lifecycle."""

    config: Mapping[str, Any]
    services: Mapping[str, Any]
    logger: logging.Logger


class Plugin(Protocol):
    """Lifecycle contract for an explicitly selected and host-managed plugin.

    The contract does not authorize discovery or code loading. The host must
    enforce manifest permissions and budgets and isolate lifecycle failures.
    """

    manifest: PluginManifest

    def validate_config(self, raw_config: Mapping[str, Any]) -> Mapping[str, Any]:
        """Validate and normalize this plugin's configuration."""
        ...

    async def initialize(self, context: PluginContext) -> None:
        """Initialize using only host-provided configuration and services."""
        ...

    async def shutdown(self) -> None:
        """Release plugin-owned resources; safe to call after partial startup."""
        ...
