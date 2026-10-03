"""Compatibility facade for the agent runtime and its public helpers."""

from . import runtime as _runtime

__all__ = [
    name for name in vars(_runtime) if not name.startswith("__")
]


def __getattr__(name: str):
    try:
        return getattr(_runtime, name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
