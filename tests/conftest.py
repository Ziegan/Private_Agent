import pathlib
import tempfile
from collections import Counter

import pytest

from private_agent.sandbox import SandboxManager
from private_agent.tools import set_active_db_path

_TEST_CATEGORIES = {
    "test_agent_runtime": "Agent runtime",
    "test_benchmark_rag": "RAG benchmarks",
    "test_chunking": "RAG chunking",
    "test_cli": "CLI",
    "test_code_tasks": "Code tasks",
    "test_compaction": "Context compaction",
    "test_config": "Configuration",
    "test_contracts": "Contracts",
    "test_database": "Database and memory",
    "test_hardware": "Hardware",
    "test_interactive": "Interactive UX",
    "test_location": "Location and weather",
    "test_media": "Media",
    "test_network": "Network",
    "test_project_paths": "Project resources",
    "test_providers": "Model providers",
    "test_rag": "RAG",
    "test_session_stats": "Session statistics",
    "test_skills": "Skills",
    "test_tools": "Tools",
}
def _test_category(nodeid):
    module = nodeid.split("::", 1)[0]
    name = pathlib.Path(module).stem
    return _TEST_CATEGORIES.get(
        name,
        name.removeprefix("test_").replace("_", " ").title() or "Other",
    )


@pytest.hookimpl(trylast=True)
def pytest_terminal_summary(terminalreporter):
    categorized = {}
    completed = {}
    for outcome in ("passed", "skipped", "failed"):
        for report in terminalreporter.stats.get(outcome, []):
            result_outcome = (
                "xfailed"
                if outcome == "skipped" and hasattr(report, "wasxfail")
                else outcome
            )
            previous = completed.get(report.nodeid)
            if (
                previous is None
                or previous["outcome"] != "failed"
                or outcome == "failed"
            ):
                completed[report.nodeid] = {
                    "category": _test_category(report.nodeid),
                    "outcome": result_outcome,
                }
    for result in completed.values():
        counts = categorized.setdefault(result["category"], Counter())
        counts[result["outcome"]] += 1

    totals = Counter(result["outcome"] for result in completed.values())
    terminalreporter.write_sep("=", "Test results by functional category")
    terminalreporter.write_line(
        "Total: {total} | Passed: {passed} | Failed: {failed} | "
        "Skipped: {skipped} | XFailed: {xfailed}".format(
            total=sum(totals.values()),
            passed=totals["passed"],
            failed=totals["failed"],
            skipped=totals["skipped"],
            xfailed=totals["xfailed"],
        )
    )
    for category in sorted(categorized):
        counts = categorized[category]
        terminalreporter.write_line(
            f"{category}: {counts['passed']} passed, {counts['failed']} failed, "
            f"{counts['skipped']} skipped, {counts['xfailed']} xfailed"
        )

    failures = sorted(
        nodeid
        for nodeid, result in completed.items()
        if result["outcome"] == "failed"
    )
    if failures:
        terminalreporter.write_sep("!", "Failed tests")
        for nodeid in failures:
            terminalreporter.write_line(nodeid)


@pytest.fixture
def temp_db():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as temporary_file:
        db_path = temporary_file.name
    set_active_db_path(db_path)
    yield db_path
    path = pathlib.Path(db_path)
    if path.exists():
        path.unlink()


@pytest.fixture
def temp_workspace():
    with tempfile.TemporaryDirectory() as temporary_directory:
        original_root = SandboxManager.root_dir
        SandboxManager.set_root(temporary_directory)
        yield pathlib.Path(temporary_directory)
        SandboxManager.set_root(str(original_root))
