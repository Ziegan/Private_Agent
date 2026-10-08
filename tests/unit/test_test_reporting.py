from types import SimpleNamespace

import conftest


def test_categories_are_derived_from_functional_test_modules():
    assert (
        conftest._test_category("tests/integration/test_agent_runtime.py::test_cli")
        == "Agent runtime"
    )
    assert conftest._test_category("tests/unit/test_database.py::test_read") == (
        "Database and memory"
    )


def test_terminal_summary_counts_results_and_lists_all_failed_tests(monkeypatch):
    reporter = SimpleNamespace(
        stats={
            "passed": [
                SimpleNamespace(
                    nodeid="tests/unit/test_config.py::test_passes",
                    wasxfail=None,
                )
            ],
            "failed": [
                SimpleNamespace(nodeid="tests/unit/test_config.py::test_fails")
            ],
            "skipped": [],
        },
        lines=[],
    )

    def write_sep(_separator, title):
        reporter.lines.append(title)

    def write_line(line):
        reporter.lines.append(line)

    reporter.write_sep = write_sep
    reporter.write_line = write_line
    conftest.pytest_terminal_summary(reporter)
    output = "\n".join(reporter.lines)

    assert "Total: 2 | Passed: 1 | Failed: 1" in output
    assert "Configuration: 1 passed, 1 failed" in output
    assert "tests/unit/test_config.py::test_fails" in output
    assert "tests/unit/test_config.py::test_passes" not in output
