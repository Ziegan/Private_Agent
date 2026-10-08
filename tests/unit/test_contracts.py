import logging
from datetime import datetime, timezone

import pytest

from private_agent.contracts import (
    Evidence,
    ExecutionError,
    MemoryRecord,
    ModelCapabilities,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TokenUsage,
    ToolCall,
    ToolResult,
)
from private_agent.plugins import PluginContext, PluginManifest, ResourceBudget


def test_model_contracts_keep_unknown_capabilities_explicit():
    capabilities = ModelCapabilities(vision=True)
    request = ModelRequest(
        messages=(ModelMessage(role="user", content="question"),),
        max_output_tokens=128,
    )
    response = ModelResponse(
        content="answer",
        usage=TokenUsage(input_tokens=4, output_tokens=2),
        tool_calls=(ToolCall("call-1", "lookup", {"query": "value"}),),
    )
    tool_result = ToolResult("call-1", "result")

    assert capabilities.vision is True
    assert capabilities.tools is None
    assert request.messages[0].role == "user"
    assert response.usage is not None
    assert response.usage.input_tokens == 4
    assert response.tool_calls[0].arguments["query"] == "value"
    assert tool_result.is_error is False


def test_memory_and_evidence_contracts_preserve_provenance():
    timestamp = datetime.now(timezone.utc)
    evidence = Evidence(
        source="README.md",
        retrieved_at=timestamp,
        excerpt="Local-first operation",
        locator="line 10",
        confidence=0.9,
    )
    record = MemoryRecord(
        record_id="memory-1",
        kind="repository",
        content="The project is local-first.",
        created_at=timestamp,
        evidence=(evidence,),
    )

    assert record.evidence[0].source == "README.md"
    assert record.evidence[0].confidence == 0.9


def test_execution_error_exposes_stable_classification():
    error = ExecutionError(
        "provider.timeout",
        "The provider request timed out.",
        retryable=True,
        details={"provider": "local"},
    )

    assert str(error) == "The provider request timed out."
    assert error.code == "provider.timeout"
    assert error.retryable is True
    assert error.details == {"provider": "local"}
    with pytest.raises(ValueError, match="must not be empty"):
        ExecutionError("", "invalid")


def test_plugin_contract_metadata_is_validated_and_bounded():
    manifest = PluginManifest(
        name="local-provider",
        version="1.2.3",
        kind="provider",
        capabilities=("chat",),
        permissions=("network:loopback",),
        resource_budget=ResourceBudget(timeout_seconds=5, max_calls=4),
    )
    context = PluginContext(config={}, services={}, logger=logging.getLogger(__name__))

    assert manifest.resource_budget.max_calls == 4
    assert context.config == {}

    with pytest.raises(ValueError, match="Plugin name"):
        PluginManifest(name="../unsafe", version="1.0.0", kind="tool")
    with pytest.raises(ValueError, match="MAJOR.MINOR.PATCH"):
        PluginManifest(name="example-plugin", version="1", kind="tool")
    with pytest.raises(ValueError, match="timeout_seconds"):
        ResourceBudget(timeout_seconds=0)
