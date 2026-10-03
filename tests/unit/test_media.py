import sys
import types
from unittest.mock import patch, MagicMock
from rich.console import Console

import pytest

try:
    from private_agent.agent import get_robust_chat_model
except ImportError:
    # Fallback definition if not explicitly exposed in private_agent.agent
    def get_robust_chat_model(primary_model_name, fallback_model_name, tools=None):
        from langchain_ollama import ChatOllama
        try:
            model = ChatOllama(model=primary_model_name)
            if tools:
                model = model.bind_tools(tools)
            return model
        except Exception:
            model = ChatOllama(model=fallback_model_name)
            if tools:
                model = model.bind_tools(tools)
            return model

from private_agent.tools.media import (
    capture_webcam_image,
    list_microphone_devices,
    record_microphone_audio,
    consume_captured_image,
    captured_image_message,
)

console = Console()

def test_webcam_capture_keeps_image_transient_and_releases_device(monkeypatch):
    class CapturedFrame:
        shape = (480, 640, 3)

    class EncodedImage:
        def tobytes(self):
            return b"fake-jpeg-frame"

    class FakeCapture:
        released = False

        def isOpened(self):
            return True

        def read(self):
            return True, CapturedFrame()

        def release(self):
            self.released = True

    capture = FakeCapture()
    cv2 = types.ModuleType("cv2")
    cv2.VideoCapture = lambda index: capture
    cv2.imencode = lambda *args: (True, EncodedImage())
    cv2.IMWRITE_JPEG_QUALITY = 1
    with patch.dict(sys.modules, {"cv2": cv2}):
        result = capture_webcam_image.invoke({"device_index": 0})
        match = __import__("re").search(r"camera-image:([0-9a-f]{32})", result)
        assert match
        message = captured_image_message(f"camera-image:{match.group(1)}")
    assert capture.released
    assert message.content[0]["type"] == "text"
    assert message.content[1]["type"] == "image_url"
    assert "ZmFrZS1qcGVnLWZyYW1l" in message.content[1]["image_url"]["url"]
    from langchain_ollama import ChatOllama
    converted = ChatOllama(model="test")._convert_messages_to_ollama_messages(
        [message]
    )
    assert converted[0]["role"] == "user"
    assert converted[0]["images"] == ["ZmFrZS1qcGVnLWZyYW1l"]
    assert consume_captured_image(f"camera-image:{match.group(1)}") is None

def test_media_capture_consent_prompt_shows_y_n_choices(monkeypatch):
    import private_agent.tools.media as media_tools
    from private_agent.run_logging import render_console_values

    prompt = MagicMock(return_value="")
    monkeypatch.setattr(media_tools, "console", MagicMock(input=prompt))
    monkeypatch.setattr(sys, "stdin", MagicMock(isatty=lambda: True))

    assert not media_tools.approve_local_capture("capture_webcam_image", {})
    rendered_prompt = render_console_values(prompt.call_args.args[0], end="")
    assert "[y/N]" in rendered_prompt

def test_microphone_transcription_uses_local_model_without_downloading(
    monkeypatch, tmp_path
):
    class FakeSoundDevice:
        def rec(self, frame_count, **kwargs):
            assert frame_count == 32000
            assert kwargs["device"] == 2
            return types.SimpleNamespace(tobytes=lambda: b"\0\0" * frame_count)

        def wait(self):
            return None

    class FakeWhisperModel:
        def __init__(self, model_path, **kwargs):
            assert model_path == str(tmp_path)
            assert kwargs["local_files_only"] is True
            assert kwargs["device"] == "cpu"

        def transcribe(self, audio_file):
            assert audio_file.read(4) == b"RIFF"
            return [types.SimpleNamespace(text=" local words ")], object()

    sounddevice = types.ModuleType("sounddevice")
    sounddevice.rec = FakeSoundDevice().rec
    sounddevice.wait = FakeSoundDevice().wait
    faster_whisper = types.ModuleType("faster_whisper")
    faster_whisper.WhisperModel = FakeWhisperModel
    monkeypatch.setenv("PRIVATE_AGENT_WHISPER_MODEL_PATH", str(tmp_path))
    with patch.dict(
        sys.modules,
        {"sounddevice": sounddevice, "faster_whisper": faster_whisper},
    ):
        result = record_microphone_audio.invoke({
            "duration_seconds": 2,
            "device_index": 2,
        })
    assert "local words" in result
    assert "Offline microphone transcript" in result

def test_microphone_device_listing_reports_available_input_indexes():
    sounddevice = types.ModuleType("sounddevice")
    sounddevice.query_devices = lambda: [
        {"name": "Output only", "max_input_channels": 0},
        {"name": "Internal microphone", "max_input_channels": 1},
    ]
    with patch.dict(sys.modules, {"sounddevice": sounddevice}):
        result = list_microphone_devices.invoke({})
    assert "1: Internal microphone (input channels: 1)" in result
    assert "Output only" not in result

@pytest.mark.asyncio
async def test_media_tools_require_local_runtime_authorization(monkeypatch):
    import private_agent.agent.runtime as agent

    camera = MagicMock()
    camera.invoke.return_value = "should not run"
    monkeypatch.setattr(agent, "AVAILABLE_TOOLS", {"capture_webcam_image": camera})
    result = await agent.execute_tool_call({
        "name": "capture_webcam_image",
        "args": {"device_index": 0},
        "id": "camera-no-consent",
    })
    assert "local Ollama session" in result.content
    camera.invoke.assert_not_called()

    result = await agent.execute_tool_call(
        {
            "name": "capture_webcam_image",
            "args": {"device_index": 0},
            "id": "camera-no-vision",
        },
        local_media_allowed=True,
        media_capture_authorized=True,
        vision_supported=False,
    )
    assert "does not declare vision support" in result.content
    camera.invoke.assert_not_called()

def test_webcam_tool_failure_is_reported_without_model_retry():
    import private_agent.agent.runtime as agent
    from langchain_core.messages import ToolMessage

    error = agent._webcam_capture_error(
        [{"name": "capture_webcam_image", "id": "camera-failed"}],
        [
            ToolMessage(
                content=(
                    "Camera capture is optional. Install media support with "
                    "`python -m pip install '.[media]'`."
                ),
                tool_call_id="camera-failed",
            )
        ],
    )

    assert error is not None
    assert "Camera capture is optional" in error
    assert (
        agent._webcam_capture_error(
            [{"name": "capture_webcam_image", "id": "camera-declined"}],
            [
                ToolMessage(
                    content="Error: Media capture was declined; the device was not accessed.",
                    tool_call_id="camera-declined",
                )
            ],
        )
        is None
    )
