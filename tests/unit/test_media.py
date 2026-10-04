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
    load_workspace_video,
    record_microphone_audio,
    transcribe_workspace_audio,
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


def test_workspace_video_sampling_is_bounded_and_transient(
    monkeypatch, temp_workspace
):
    import private_agent.tools.media as media

    video_path = temp_workspace / "clip.mp4"
    video_path.write_bytes(b"bounded mock video")
    frame = types.SimpleNamespace(shape=(360, 640, 3), ndim=3)

    class EncodedImage:
        def tobytes(self):
            return b"sampled-jpeg"

    class FakeCapture:
        released = False

        def __init__(self):
            self.read_count = 0
            self.positions = []

        def isOpened(self):
            return True

        def get(self, prop):
            return {1: 8, 2: 2, 4: 640, 5: 360}[prop]

        def set(self, prop, value):
            self.positions.append(value)

        def read(self):
            self.read_count += 1
            return True, frame

        def release(self):
            self.released = True

    capture = FakeCapture()
    cv2 = types.ModuleType("cv2")
    cv2.VideoCapture = lambda _path: capture
    cv2.CAP_PROP_FRAME_COUNT = 1
    cv2.CAP_PROP_FPS = 2
    cv2.CAP_PROP_POS_MSEC = 3
    cv2.CAP_PROP_FRAME_WIDTH = 4
    cv2.CAP_PROP_FRAME_HEIGHT = 5
    cv2.IMWRITE_JPEG_QUALITY = 4
    cv2.imencode = lambda *_args: (True, EncodedImage())
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    monkeypatch.setattr(media, "_image_capture_count", 0)
    media._captured_images.clear()

    result = load_workspace_video.invoke({"file_path": "clip.mp4"})
    references = __import__("re").findall(
        r"video-image:([0-9a-f]{32})", result
    )

    assert len(references) == 3
    assert capture.read_count == 3
    assert capture.positions == [0, 2000, 4000]
    assert capture.released
    assert "not saved" in result
    messages = [captured_image_message(f"video-image:{ref}") for ref in references]
    assert all(message.content[0]["text"].endswith("video frame.") for message in messages)
    assert all(consume_captured_image(f"video-image:{ref}") is None for ref in references)


def test_workspace_audio_file_is_bounded_and_transcribed_locally(
    monkeypatch, temp_workspace
):
    import io
    import wave

    audio_path = temp_workspace / "speech.wav"
    with audio_path.open("wb") as raw_audio:
        with wave.open(raw_audio, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(b"\0\0" * 16000)

    def transcribe(audio_file):
        assert isinstance(audio_file, io.BytesIO)
        assert audio_file.read(4) == b"RIFF"
        return "locally transcribed words"

    monkeypatch.setattr(
        "private_agent.tools.media._transcribe_local_audio", transcribe
    )
    result = transcribe_workspace_audio.invoke({"file_path": "speech.wav"})

    assert "1.0s" in result
    assert "locally transcribed words" in result


def test_workspace_audio_rejects_unsupported_and_oversized_inputs(
    monkeypatch, temp_workspace
):
    from private_agent.config import MAX_MEDIA_FILE_BYTES

    from private_agent.tools.media import transcribe_workspace_audio

    assert "PCM WAV files only" in transcribe_workspace_audio.invoke(
        {"file_path": "speech.mp3"}
    )
    oversized = temp_workspace / "oversized.wav"
    oversized.write_bytes(b"x" * (MAX_MEDIA_FILE_BYTES + 1))
    assert "safety limit" in transcribe_workspace_audio.invoke(
        {"file_path": "oversized.wav"}
    )


def test_workspace_video_rejects_overlong_media_and_releases_capture(
    monkeypatch, temp_workspace
):
    video_path = temp_workspace / "long.mp4"
    video_path.write_bytes(b"bounded mock video")

    class FakeCapture:
        released = False

        def isOpened(self):
            return True

        def get(self, prop):
            return {1: 10000, 2: 10, 4: 640, 5: 360}[prop]

        def release(self):
            self.released = True

    capture = FakeCapture()
    cv2 = types.ModuleType("cv2")
    cv2.VideoCapture = lambda _path: capture
    cv2.CAP_PROP_FRAME_COUNT = 1
    cv2.CAP_PROP_FPS = 2
    cv2.CAP_PROP_FRAME_WIDTH = 4
    cv2.CAP_PROP_FRAME_HEIGHT = 5
    monkeypatch.setitem(sys.modules, "cv2", cv2)

    result = load_workspace_video.invoke({"file_path": "long.mp4"})

    assert "at most" in result
    assert capture.released


def test_workspace_video_rejects_oversized_frame_before_decoding(
    monkeypatch, temp_workspace
):
    from private_agent.config import MAX_DECODED_IMAGE_PIXELS

    video_path = temp_workspace / "huge-frames.mp4"
    video_path.write_bytes(b"bounded mock video")

    class FakeCapture:
        read_called = False
        released = False

        def isOpened(self):
            return True

        def get(self, prop):
            return {
                1: 30,
                2: 30,
                4: MAX_DECODED_IMAGE_PIXELS + 1,
                5: 1,
            }[prop]

        def read(self):
            self.read_called = True
            raise AssertionError("Oversized video frame must not be decoded")

        def release(self):
            self.released = True

    capture = FakeCapture()
    cv2 = types.ModuleType("cv2")
    cv2.VideoCapture = lambda _path: capture
    cv2.CAP_PROP_FRAME_COUNT = 1
    cv2.CAP_PROP_FPS = 2
    cv2.CAP_PROP_FRAME_WIDTH = 4
    cv2.CAP_PROP_FRAME_HEIGHT = 5
    monkeypatch.setitem(sys.modules, "cv2", cv2)

    result = load_workspace_video.invoke({"file_path": "huge-frames.mp4"})

    assert "pixel safety limit" in result
    assert not capture.read_called
    assert capture.released


def test_runtime_attaches_every_transient_video_frame(monkeypatch):
    import private_agent.agent.runtime as agent
    from langchain_core.messages import HumanMessage, ToolMessage

    references = [
        f"video-image:{index:032x}"
        for index in range(3)
    ]
    attached = []

    def capture(reference):
        attached.append(reference)
        return HumanMessage(content=reference)

    monkeypatch.setattr(agent, "captured_image_message", capture)
    messages = []
    agent._append_captured_images_from_tool_results(
        messages,
        [
            ToolMessage(
                content="; ".join(references),
                tool_call_id="video-frames",
            )
        ],
    )

    assert attached == references
    assert [message.content for message in messages] == references

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


@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        ("load_workspace_video", {"file_path": "clip.mp4"}),
        ("transcribe_workspace_audio", {"file_path": "speech.wav"}),
    ],
)
@pytest.mark.asyncio
async def test_workspace_media_files_are_blocked_for_online_provider(
    monkeypatch, tool_name, arguments
):
    import private_agent.agent.runtime as agent

    media_tool = MagicMock()
    media_tool.ainvoke = MagicMock()
    monkeypatch.setattr(agent, "AVAILABLE_TOOLS", {tool_name: media_tool})
    monkeypatch.setattr(agent, "LOCAL_MEDIA_TOOL_NAMES", {tool_name})

    result = await agent.execute_tool_call(
        {
            "name": tool_name,
            "args": arguments,
            "id": "online-media-file",
        },
        local_media_allowed=False,
        media_capture_authorized=True,
        vision_supported=True,
    )

    assert "only in an explicitly selected local" in result.content
    media_tool.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_workspace_video_requires_vision_capability(monkeypatch):
    import private_agent.agent.runtime as agent

    video_tool = MagicMock()
    monkeypatch.setattr(agent, "AVAILABLE_TOOLS", {"load_workspace_video": video_tool})
    monkeypatch.setattr(agent, "LOCAL_MEDIA_TOOL_NAMES", {"load_workspace_video"})

    result = await agent.execute_tool_call(
        {
            "name": "load_workspace_video",
            "args": {"file_path": "clip.mp4"},
            "id": "video-no-vision",
        },
        local_media_allowed=True,
        media_capture_authorized=True,
        vision_supported=False,
    )

    assert "does not declare vision support" in result.content
    video_tool.ainvoke.assert_not_called()


def test_webcam_tool_failure_is_reported_without_model_retry():
    import private_agent.agent.runtime as agent
    from langchain_core.messages import ToolMessage

    error = agent._local_image_input_error(
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
        agent._local_image_input_error(
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
