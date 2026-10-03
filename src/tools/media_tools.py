"""Optional, session-scoped local camera and microphone tools."""

import io
import base64
import os
import threading
import uuid
import wave
from typing import Optional

from pydantic import BaseModel, Field
from langchain_core.tools import tool
from rich.console import Console

_captured_images: dict[str, bytes] = {}
_image_capture_count = 0
_image_lock = threading.Lock()
MAX_CAPTURED_IMAGE_BYTES = 2 * 1024 * 1024
MAX_IMAGES_PER_TURN = 3
MAX_MICROPHONE_SECONDS = 30
SAMPLE_RATE = 16_000
console = Console()


class WebcamCaptureInput(BaseModel):
    device_index: int = Field(
        default=0,
        ge=0,
        le=32,
        description="Camera device index; usually 0 for the built-in or first camera.",
    )


class MicrophoneInput(BaseModel):
    duration_seconds: float = Field(
        default=5.0,
        gt=0,
        le=MAX_MICROPHONE_SECONDS,
        description="Recording length in seconds (maximum 30 seconds).",
    )
    device_index: Optional[int] = Field(
        default=None,
        ge=0,
        le=128,
        description="Input-device index, or omit to use the system default microphone.",
    )


@tool
def list_microphone_devices() -> str:
    """List available local microphone input devices and their device indexes."""
    try:
        import sounddevice as sd
    except ImportError:
        return (
            "Microphone capture is optional. Install media support with "
            "`python -m pip install '.[media]'`."
        )
    try:
        devices = sd.query_devices()
        available = [
            f"{index}: {device['name']} (input channels: {device['max_input_channels']})"
            for index, device in enumerate(devices)
            if device["max_input_channels"] > 0
        ]
        return "\n".join(available) if available else "No microphone input devices were found."
    except Exception as exc:
        return f"Could not list microphone devices ({type(exc).__name__}): {exc}"


def captured_image_message(reference: str):
    """Build a transient multimodal message for an approved camera capture."""
    image = consume_captured_image(reference)
    if image is None:
        return None
    encoded = base64.b64encode(image).decode("ascii")
    from langchain_core.messages import HumanMessage

    return HumanMessage(
        content=[
            {"type": "text", "text": "Analyze this user-approved webcam frame."},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
            },
        ]
    )


def consume_captured_image(reference: str) -> Optional[bytes]:
    """Consume a single-use image reference returned by the camera tool."""
    prefix = "camera-image:"
    if not reference.startswith(prefix):
        return None
    return _captured_images.pop(reference[len(prefix):], None)


def clear_captured_images() -> None:
    global _image_capture_count
    _captured_images.clear()
    with _image_lock:
        _image_capture_count = 0


def approve_local_capture(tool_name: str, args: dict) -> bool:
    """Ask for explicit per-capture user consent; never approve headless capture."""
    if not __import__("sys").stdin.isatty():
        return False
    details = (
        f"for {float(args.get('duration_seconds', 5)):.1f} seconds"
        if tool_name == "record_microphone_audio"
        else f"from camera device {args.get('device_index', 0)}"
    )
    answer = console.input(
        f"[yellow]Allow {tool_name.replace('_', ' ')} {details}? "
        r"Capture/transcription stays on this machine. \[y/N]: [/yellow]"
    )
    return answer.strip().lower() == "y"


@tool(args_schema=WebcamCaptureInput)
def capture_webcam_image(device_index: int = 0) -> str:
    """Capture one still frame from a local webcam for the vision-capable local model."""
    capture = None
    try:
        import cv2
    except ImportError:
        return (
            "Camera capture is optional. Install media support with "
            "`python -m pip install '.[media]'`."
        )
    global _image_capture_count
    with _image_lock:
        if _image_capture_count >= MAX_IMAGES_PER_TURN:
            return (
                f"Camera capture limit reached ({MAX_IMAGES_PER_TURN} frames per "
                "user request); ask the user to start a new request."
            )
        _image_capture_count += 1
    try:
        capture = cv2.VideoCapture(device_index)
        if not capture.isOpened():
            return f"Could not open camera device {device_index}; check its index and permissions."
        ok, frame = capture.read()
        if not ok or frame is None:
            return f"Camera device {device_index} did not provide a frame."
        height, width = frame.shape[:2]
        scale = min(1.0, 1280 / width, 720 / height)
        if scale < 1.0:
            frame = cv2.resize(
                frame,
                (max(1, int(width * scale)), max(1, int(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        encoded, image = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        )
        if not encoded:
            return "Camera frame could not be encoded; no image was sent."
        image_bytes = image.tobytes()
        if len(image_bytes) > MAX_CAPTURED_IMAGE_BYTES:
            return "Captured image exceeded the 2 MiB safety limit."
        reference = uuid.uuid4().hex
        _captured_images[reference] = image_bytes
        return (
            f"Image captured in memory as camera-image:{reference}. "
            "Analyze the attached frame; it is not saved to disk."
        )
    except Exception as exc:
        return f"Camera capture failed ({type(exc).__name__}): {exc}"
    finally:
        if capture is not None:
            capture.release()


@tool(args_schema=MicrophoneInput)
def record_microphone_audio(
    duration_seconds: float = 5.0,
    device_index: Optional[int] = None,
) -> str:
    """Record a short local microphone clip and transcribe it with a configured offline Whisper model."""
    try:
        import sounddevice as sd
    except ImportError:
        return (
            "Microphone capture is optional. Install media support with "
            "`python -m pip install '.[media]'`."
        )

    model_path = os.environ.get("PRIVATE_AGENT_WHISPER_MODEL_PATH") or os.environ.get(
        "PRIVATE_AGENT_AUDIO_MODEL_PATH"
    )
    if not model_path:
        return (
            "Microphone transcription needs a local Whisper model directory. "
            "Set PRIVATE_AGENT_WHISPER_MODEL_PATH to an existing model directory; "
            "the tool will not download a model."
        )
    model_path = os.path.abspath(os.path.expanduser(model_path))
    if not os.path.isdir(model_path):
        return "Configured local Whisper model path is not an existing directory."

    try:
        import faster_whisper
    except ImportError:
        return (
            "Offline transcription support is unavailable. Reinstall optional "
            "media dependencies with `python -m pip install '.[media]'`."
        )

    audio = None
    try:
        audio = sd.rec(
            int(duration_seconds * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            device=device_index,
        )
        sd.wait()
    except Exception as exc:
        return f"Microphone recording failed ({type(exc).__name__}): {exc}"

    try:
        with io.BytesIO() as audio_file:
            with wave.open(audio_file, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(SAMPLE_RATE)
                wav_file.writeframes(audio.tobytes())
            audio_file.seek(0)
            model = faster_whisper.WhisperModel(
                model_path,
                device="cpu",
                compute_type="int8",
                local_files_only=True,
            )
            segments, _info = model.transcribe(audio_file)
            transcript = " ".join(
                segment.text.strip() for segment in segments if segment.text.strip()
            )
        if not transcript:
            return "No speech was recognized in the recording."
        return f"Offline microphone transcript ({duration_seconds:.1f}s): {transcript}"
    except Exception as exc:
        return f"Offline transcription failed ({type(exc).__name__}): {exc}"
