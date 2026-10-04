"""Optional, session-scoped local camera and microphone tools."""

import base64
import io
import math
import os
import threading
import uuid
import wave
from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from rich.console import Console
from rich.markup import escape

from ..config import (
    DEFAULT_CAMERA_DEVICE_INDEX,
    DEFAULT_MICROPHONE_SECONDS,
    JPEG_QUALITY,
    MAX_CAMERA_DEVICE_INDEX,
    MAX_CAPTURED_IMAGE_BYTES,
    MAX_DECODED_IMAGE_PIXELS,
    MAX_IMAGE_HEIGHT,
    MAX_IMAGE_WIDTH,
    MAX_IMAGES_PER_TURN,
    MAX_MEDIA_FILE_BYTES,
    MAX_MICROPHONE_DEVICE_INDEX,
    MAX_MICROPHONE_SECONDS,
    MAX_VIDEO_SECONDS,
    MICROPHONE_SAMPLE_RATE,
)
from ..sandbox import SandboxManager

_captured_images: dict[str, bytes] = {}
_image_capture_count = 0
_image_lock = threading.Lock()
console = Console()


class WebcamCaptureInput(BaseModel):
    device_index: int = Field(
        default=DEFAULT_CAMERA_DEVICE_INDEX,
        ge=0,
        le=MAX_CAMERA_DEVICE_INDEX,
        description="Camera device index; usually 0 for the built-in or first camera.",
    )


class WorkspaceImageInput(BaseModel):
    file_path: str = Field(
        ...,
        min_length=1,
        max_length=1024,
        description="Workspace-relative image file to attach to the local vision model.",
    )


class WorkspaceVideoInput(BaseModel):
    file_path: str = Field(
        ...,
        min_length=1,
        max_length=1024,
        description="Workspace-relative video file to sample for the local vision model.",
    )


class WorkspaceAudioInput(BaseModel):
    file_path: str = Field(
        ...,
        min_length=1,
        max_length=1024,
        description="Workspace-relative PCM WAV file to transcribe with local Whisper.",
    )


class MicrophoneInput(BaseModel):
    duration_seconds: float = Field(
        default=DEFAULT_MICROPHONE_SECONDS,
        gt=0,
        le=MAX_MICROPHONE_SECONDS,
        description="Recording length in seconds (maximum 30 seconds).",
    )
    device_index: Optional[int] = Field(
        default=None,
        ge=0,
        le=MAX_MICROPHONE_DEVICE_INDEX,
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
    """Build a transient multimodal message for a user-approved local image."""
    image = consume_captured_image(reference)
    if image is None:
        return None
    encoded = base64.b64encode(image).decode("ascii")
    from langchain_core.messages import HumanMessage

    if reference.startswith("camera-image:"):
        image_kind = "webcam frame"
    elif reference.startswith("video-image:"):
        image_kind = "video frame"
    else:
        image_kind = "workspace image"
    return HumanMessage(
        content=[
            {"type": "text", "text": f"Analyze this user-approved local {image_kind}."},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
            },
        ]
    )


def consume_captured_image(reference: str) -> Optional[bytes]:
    """Consume a single-use image reference returned by a local media tool."""
    prefix = next(
        (
            prefix
            for prefix in ("camera-image:", "workspace-image:", "video-image:")
            if reference.startswith(prefix)
        ),
        None,
    )
    if prefix is None:
        return None
    return _captured_images.pop(reference[len(prefix):], None)


def clear_captured_images() -> None:
    global _image_capture_count
    _captured_images.clear()
    with _image_lock:
        _image_capture_count = 0


def approve_local_capture(tool_name: str, args: dict) -> bool:
    """Ask explicit per-use consent for local media access; never approve headless."""
    if not __import__("sys").stdin.isatty():
        return False
    if tool_name == "record_microphone_audio":
        details = (
            f"for {float(args.get('duration_seconds', DEFAULT_MICROPHONE_SECONDS)):.1f} seconds"
        )
    elif tool_name == "capture_webcam_image":
        details = f"from camera device {args.get('device_index', 0)}"
    elif tool_name == "load_workspace_video":
        details = f"from workspace video '{escape(str(args.get('file_path', '')))}'"
    elif tool_name == "transcribe_workspace_audio":
        details = (
            f"from workspace audio file "
            f"'{escape(str(args.get('file_path', '')))}'"
        )
    else:
        details = f"from workspace file '{escape(str(args.get('file_path', '')))}'"
    if tool_name in {"record_microphone_audio", "transcribe_workspace_audio"}:
        privacy_action = "Audio transcription stays on this machine."
    elif tool_name == "load_workspace_video":
        privacy_action = "Video frames are processed on this machine."
    else:
        privacy_action = "Image processing stays on this machine."
    answer = console.input(
        f"[yellow]Allow {tool_name.replace('_', ' ')} {details}? "
        f"{privacy_action} \\[y/N]: [/yellow]"
    )
    return answer.strip().lower() == "y"


def _normalize_video_frame(cv2, frame) -> Optional[bytes]:
    if frame is None or getattr(frame, "ndim", 0) < 2:
        return None
    height, width = frame.shape[:2]
    if (
        height < 1
        or width < 1
        or height * width > MAX_DECODED_IMAGE_PIXELS
    ):
        return None
    scale = min(1.0, MAX_IMAGE_WIDTH / width, MAX_IMAGE_HEIGHT / height)
    if scale < 1.0:
        frame = cv2.resize(
            frame,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    encoded, image = cv2.imencode(
        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    )
    if not encoded:
        return None
    image_bytes = image.tobytes()
    if len(image_bytes) > MAX_CAPTURED_IMAGE_BYTES:
        return None
    return image_bytes


@tool(args_schema=WorkspaceVideoInput)
def load_workspace_video(file_path: str) -> str:
    """Sample a bounded number of frames from an approved local workspace video."""
    try:
        import cv2
    except ImportError:
        return (
            "Video sampling is optional. Install media support with "
            "`python -m pip install '.[media]'`."
        )
    if os.path.splitext(file_path)[1].lower() not in {
        ".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"
    }:
        return "Error: Workspace video must use a supported video file extension."
    capture = None
    try:
        target = SandboxManager.validate_path(file_path)
        if not target.is_file():
            return "Error: Workspace video path is not a regular file."
        if target.stat().st_size > MAX_MEDIA_FILE_BYTES:
            return (
                f"Error: Video exceeds the {MAX_MEDIA_FILE_BYTES}-byte safety limit."
            )
        capture = cv2.VideoCapture(str(target))
        if not capture.isOpened():
            return "Error: Workspace video could not be opened."
        frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if (
            not math.isfinite(frame_width)
            or not math.isfinite(frame_height)
            or frame_width <= 0
            or frame_height <= 0
        ):
            return "Error: Workspace video has invalid frame dimensions."
        if frame_width * frame_height > MAX_DECODED_IMAGE_PIXELS:
            return (
                f"Error: Video frame dimensions exceed the "
                f"{MAX_DECODED_IMAGE_PIXELS}-pixel safety limit."
            )
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        frames_per_second = capture.get(cv2.CAP_PROP_FPS)
        if (
            not math.isfinite(frame_count)
            or not math.isfinite(frames_per_second)
            or frame_count <= 0
            or frames_per_second <= 0
        ):
            return "Error: Workspace video has invalid duration metadata."
        duration = frame_count / frames_per_second
        if duration <= 0 or duration > MAX_VIDEO_SECONDS:
            return (
                f"Error: Video duration must be greater than zero and at most "
                f"{MAX_VIDEO_SECONDS} seconds."
            )
        global _image_capture_count
        with _image_lock:
            available = MAX_IMAGES_PER_TURN - _image_capture_count
            if available <= 0:
                return (
                    f"Error: Local image limit reached ({MAX_IMAGES_PER_TURN} "
                    "images per user request)."
                )
            sample_count = min(available, 3)
            references = []
            for index in range(sample_count):
                timestamp = (
                    duration / 2
                    if sample_count == 1
                    else duration * index / (sample_count - 1)
                )
                # Seeking exactly to the end yields no frame; stop one frame early.
                timestamp = max(0.0, min(timestamp, duration - 1 / frames_per_second))
                capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
                ok, frame = capture.read()
                if not ok:
                    continue
                image_bytes = _normalize_video_frame(cv2, frame)
                if image_bytes is None:
                    continue
                reference = uuid.uuid4().hex
                _captured_images[reference] = image_bytes
                _image_capture_count += 1
                references.append((timestamp, reference))
        if not references:
            return "Error: No bounded video frames could be decoded and normalized."
        return (
            "Sampled local video frames in memory (not saved): "
            + "; ".join(
                f"{timestamp:.1f}s video-image:{reference}"
                for timestamp, reference in references
            )
        )
    except (OSError, ValueError) as exc:
        return f"Error loading workspace video: {exc}"
    except Exception as exc:
        return f"Video processing failed ({type(exc).__name__}): {exc}"
    finally:
        if capture is not None:
            capture.release()


def _load_local_whisper_model():
    model_path = os.environ.get("PRIVATE_AGENT_WHISPER_MODEL_PATH") or os.environ.get(
        "PRIVATE_AGENT_AUDIO_MODEL_PATH"
    )
    if not model_path:
        raise ValueError(
            "Set PRIVATE_AGENT_WHISPER_MODEL_PATH to an existing local Whisper model; "
            "the tool will not download one."
        )
    model_path = os.path.abspath(os.path.expanduser(model_path))
    if not os.path.isdir(model_path):
        raise ValueError("Configured local Whisper model path is not an existing directory.")
    try:
        import faster_whisper
    except ImportError as exc:
        raise RuntimeError(
            "Offline transcription support is unavailable. Install media dependencies "
            "with `python -m pip install '.[media]'`."
        ) from exc
    return faster_whisper.WhisperModel(
        model_path,
        device="cpu",
        compute_type="int8",
        local_files_only=True,
    )


def _transcribe_local_audio(audio_file) -> str:
    model = _load_local_whisper_model()
    segments, _info = model.transcribe(audio_file)
    return " ".join(
        segment.text.strip() for segment in segments if segment.text.strip()
    )


@tool(args_schema=WorkspaceAudioInput)
def transcribe_workspace_audio(file_path: str) -> str:
    """Transcribe a bounded PCM WAV file from the workspace using local Whisper."""
    if os.path.splitext(file_path)[1].lower() != ".wav":
        return "Error: Audio-file transcription currently accepts PCM WAV files only."
    try:
        target = SandboxManager.validate_path(file_path)
        if not target.is_file():
            return "Error: Workspace audio path is not a regular file."
        if target.stat().st_size > MAX_MEDIA_FILE_BYTES:
            return (
                f"Error: Audio file exceeds the {MAX_MEDIA_FILE_BYTES}-byte safety limit."
            )
        with target.open("rb") as source:
            audio_bytes = source.read(MAX_MEDIA_FILE_BYTES + 1)
        if len(audio_bytes) > MAX_MEDIA_FILE_BYTES:
            return (
                f"Error: Audio file exceeds the {MAX_MEDIA_FILE_BYTES}-byte safety limit."
            )
        with io.BytesIO(audio_bytes) as audio_file:
            try:
                with wave.open(audio_file, "rb") as wav_file:
                    duration = wav_file.getnframes() / wav_file.getframerate()
                    if not 0 < duration <= MAX_MICROPHONE_SECONDS:
                        return (
                            f"Error: WAV duration must be greater than zero and at "
                            f"most {MAX_MICROPHONE_SECONDS} seconds."
                        )
                    if (
                        wav_file.getcomptype() != "NONE"
                        or wav_file.getnchannels() not in (1, 2)
                        or wav_file.getsampwidth() not in (1, 2, 3, 4)
                    ):
                        return (
                            "Error: Audio must be uncompressed PCM WAV with one or "
                            "two channels."
                        )
            except (wave.Error, EOFError, ZeroDivisionError) as exc:
                return f"Error: Invalid PCM WAV file ({exc})."
            audio_file.seek(0)
            transcript = _transcribe_local_audio(audio_file)
        if not transcript:
            return "No speech was recognized in the audio file."
        return f"Offline workspace audio transcript ({duration:.1f}s): {transcript}"
    except (OSError, ValueError, RuntimeError) as exc:
        return f"Offline audio-file transcription unavailable: {exc}"
    except Exception as exc:
        return f"Offline audio-file transcription failed ({type(exc).__name__}): {exc}"


@tool(args_schema=WebcamCaptureInput)
def capture_webcam_image(device_index: int = DEFAULT_CAMERA_DEVICE_INDEX) -> str:
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
        scale = min(1.0, MAX_IMAGE_WIDTH / width, MAX_IMAGE_HEIGHT / height)
        if scale < 1.0:
            frame = cv2.resize(
                frame,
                (max(1, int(width * scale)), max(1, int(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        encoded, image = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        )
        if not encoded:
            return "Camera frame could not be encoded; no image was sent."
        image_bytes = image.tobytes()
        if len(image_bytes) > MAX_CAPTURED_IMAGE_BYTES:
            return (
                f"Captured image exceeded the {MAX_CAPTURED_IMAGE_BYTES}-byte "
                "safety limit."
            )
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


@tool(args_schema=WorkspaceImageInput)
def load_workspace_image(file_path: str) -> str:
    """Load a bounded workspace image for one approved local vision-model request."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return (
            "Image loading is optional. Install media support with "
            "`python -m pip install '.[media]'`."
        )

    if os.path.splitext(file_path)[1].lower() not in {
        ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"
    }:
        return "Error: Workspace image must use a supported image file extension."

    try:
        target = SandboxManager.validate_path(file_path)
        if not target.is_file():
            return "Error: Workspace image path is not a regular file."
        with target.open("rb") as image_file:
            image_bytes = image_file.read(MAX_CAPTURED_IMAGE_BYTES + 1)
        if len(image_bytes) > MAX_CAPTURED_IMAGE_BYTES:
            return (
                f"Error: Image exceeds the {MAX_CAPTURED_IMAGE_BYTES}-byte "
                "safety limit."
            )
        if not image_bytes:
            return "Error: Image file is empty."
        from PIL import Image

        try:
            with Image.open(io.BytesIO(image_bytes)) as image_header:
                source_width, source_height = image_header.size
        except Exception as exc:
            return f"Error: File could not be decoded as an image ({exc})."
        if (
            source_width < 1
            or source_height < 1
            or source_width * source_height > MAX_DECODED_IMAGE_PIXELS
        ):
            return (
                f"Error: Image dimensions are invalid or exceed the "
                f"{MAX_DECODED_IMAGE_PIXELS}-pixel safety limit."
            )
        frame = cv2.imdecode(
            np.frombuffer(image_bytes, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if frame is None or getattr(frame, "ndim", 0) < 2:
            return "Error: File could not be decoded as an image."
        height, width = frame.shape[:2]
        if height < 1 or width < 1 or height * width > MAX_DECODED_IMAGE_PIXELS:
            return "Error: Decoded image exceeds the validated pixel safety limit."
        scale = min(1.0, MAX_IMAGE_WIDTH / width, MAX_IMAGE_HEIGHT / height)
        if scale < 1.0:
            frame = cv2.resize(
                frame,
                (max(1, int(width * scale)), max(1, int(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        encoded, image = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        )
        if not encoded:
            return "Error: Image could not be normalized; no image was sent."
        normalized_image = image.tobytes()
        if len(normalized_image) > MAX_CAPTURED_IMAGE_BYTES:
            return (
                f"Error: Normalized image exceeds the "
                f"{MAX_CAPTURED_IMAGE_BYTES}-byte safety limit."
            )
        global _image_capture_count
        with _image_lock:
            if _image_capture_count >= MAX_IMAGES_PER_TURN:
                return (
                    f"Error: Local image limit reached ({MAX_IMAGES_PER_TURN} "
                    "images per user request)."
                )
            _image_capture_count += 1
            reference = uuid.uuid4().hex
            _captured_images[reference] = normalized_image
        return (
            f"Image loaded in memory as workspace-image:{reference}. "
            "Analyze the attached image; it is not copied or saved."
        )
    except (OSError, PermissionError, ValueError) as exc:
        return f"Error loading workspace image: {exc}"
    except Exception as exc:
        return f"Image processing failed ({type(exc).__name__}): {exc}"


@tool(args_schema=MicrophoneInput)
def record_microphone_audio(
    duration_seconds: float = DEFAULT_MICROPHONE_SECONDS,
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

    audio = None
    try:
        audio = sd.rec(
            int(duration_seconds * MICROPHONE_SAMPLE_RATE),
            samplerate=MICROPHONE_SAMPLE_RATE,
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
                wav_file.setframerate(MICROPHONE_SAMPLE_RATE)
                wav_file.writeframes(audio.tobytes())
            audio_file.seek(0)
            transcript = _transcribe_local_audio(audio_file)
        if not transcript:
            return "No speech was recognized in the recording."
        return f"Offline microphone transcript ({duration_seconds:.1f}s): {transcript}"
    except Exception as exc:
        return f"Offline transcription failed ({type(exc).__name__}): {exc}"
