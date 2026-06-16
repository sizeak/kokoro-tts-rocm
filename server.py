"""Kokoro-82M TTS as a small HTTP service (native + OpenAI-compatible).

The model loads once at startup and stays warm on the GPU. Kokoro is per-language:
a `KPipeline` is created per language code and cached. The language code is the
first letter of the voice name (``a``=American, ``b``=British English), so callers
only ever pick a voice — the language follows from it.
"""

from __future__ import annotations

import io
import os
import subprocess
from contextlib import asynccontextmanager

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

SAMPLE_RATE = 24000

REPO = os.environ.get("KOKORO_REPO", "hexgrad/Kokoro-82M")
DEFAULT_VOICE = os.environ.get("KOKORO_VOICE", "af_heart")
DEVICE = os.environ.get("KOKORO_DEVICE", "cuda")

# Voices shipped by Kokoro for the English language codes (a = American, b =
# British). Other languages exist upstream but need espeak-ng language data
# and/or misaki extras that this English-only image doesn't install.
VOICES = [
    # American female / male
    "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore",
    "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael",
    "am_onyx", "am_puck", "am_santa",
    # British female / male
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
]
VOICE_SET = set(VOICES)

# OpenAI voice names mapped onto the closest Kokoro voice. Several line up by name.
OPENAI_VOICE_MAP = {
    "alloy": "af_alloy",
    "echo": "am_echo",
    "fable": "bm_fable",
    "onyx": "am_onyx",
    "nova": "af_nova",
    "shimmer": "af_sky",
    "ash": "am_puck",
    "ballad": "bm_george",
    "coral": "af_kore",
    "sage": "af_sarah",
    "verse": "am_eric",
}

SUPPORTED_FORMATS = {"mp3", "opus", "aac", "flac", "wav", "pcm"}

# Lazily-built KPipeline per language code, e.g. {"a": <KPipeline>}.
_pipelines: dict[str, object] = {}
_state: dict[str, object] = {}


def _get_pipeline(lang_code: str):
    """Return a warm KPipeline for the language code, building it on first use.

    All pipelines share one underlying KModel (loaded once, on the GPU) so adding
    a language doesn't reload the 82M weights.
    """
    pipe = _pipelines.get(lang_code)
    if pipe is None:
        from kokoro import KPipeline

        pipe = KPipeline(lang_code=lang_code, repo_id=REPO, model=_state["model"])
        _pipelines[lang_code] = pipe
    return pipe


@asynccontextmanager
async def lifespan(_: FastAPI):
    from kokoro import KModel

    model = KModel(repo_id=REPO).to(DEVICE).eval()
    _state["model"] = model
    # Warm the default voice's pipeline so the first request isn't cold.
    _get_pipeline(DEFAULT_VOICE[0])
    yield
    _pipelines.clear()
    _state.clear()


app = FastAPI(title="Kokoro-82M TTS (ROCm)", lifespan=lifespan)


def resolve_voice(voice: str | None) -> str:
    """Map an OpenAI voice name onto a Kokoro voice, pass through native names,
    or fall back to the configured default. Raises 400 on an unknown name."""
    if not voice:
        return DEFAULT_VOICE
    if voice in VOICE_SET:
        return voice
    mapped = OPENAI_VOICE_MAP.get(voice.lower())
    if mapped:
        return mapped
    raise HTTPException(
        status_code=400,
        detail=f"unknown voice {voice!r}; see GET /voices",
    )


def synthesize(text: str, voice: str, speed: float) -> np.ndarray:
    """Run Kokoro and concatenate its streamed chunks into one float32 waveform."""
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="text/input must not be empty")
    pipeline = _get_pipeline(voice[0])
    chunks: list[np.ndarray] = []
    for result in pipeline(text, voice=voice, speed=speed):
        audio = result.audio
        if audio is None:
            continue
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()
        chunks.append(np.asarray(audio, dtype=np.float32))
    if not chunks:
        raise HTTPException(status_code=500, detail="model produced no audio")
    return np.concatenate(chunks)


def _ffmpeg_encode(wav_bytes: bytes, args: list[str]) -> bytes:
    """Pipe a WAV through ffmpeg to a compressed format (no temp files, so this
    works on a read-only root filesystem)."""
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0", *args, "pipe:1"],
        input=wav_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail=f"ffmpeg failed: {proc.stderr.decode('utf-8', 'replace')[:500]}",
        )
    return proc.stdout


def encode_audio(wav: np.ndarray, fmt: str) -> tuple[bytes, str]:
    """Encode a float32 waveform to the requested format; return (bytes, mime)."""
    fmt = fmt.lower()
    if fmt not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported format {fmt!r}; one of {sorted(SUPPORTED_FORMATS)}",
        )
    if fmt == "pcm":  # raw signed 16-bit little-endian, no header
        pcm = np.clip(wav, -1.0, 1.0)
        return (pcm * 32767.0).astype("<i2").tobytes(), "audio/pcm"

    buf = io.BytesIO()
    if fmt == "wav":
        sf.write(buf, wav, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        return buf.getvalue(), "audio/wav"
    if fmt == "flac":
        sf.write(buf, wav, SAMPLE_RATE, format="FLAC")
        return buf.getvalue(), "audio/flac"

    # Compressed: render a WAV in memory, then transcode via ffmpeg.
    sf.write(buf, wav, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    wav_bytes = buf.getvalue()
    if fmt == "mp3":
        return _ffmpeg_encode(wav_bytes, ["-f", "mp3", "-b:a", "128k"]), "audio/mpeg"
    if fmt == "opus":
        return _ffmpeg_encode(wav_bytes, ["-f", "opus", "-b:a", "64k"]), "audio/opus"
    if fmt == "aac":
        return _ffmpeg_encode(wav_bytes, ["-f", "adts", "-b:a", "128k"]), "audio/aac"
    raise HTTPException(status_code=400, detail=f"unsupported format {fmt!r}")


@app.get("/health")
def health():
    model_loaded = "model" in _state
    return {
        "status": "ok" if model_loaded else "loading",
        "model": REPO,
        "device": DEVICE,
        "torch_hip_version": torch.version.hip,
        "cuda_available": torch.cuda.is_available(),
        "default_voice": DEFAULT_VOICE,
    }


@app.get("/voices")
def voices():
    return {"voices": VOICES, "openai_aliases": OPENAI_VOICE_MAP, "sample_rate": SAMPLE_RATE}


class TTSRequest(BaseModel):
    text: str
    voice: str | None = None
    speed: float = 1.0
    format: str = "wav"


@app.post("/tts")
def tts(req: TTSRequest):
    voice = resolve_voice(req.voice)
    wav = synthesize(req.text, voice, req.speed)
    data, mime = encode_audio(wav, req.format)
    return Response(content=data, media_type=mime)


class SpeechRequest(BaseModel):
    input: str
    model: str | None = None  # accepted and ignored (one model is served)
    voice: str | None = None
    response_format: str = "mp3"
    speed: float = 1.0
    instructions: str | None = None  # accepted and ignored (Kokoro has no instruct)


@app.post("/v1/audio/speech")
def openai_speech(req: SpeechRequest):
    voice = resolve_voice(req.voice)
    wav = synthesize(req.input, voice, req.speed)
    data, mime = encode_audio(wav, req.response_format)
    return Response(content=data, media_type=mime)


@app.get("/v1/models")
def list_models():
    return JSONResponse(
        {"object": "list", "data": [{"id": REPO, "object": "model", "owned_by": "hexgrad"}]}
    )
