# Kokoro-82M on ROCm

Self-hosted [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) text-to-speech
running on **AMD GPUs via ROCm**, in a container, behind a small HTTP service. It
exposes both a native API and an **OpenAI-compatible** `/v1/audio/speech` endpoint.

Kokoro is a tiny (82M-parameter, Apache-2.0) open-weight model — fast and light
enough to run comfortably even on an integrated GPU. This image ships **English**
(American + British) voices.

## Requirements

- AMD GPU with ROCm support and the `amdgpu` driver loaded (`/dev/kfd` + `/dev/dri` present).
  - Officially supported cards (e.g. RX 7900 / gfx1100) work out of the box.
  - Unsupported GPUs (e.g. the Radeon 780M / gfx1103 iGPU) also work, with one extra
    step — see [Unsupported GPUs](#unsupported-gpus).
- Docker + the Compose plugin. (No ROCm install needed on the host beyond the driver —
  the container ships its own.)
- < 1 GB disk for the model + voice weights.

### Unsupported GPUs

GPUs not on ROCm's official list (e.g. the Radeon 780M / gfx1103) need an
`HSA_OVERRIDE_GFX_VERSION` to masquerade as a supported arch. This is **machine-local**
and deliberately not shipped — create a gitignored `docker-compose.override.yml` (Compose
merges it automatically):

```yaml
services:
  kokoro-tts:
    environment: { HSA_OVERRIDE_GFX_VERSION: "11.0.0" }
  download:
    environment: { HSA_OVERRIDE_GFX_VERSION: "11.0.0" }
```

`11.0.0` (gfx1100) works for the 780M. Supported cards need none of this.

## Getting started

```bash
# 1. Record your GPU's device-node group IDs into .env (one-time, per host)
echo "RENDER_GID=$(getent group render | cut -d: -f3)"  > .env
echo "VIDEO_GID=$(getent group video | cut -d: -f3)"    >> .env

# 2. Download the model + default voice weights once (network on, no server yet)
docker compose --profile download run --rm download

# 3. Start the service on http://127.0.0.1:8002
docker compose up --build -d

# 4. Check it's healthy and on the GPU
curl -s http://127.0.0.1:8002/health
```

> The bundled `.env` targets officially-supported GPUs. On an unsupported GPU like the
> Radeon 780M, do the one extra step in [Unsupported GPUs](#unsupported-gpus) first.

Quick test:

```bash
./smoke-test.sh        # exercises /health, /voices, /tts and the OpenAI endpoint
```

## Choosing a voice

Voices are named `{lang}{gender}_{name}` — the **first letter is the language**
(`a` = American English, `b` = British English), the second is gender (`f`/`m`).
You only pick a voice; the language follows from it. List them live:

```bash
curl -s http://127.0.0.1:8002/voices
```

American: `af_heart` (default), `af_bella`, `af_nova`, `af_sky`, `am_adam`, `am_echo`,
`am_onyx`, `am_puck`, … British: `bf_emma`, `bf_alice`, `bm_fable`, `bm_george`, …

Set the **default** voice (used when a request omits one) in `.env`:

```ini
KOKORO_VOICE=bm_fable
```

then `docker compose up -d` to apply. Any request can still override per-call.

## Using the API

The service is reachable at `http://127.0.0.1:8002`. Two interfaces are available.

### OpenAI-compatible (drop-in)

`POST /v1/audio/speech` matches OpenAI's TTS API, so existing OpenAI client code/SDKs
work by just pointing the base URL here. OpenAI voice names (`alloy`, `nova`, …) are
mapped onto Kokoro voices; native names (`af_heart`, `bm_fable`, …) also work.

```bash
curl -s http://127.0.0.1:8002/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"tts-1","input":"Hello there.","voice":"nova","response_format":"mp3"}' \
  -o hello.mp3
```

| Field | Notes |
|-------|-------|
| `input` | text to speak (required) |
| `voice` | OpenAI name or Kokoro voice; omitted → `KOKORO_VOICE` |
| `response_format` | `mp3` (default), `opus`, `aac`, `flac`, `wav`, `pcm` |
| `speed` | 0.25–4.0 (default 1.0) |
| `model` | accepted and ignored (one model is served) |
| `instructions` | accepted and ignored — Kokoro has no delivery control |

Python (official OpenAI SDK):

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8002/v1", api_key="not-needed")
client.audio.speech.create(model="tts-1", voice="bm_fable", input="Hello from Kokoro.").stream_to_file("out.mp3")
```

### Native

`POST /tts` returns a WAV by default:

```bash
curl -s http://127.0.0.1:8002/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello from Kokoro.","voice":"af_heart","speed":1.0}' \
  -o out.wav
```

**Which to use?** Use the **OpenAI** endpoint for compatibility with existing tooling and
for format options. Use **native** `/tts` for the simplest request shape. Both run the
same model at 24 kHz.

## Configuration (`.env`)

| Variable | Default | Purpose |
|----------|---------|---------|
| `KOKORO_VOICE` | `af_heart` | default voice (its first letter sets the language) |
| `KOKORO_REPO` | `hexgrad/Kokoro-82M` | model repo on Hugging Face |
| `KOKORO_DEVICE` | `cuda` | `cpu` to run without a GPU |
| `HSA_OVERRIDE_GFX_VERSION` | _(unset)_ | unsupported GPUs only — set via a local override (see above) |
| `RENDER_GID` / `VIDEO_GID` | host-specific | GPU device-node access (set in step 1) |

### Other languages

This image ships English only. Kokoro also supports Spanish, French, Italian,
Portuguese, Hindi (via espeak-ng) and Japanese / Mandarin (via `misaki[ja]` /
`misaki[zh]`). To add them, install the relevant package in the `Dockerfile` and use
the matching voices (e.g. `ff_siwis` for French, `jf_alpha` for Japanese).

## Notes & troubleshooting

- **`Permission denied` on `/dev/kfd`** — `RENDER_GID`/`VIDEO_GID` in `.env` don't match
  your host; regenerate them (step 1).
- **Prod RX 7900** — drop the `docker-compose.override.yml` entirely; `gfx1100` is
  officially supported and needs no HSA override.
- **Security** — the service binds to `127.0.0.1` only, runs as a non-root user with a
  read-only root filesystem and all Linux capabilities dropped. GPU passthrough (`/dev/kfd`)
  is not a hardware sandbox, but weights are fetched in a separate step so the serving
  container needs no internet access.
- **Performance** — Kokoro is tiny, so it's many times faster than real time even on the
  780M iGPU. The model is loaded once at startup and kept warm; pipelines are built
  per-language on first use.
