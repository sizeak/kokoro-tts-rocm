#!/usr/bin/env bash
# Quick end-to-end check against a running service (default 127.0.0.1:8002).
set -euo pipefail
BASE="${1:-http://127.0.0.1:8002}"

echo "== /health =="
curl -fsS "$BASE/health"; echo

echo "== /voices =="
curl -fsS "$BASE/voices" | head -c 400; echo; echo

echo "== /tts (default voice -> WAV) =="
curl -fsS "$BASE/tts" \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello from Kokoro on ROCm."}' \
  -o smoke-default.wav
echo "wrote smoke-default.wav ($(stat -c%s smoke-default.wav) bytes)"

echo "== /tts (British voice, faster) =="
curl -fsS "$BASE/tts" \
  -H 'Content-Type: application/json' \
  -d '{"text":"And now, a little quicker.","voice":"bm_fable","speed":1.3}' \
  -o smoke-british.wav
echo "wrote smoke-british.wav ($(stat -c%s smoke-british.wav) bytes)"

echo "== /v1/audio/speech (OpenAI voice 'nova' -> MP3) =="
curl -fsS "$BASE/v1/audio/speech" \
  -H 'Content-Type: application/json' \
  -d '{"model":"tts-1","input":"OpenAI-compatible endpoint works.","voice":"nova","response_format":"mp3"}' \
  -o smoke-openai.mp3
echo "wrote smoke-openai.mp3 ($(stat -c%s smoke-openai.mp3) bytes)"

echo "All smoke checks passed."
