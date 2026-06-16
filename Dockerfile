# Kokoro-82M TTS on ROCm.
#
# Base ships a ROCm build of PyTorch (torch presents to CUDA APIs as if it were
# CUDA, so device="cuda" works on AMD). Do NOT reinstall torch — kokoro only
# needs torch (no version pin), and pip won't upgrade an already-satisfied dep,
# so the ROCm wheel survives. The hip assert below is the safety net.
FROM rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0

# espeak-ng: kokoro's G2P fallback for out-of-dictionary English words (and the
# backend for non-English languages). libsndfile1 for soundfile; ffmpeg to encode
# mp3/opus/aac for the OpenAI-compatible endpoint.
RUN apt-get update \
    && apt-get install -y --no-install-recommends espeak-ng libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# English only (lang codes a + b) — no misaki[ja]/[zh] extras needed.
RUN pip install --no-cache-dir \
        "kokoro>=0.9.4" "fastapi>=0.115" "uvicorn[standard]>=0.30" soundfile

# misaki (Kokoro's English G2P) lazily pip-installs the spaCy model on first use;
# that write fails on the read-only rootfs at runtime, so bake it in now.
RUN python -m spacy download en_core_web_sm

RUN python -c "import torch; assert torch.version.hip, 'torch is not a ROCm build! hip=%r' % torch.version.hip; print('OK: torch', torch.__version__, 'hip', torch.version.hip)"

RUN useradd -m -u 1001 appuser
WORKDIR /app
COPY server.py /app/

# Weights at /hf (outside ~/.cache, which is a writable tmpfs at runtime).
# Read-only rootfs needs all cache writes redirected; ~/.cache tmpfs covers the
# ROCm/HIP/MIOpen kernel caches — a read-only rootfs otherwise turns those writes
# into a GPU hang. NUMBA_CACHE_DIR matters because librosa (pulled in by kokoro)
# uses @jit(cache=True) and won't import without a writable cache dir.
ENV HF_HOME=/hf \
    NUMBA_CACHE_DIR=/tmp/numba \
    MPLCONFIGDIR=/tmp/mpl \
    XDG_CACHE_HOME=/tmp/xdg \
    MIOPEN_FIND_MODE=2 \
    MIOPEN_USER_DB_PATH=/tmp/miopen \
    MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen \
    KOKORO_REPO=hexgrad/Kokoro-82M \
    KOKORO_VOICE=af_heart \
    KOKORO_LANG=a \
    KOKORO_DEVICE=cuda \
    PORT=8000
RUN mkdir -p /hf && chown -R appuser:appuser /hf
USER appuser
EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
