# syntax=docker/dockerfile:1
# Fanfare Studio: local text-to-speech server. Default: Chatterbox 4-bit ONNX on CPU (best quality
# in the listening tests, see docs/model-choice.md). Build variants via build args:
#   TORCH_EXTRA = cpu (default) | rocm | cuda   which PyTorch wheels (PerTh watermarker + torch backend)
#   EXTRAS      = space-separated extras to add, e.g. "onnx" (default), "onnx piper", "chatterbox"

# ---- builder: resolve + install into /app/.venv with uv --------------------------------------
FROM python:3.12-slim-bookworm AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /usr/local/bin/uv
ARG TORCH_EXTRA=cpu
ARG EXTRAS="onnx"
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --extra "${TORCH_EXTRA}" \
        $(for e in ${EXTRAS}; do printf -- '--extra %s ' "$e"; done)
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra "${TORCH_EXTRA}" \
        $(for e in ${EXTRAS}; do printf -- '--extra %s ' "$e"; done)

# ---- runtime ------------------------------------------------------------------------------------
FROM python:3.12-slim-bookworm
RUN useradd --system --uid 10001 --create-home --home-dir /home/fanfare fanfare \
 && mkdir -p /models /data /config /app && chown fanfare:fanfare /models /data /config
COPY --from=builder --chown=fanfare:fanfare /app/.venv /app/.venv
COPY --chown=fanfare:fanfare voices.toml eval.toml /app/
COPY --chown=fanfare:fanfare voices /app/voices
WORKDIR /app
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/huggingface \
    FANFARE_STUDIO_DATA=/data \
    FANFARE_STUDIO_CONFIG_DIR=/config \
    FANFARE_HOST=0.0.0.0 \
    NUMBA_CACHE_DIR=/tmp/numba \
    MPLCONFIGDIR=/tmp/mpl
# Models (about 3 GB for the default backend) download into this volume on first start.
VOLUME ["/models", "/data", "/config"]
USER fanfare
EXPOSE 5123
# Request without an Origin header (not a web page): always allowed.
HEALTHCHECK --interval=15s --timeout=5s --start-period=900s --retries=5 \
    CMD ["python", "-c", "import urllib.request as u; u.urlopen('http://127.0.0.1:5123/voices', timeout=4).read()"]
# Binds 0.0.0.0 INSIDE the container only; publish on 127.0.0.1 of the host (see compose.yaml).
ENTRYPOINT ["fanfare-studio"]
CMD ["--backend", "chatterbox-onnx"]
