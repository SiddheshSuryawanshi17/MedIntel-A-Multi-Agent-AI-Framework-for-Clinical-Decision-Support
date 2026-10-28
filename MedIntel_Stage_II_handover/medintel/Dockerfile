# MedIntel server: MCP at /mcp, REST API at /v1 (docs at /docs), dashboard at /dashboard.
# Build: docker build -t medintel .   Run: see the README (Deploying).
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.21 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/huggingface \
    MEDINTEL_HOME=/data

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project
COPY src ./src
COPY config ./config
RUN uv sync --locked --no-dev

# Train the risk model (downloads UCI Heart Disease, CC BY 4.0) and cache the embedding model now, so the container
# starts without network access. The raw data is not kept in the image.
RUN uv run --no-sync medintel fetch-data && uv run --no-sync medintel train && rm -rf data
RUN uv run --no-sync python -c "from sentence_transformers import SentenceTransformer as S; S('sentence-transformers/all-MiniLM-L6-v2', device='cpu')"

RUN useradd --create-home --uid 10001 medintel && mkdir -p /data && chown medintel /data
USER medintel
VOLUME /data
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/healthz', timeout=3)"
CMD ["uv", "run", "--no-sync", "medintel", "serve", "--host", "0.0.0.0", "--port", "8000"]
