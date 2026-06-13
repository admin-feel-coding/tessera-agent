FROM python:3.12-slim AS builder
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY app/ ./app/

FROM python:3.12-slim
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY --from=builder /app/.venv ./.venv
COPY --from=builder /app/app ./app
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8001
CMD ["fastapi", "run", "app/main.py", "--port", "8001", "--host", "0.0.0.0"]
