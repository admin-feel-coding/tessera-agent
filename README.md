# tessera-agent

AI fraud analyst agent service — Python / FastAPI.

## Setup

```bash
uv sync
cp .env.example .env  # fill in your API keys
```

## Run

```bash
uv run fastapi dev app/main.py --port 8001
```

## Test

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```
