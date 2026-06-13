import os

# Set required env vars before any app module is imported.
# This keeps tests hermetic — no .env file needed in CI.
os.environ.setdefault("INTERNAL_API_KEY", "test-secret")
os.environ.setdefault("ANTHROPIC_API_KEY", "placeholder-replace-me")
