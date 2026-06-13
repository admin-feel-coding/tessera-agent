from __future__ import annotations

import hashlib
import random

import structlog

from app.config import settings

log = structlog.get_logger(__name__)

_PLACEHOLDER_KEYS = ("", "placeholder-replace-me")
_DIM = 1536


def _mock_embed(text: str, dim: int = _DIM) -> list[float]:
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
    rng = random.Random(seed)
    return [rng.uniform(-1.0, 1.0) for _ in range(dim)]


async def embed(text: str) -> list[float]:
    if settings.anthropic_api_key not in _PLACEHOLDER_KEYS:
        # TODO: replace with real embedding API call when available
        log.warning(
            "embedding_mock_in_real_mode",
            msg="Real API key is set but using mock embedding — integrate a real embedding model.",
        )

    return _mock_embed(text)
