import pytest

from app.services.embedding import embed


@pytest.mark.asyncio
async def test_embed_returns_1536_floats() -> None:
    result = await embed("hello")
    assert len(result) == 1536


@pytest.mark.asyncio
async def test_embed_is_deterministic() -> None:
    first = await embed("hello")
    second = await embed("hello")
    assert first == second


@pytest.mark.asyncio
async def test_embed_values_in_range() -> None:
    result = await embed("hello")
    assert all(-1.0 <= v <= 1.0 for v in result)
