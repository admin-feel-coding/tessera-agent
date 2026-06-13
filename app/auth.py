import secrets

from fastapi import Header, HTTPException

from app.config import settings


async def verify_internal_key(
    x_internal_key: str | None = Header(default=None, alias="X-Internal-Key"),
) -> None:
    if x_internal_key is None or not secrets.compare_digest(
        x_internal_key, settings.internal_api_key
    ):
        raise HTTPException(status_code=401, detail="Invalid or missing X-Internal-Key.")
