from fastapi import APIRouter, Depends, Query

from app.auth import verify_internal_key
from app.clients.tessera_data import TesseraDataClient
from app.config import settings

router = APIRouter()


@router.get("/verdicts", dependencies=[Depends(verify_internal_key)])
async def list_verdicts(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> dict:
    client = TesseraDataClient(settings.tessera_data_url, settings.internal_api_key)
    try:
        result = await client.list_verdicts(limit=limit, offset=offset)
    finally:
        await client.close()
    return result
