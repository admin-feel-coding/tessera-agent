import json

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.auth import verify_internal_key
from app.clients.tessera_data import TesseraDataClient
from app.config import settings
from app.schemas.transaction import Transaction
from app.schemas.verdict import Verdict
from app.services.analyze import IS_MOCK, stream_mock, stream_real
from app.services.analyze import analyze as analyze_service

router = APIRouter()


@router.post("/analyze", dependencies=[Depends(verify_internal_key)])
async def analyze(transaction: Transaction) -> Verdict:
    return await analyze_service(transaction)


async def _to_sse(gen):
    async for event in gen:
        yield f"event: {event.type}\ndata: {json.dumps(event.data)}\n\n"


@router.post("/analyze/stream", dependencies=[Depends(verify_internal_key)])
async def analyze_stream(transaction: Transaction, request: Request) -> StreamingResponse:
    trace_id = request.headers.get("X-Trace-ID", "")
    data_client = TesseraDataClient(settings.tessera_data_url, settings.internal_api_key)
    gen = (
        stream_mock(transaction, data_client, trace_id)
        if IS_MOCK
        else stream_real(transaction, data_client, trace_id)
    )
    return StreamingResponse(
        _to_sse(gen),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
