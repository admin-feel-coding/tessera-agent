import structlog

from app.clients.tessera_data import TesseraDataClient
from app.config import settings
from app.schemas.feedback import FeedbackPayload
from app.services.embedding import embed

log = structlog.get_logger(__name__)


async def process_feedback(payload: FeedbackPayload) -> None:
    log.info(
        "feedback_received",
        transaction_id=payload.transaction_id,
        corrected_decision=payload.corrected_decision,
    )

    embedding = await embed(f"{payload.analyst_note} {payload.original_verdict.reasoning}")

    case_data: dict = {
        "transaction_id": payload.transaction_id,
        "decision": payload.corrected_decision,
        "reasoning": payload.analyst_note or payload.original_verdict.reasoning,
        "signals": payload.original_verdict.signals.model_dump(),
        "embedding": embedding,
        "metadata": {
            "source": "analyst_feedback",
            "decisive_signals": payload.decisive_signals,
            "original_decision": payload.original_verdict.decision,
        },
    }

    client = TesseraDataClient(settings.tessera_data_url, settings.internal_api_key)
    try:
        result = await client.save_case(case_data)
    finally:
        await client.close()

    case_id = result.get("id", "")
    if case_id:
        log.info(
            "feedback_persisted",
            transaction_id=payload.transaction_id,
            case_id=case_id,
        )
    else:
        log.warn(
            "feedback_save_failed",
            transaction_id=payload.transaction_id,
        )
