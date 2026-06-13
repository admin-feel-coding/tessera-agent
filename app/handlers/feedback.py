from fastapi import APIRouter, Depends

from app.auth import verify_internal_key
from app.schemas.feedback import FeedbackPayload
from app.services.feedback import process_feedback as feedback_service

router = APIRouter()


@router.post("/feedback", status_code=202, dependencies=[Depends(verify_internal_key)])
async def feedback(payload: FeedbackPayload) -> dict:
    await feedback_service(payload)
    return {"accepted": True}
