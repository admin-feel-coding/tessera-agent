from fastapi import APIRouter, Depends

from app.auth import verify_internal_key
from app.schemas.transaction import Transaction
from app.schemas.verdict import Verdict
from app.services.analyze import analyze as analyze_service

router = APIRouter()


@router.post("/analyze", dependencies=[Depends(verify_internal_key)])
async def analyze(transaction: Transaction) -> Verdict:
    return await analyze_service(transaction)
