import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from app.auth import verify_internal_key

router = APIRouter()

_RESULTS_PATH = Path("evals/results/latest.json")


@router.get("/evals/latest", dependencies=[Depends(verify_internal_key)])
async def get_latest_eval() -> dict:
    if not _RESULTS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="no eval runs yet — run `uv run python -m evals`",
        )
    return json.loads(_RESULTS_PATH.read_text())
