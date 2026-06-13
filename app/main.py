from fastapi import FastAPI

from app.config import settings
from app.errors import register_exception_handlers
from app.handlers.analyze import router as analyze_router
from app.handlers.feedback import router as feedback_router
from app.handlers.health import router as health_router
from app.handlers.verdicts import router as verdicts_router
from app.logging_config import configure_logging

configure_logging(settings.log_level)

app = FastAPI(title="Tessera Agent", version="0.1.0")

register_exception_handlers(app)

app.include_router(health_router)
app.include_router(analyze_router)
app.include_router(feedback_router)
app.include_router(verdicts_router)
