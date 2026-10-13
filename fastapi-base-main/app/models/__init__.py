# Import all the models, so that Base has them before being
# imported by Alembic
from app.models.model_base import Base  # noqa
from app.models.model_user import User  # noqa
from app.models.model_ocr import (  # noqa
    OcrSubmission,
    OcrCodeVersion,
    OcrSandboxCheck,
    OcrLlmCallLog,
)
