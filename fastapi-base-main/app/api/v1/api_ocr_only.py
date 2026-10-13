"""Raw OCR endpoint.

This endpoint intentionally runs PaddleOCR only. It does not create an OCR
submission, call the LLM, run the sandbox, or write a trace to PostgreSQL.
"""

import os
from typing import Any

from fastapi import APIRouter, File, UploadFile, status

from app.core.config import settings
from app.engines.ocr_paddle import PaddleOcrEngine
from app.schemas.sche_ocr_only import OcrOnlyResponse
from app.schemas.sche_response import DataResponse
from app.utils.exception_handler import CustomException
from app.utils.image import InvalidImage


router = APIRouter(prefix="/ocr")

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# PaddleOCR is loaded lazily by the engine and reused by subsequent requests.
ocr_engine = PaddleOcrEngine(
    lang=settings.OCR_LANG,
    device=settings.OCR_DEVICE,
    det_model=settings.OCR_DET_MODEL,
    rec_model=settings.OCR_REC_MODEL,
)


@router.post(
    "/raw",
    response_model=DataResponse[OcrOnlyResponse],
    status_code=status.HTTP_200_OK,
    summary="Run OCR only",
    description="Extract text with PaddleOCR only; does not call LLM or sandbox.",
)
def run_raw_ocr(file: UploadFile = File(...)) -> Any:
    extension = os.path.splitext(file.filename or "")[1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise CustomException(
            http_code=400,
            message=f"Unsupported file type, allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    max_bytes = settings.OCR_MAX_UPLOAD_MB * 1024 * 1024
    data = file.file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise CustomException(
            http_code=413,
            message=f"File is larger than {settings.OCR_MAX_UPLOAD_MB}MB",
        )

    try:
        boxes = ocr_engine.predict(data)
    except InvalidImage as exc:
        raise CustomException(http_code=400, message=str(exc)) from exc

    result = OcrOnlyResponse(
        filename=file.filename or "unknown",
        text="\n".join(box.text for box in boxes),
        boxes=boxes,
    )
    return DataResponse(http_code=status.HTTP_200_OK, data=result)
