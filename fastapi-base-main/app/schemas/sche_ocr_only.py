"""Schemas for the raw OCR-only endpoint."""

from pydantic import BaseModel

from app.schemas.sche_ocr_pipeline import OcrBox


class OcrOnlyResponse(BaseModel):
    """Raw text detected by PaddleOCR, without LLM or sandbox processing."""

    filename: str
    text: str
    boxes: list[OcrBox]
