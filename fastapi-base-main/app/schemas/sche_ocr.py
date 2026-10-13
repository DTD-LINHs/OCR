from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class OcrSubmissionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    status: str
    run: int
    original_filename: Optional[str] = None
    language: Optional[str] = None
    final_code: Optional[str] = None
    syntax_ok: Optional[bool] = None
    syntax_errors: Optional[list[dict[str, Any]]] = None
    need_review: bool
    review_reasons: Optional[list[str]] = None
    uncertain_lines: Optional[list[int]] = None
    llm_used: Optional[bool] = None
    corrections: Optional[list[dict[str, Any]]] = None
    review_items: Optional[list[dict[str, Any]]] = None
    confidence: Optional[float] = None
    timings: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float
    updated_at: float


class OcrCodeVersionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    run: int
    stage: str
    fix_round: int
    language: Optional[str] = None
    code: str
    corrections: Optional[list[dict[str, Any]]] = None
    created_at: float


class OcrSandboxCheckResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    run: int
    code_version_id: Optional[int] = None
    language: str
    image: Optional[str] = None
    exit_code: Optional[int] = None
    ok: bool
    timed_out: bool
    errors: Optional[list[dict[str, Any]]] = None
    output: Optional[str] = None
    duration_ms: Optional[int] = None
    created_at: float


class OcrLlmCallResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    run: int
    purpose: str
    model: Optional[str] = None
    prompt: Optional[str] = None
    response: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    latency_ms: Optional[int] = None
    parse_ok: bool
    error: Optional[str] = None
    created_at: float


class OcrTraceResponse(BaseModel):
    submission: OcrSubmissionResponse
    versions: list[OcrCodeVersionResponse]
    sandbox_checks: list[OcrSandboxCheckResponse]
    llm_calls: list[OcrLlmCallResponse]
