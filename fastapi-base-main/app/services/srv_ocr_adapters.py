"""Adapters plugging the teammates' modules into the OCR pipeline.

- LLM: app.services.srv_llm (feature/llm) -> pipeline `CodeLlm`
- Sandbox: app.sandbox (toilavu) -> pipeline `Sandbox`

Both modules only handle Python for now, so the pipeline is limited to Python:
other languages skip the syntax check (C/C++/Java need a sandbox backend first).
"""

import base64
import time
from typing import Optional

from app.sandbox.base import RunStatus
from app.sandbox.precheck import precheck
from app.schemas.sche_llm import CorrectOcrRequest, OcrLine
from app.schemas.sche_ocr_pipeline import (
    Correction,
    LayoutResult,
    LlmCallRecord,
    LlmTranscription,
    ReviewItem,
    SandboxResult,
    SyntaxIssue,
)
from app.services.srv_llm import LLMService
from app.utils.enums import LlmPurpose

LANGUAGE = "python"
LANGUAGE_ALIASES = {"py": "python", "python3": "python"}


def canonical_language(language: Optional[str]) -> str:
    key = (language or "unknown").strip().lower()
    return LANGUAGE_ALIASES.get(key, key)


class LlmServiceAdapter:
    """`CodeLlm` backed by LLMService.correct_ocr."""

    def __init__(self, service: Optional[LLMService] = None):
        self.service = service or LLMService()

    def transcribe(
        self, image_jpeg: bytes, draft: LayoutResult, syntax_errors: list[SyntaxIssue], log
    ) -> LlmTranscription:
        # correct_ocr has no field for the sandbox errors yet, so `syntax_errors` is not sent.
        ocr_text = draft.render()
        request = CorrectOcrRequest(
            ocr_text=ocr_text,
            lines=[
                OcrLine(text=text, score=line.confidence)
                for text, line in zip(ocr_text.split("\n"), draft.lines)
            ],
            language=LANGUAGE,
            image_base64=base64.b64encode(image_jpeg).decode("ascii"),
        )
        record = LlmCallRecord(purpose=LlmPurpose.TRANSCRIBE.value, model=self.service.client.model, prompt=ocr_text)
        started = time.monotonic()
        try:
            response = self.service.correct_ocr(request)
        except Exception as e:
            record.latency_ms = int((time.monotonic() - started) * 1000)
            record.response = self.service.client.last_raw or None
            record.error = getattr(e, "message", None) or str(e)
            log(record)
            raise
        record.latency_ms = int((time.monotonic() - started) * 1000)
        record.response = response.model_dump_json()
        record.parse_ok = True
        log(record)

        return LlmTranscription(
            language=LANGUAGE,
            code=response.code,
            corrections=[
                Correction(line=c.line, old=c.before, new=c.after, reason=c.reason) for c in response.changes
            ],
            uncertain=[
                ReviewItem(line=u.line or None, reason=f"LLM không chắc '{u.text}': {u.note}", source="llm")
                for u in response.uncertain
            ],
            needs_review=response.needs_review,
            review_reasons=response.review_reasons,
        )


class PrecheckSandbox:
    """`Sandbox` backed by app.sandbox.precheck: parses the code, never runs it."""

    def supports(self, language: str) -> bool:
        return canonical_language(language) == LANGUAGE

    def check(self, code: str, language: str) -> SandboxResult:
        started = time.monotonic()
        result = precheck(code)
        duration_ms = int((time.monotonic() - started) * 1000)
        if result is None:
            return SandboxResult(
                language=LANGUAGE, image="in-process", command="app.sandbox.precheck",
                exit_code=0, ok=True, duration_ms=duration_ms,
            )
        message = result.stderr if result.status is RunStatus.SYNTAX_ERROR else f"{result.status.value}: {result.stderr}"
        return SandboxResult(
            language=LANGUAGE, image="in-process", command="app.sandbox.precheck",
            exit_code=1, ok=False, errors=[SyntaxIssue(line=result.error_line, message=message)],
            output=result.stderr, duration_ms=duration_ms,
        )
