"""Adapters plugging the teammates' modules into the OCR pipeline.

- LLM: app.services.srv_llm (feature/llm) -> pipeline `CodeLlm`
- Sandbox: app.sandbox (toilavu) -> pipeline `Sandbox`

Both modules only handle Python for now, so the pipeline is limited to Python:
other languages skip the syntax check (C/C++/Java need a sandbox backend first).
"""

import base64
import time
from typing import Optional

from app.core.config import settings
from app.sandbox.base import RunStatus
from app.sandbox.docker_runner import DockerRunner
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
from app.services.srv_ocr_pipeline import SandboxUnavailable
from app.utils.enums import LlmPurpose

LANGUAGE = "python"
LANGUAGE_ALIASES = {"py": "python", "python3": "python"}


def _memory_to_mb(value: str) -> int:
    """Convert Docker memory config (for example ``256m``) to megabytes."""

    normalized = value.strip().lower()
    if normalized.endswith("m"):
        return max(int(float(normalized[:-1])), 1)
    if normalized.endswith("g"):
        return max(int(float(normalized[:-1]) * 1024), 1)
    return max(int(float(normalized)), 1)


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
        # Give the OCR-correction LLM the exact sandbox diagnostic so it can
        # distinguish an OCR transcription error from a student error.
        ocr_text = draft.render()
        request = CorrectOcrRequest(
            ocr_text=ocr_text,
            lines=[
                OcrLine(text=text, score=line.confidence)
                for text, line in zip(ocr_text.split("\n"), draft.lines)
            ],
            language=LANGUAGE,
            image_base64=base64.b64encode(image_jpeg).decode("ascii"),
            syntax_error="\n".join(
                f"line {error.line}: {error.message}" if error.line else error.message
                for error in syntax_errors
            ) or None,
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


class DockerSandbox:
    """Adapter from the project Docker sandbox to the OCR pipeline contract."""

    def __init__(self, runner: Optional[DockerRunner] = None):
        self.runner = runner or DockerRunner(
            mem_mb=_memory_to_mb(settings.SANDBOX_MEMORY),
            cpus=settings.SANDBOX_CPUS,
            image=settings.SANDBOX_IMAGE,
        )
        self.timeout_s = float(settings.SANDBOX_TIMEOUT)

    def supports(self, language: str) -> bool:
        return canonical_language(language) == LANGUAGE

    def check(self, code: str, language: str) -> SandboxResult:
        result = self.runner.run_with_precheck(code, timeout_s=self.timeout_s)

        # A Docker daemon/image failure is different from a student's failed
        # program. Do not send infrastructure errors to the OCR-correction LLM.
        if result.status is RunStatus.SANDBOX_ERROR:
            raise SandboxUnavailable(result.stderr or "Docker sandbox unavailable")

        message = result.stderr.strip() or result.status.value
        errors = [] if result.ok else [SyntaxIssue(line=result.error_line, message=message)]
        output = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
        return SandboxResult(
            language=canonical_language(language),
            image=getattr(self.runner, "image", settings.SANDBOX_IMAGE),
            command="DockerRunner.run_with_precheck",
            exit_code=result.exit_code,
            ok=result.ok,
            timed_out=result.status is RunStatus.TIMEOUT,
            errors=errors,
            output=output,
            duration_ms=result.duration_ms,
        )
