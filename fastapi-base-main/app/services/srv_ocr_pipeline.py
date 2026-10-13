"""Orchestrates Paddle OCR -> layout/normalize -> LLM correction -> sandbox syntax check.

The pipeline knows nothing about the database: every intermediate artefact is
reported to a `PipelineRecorder`, so it can be persisted (see srv_ocr_submission)
or captured in memory for tests.
"""

import difflib
import logging
from typing import Callable, Optional, Protocol

from app.schemas.sche_ocr_pipeline import (
    Correction,
    LayoutResult,
    LlmCallRecord,
    LlmTranscription,
    OcrBox,
    PipelineResult,
    SandboxResult,
)
from app.services.srv_layout import build_layout
from app.services.srv_normalize import normalize_layout
from app.services.srv_ocr_adapters import canonical_language
from app.utils.enums import CodeStage, OcrStatus

logger = logging.getLogger(__name__)

CallLogger = Callable[[LlmCallRecord], None]

# Below this similarity between the OCR draft and the final code, the LLM has
# probably rewritten rather than transcribed, so a human should look at it.
MIN_DRAFT_SIMILARITY = 0.5


class OcrEngine(Protocol):
    def predict(self, image: bytes) -> list[OcrBox]: ...


class CodeLlm(Protocol):
    def transcribe(self, image_jpeg: bytes, draft: LayoutResult, log: CallLogger) -> LlmTranscription: ...


class Sandbox(Protocol):
    def supports(self, language: str) -> bool: ...

    def check(self, code: str, language: str) -> SandboxResult: ...


class PipelineRecorder(Protocol):
    def on_status(self, status: OcrStatus) -> None: ...

    def on_raw_ocr(self, boxes: list[OcrBox]) -> None: ...

    def on_code_version(
        self,
        stage: CodeStage,
        code: str,
        corrections: list[Correction],
        language: Optional[str] = None,
        fix_round: int = 0,
    ) -> Optional[int]: ...

    def on_sandbox_check(self, version_id: Optional[int], result: SandboxResult) -> None: ...

    def on_llm_call(self, record: LlmCallRecord) -> None: ...


class PipelineError(Exception):
    pass


class SandboxUnavailable(Exception):
    """The sandbox itself failed, not the checked code."""


def _default_llm_image(image: bytes) -> bytes:
    from app.utils.image import to_llm_jpeg

    return to_llm_jpeg(image)


class OcrPipeline:
    def __init__(
        self,
        ocr: OcrEngine,
        llm: CodeLlm,
        sandbox: Sandbox,
        low_confidence_threshold: float = 0.8,
        review_confidence_threshold: float = 0.75,
        llm_image_fn: Callable[[bytes], bytes] = _default_llm_image,
    ):
        self.ocr = ocr
        self.llm = llm
        self.sandbox = sandbox
        self.low_confidence_threshold = low_confidence_threshold
        self.review_confidence_threshold = review_confidence_threshold
        self.llm_image_fn = llm_image_fn

    def run(self, image: bytes, recorder: PipelineRecorder) -> PipelineResult:
        # 1. Paddle OCR
        recorder.on_status(OcrStatus.OCR_RUNNING)
        boxes = self.ocr.predict(image)
        recorder.on_raw_ocr(boxes)
        if not boxes:
            raise PipelineError("No text detected in the image")

        # 2. Rebuild lines/indentation, then clean characters
        raw_layout = build_layout(boxes, low_confidence_threshold=self.low_confidence_threshold)
        recorder.on_code_version(CodeStage.OCR_RAW, raw_layout.render(), [])
        draft, normalize_fixes = normalize_layout(raw_layout)
        recorder.on_code_version(CodeStage.SERVICE_NORMALIZED, draft.render(), normalize_fixes)

        # 3. The LLM reads the photo and corrects OCR misreads only
        recorder.on_status(OcrStatus.LLM_CORRECTING)
        llm_image = self.llm_image_fn(image)
        transcription = self.llm.transcribe(llm_image, draft, log=recorder.on_llm_call)
        language = canonical_language(transcription.language)
        code = transcription.code
        uncertain = set(transcription.uncertain_lines)
        version_id = recorder.on_code_version(
            CodeStage.LLM_CORRECTED, code, transcription.corrections, language=language
        )

        # 4. Syntax check in the sandbox. Remaining errors may be the author's, so they are only flagged.
        check: Optional[SandboxResult] = None
        sandbox_unavailable = False
        if self.sandbox.supports(language):
            try:
                recorder.on_status(OcrStatus.SANDBOX_CHECKING)
                check = self.sandbox.check(code, language)
                recorder.on_sandbox_check(version_id, check)
            except SandboxUnavailable as e:
                logger.error("Sandbox unavailable: %s", e)
                sandbox_unavailable = True

        # 5. Decide whether a human must review. Pipeline reasons are codes; the LLM's are prefixed text.
        confidence = sum(box.confidence for box in boxes) / len(boxes)
        reasons = []
        if check is None:
            reasons.append("sandbox_unavailable" if sandbox_unavailable else "language_not_checked")
        elif not check.ok:
            reasons.append("syntax_errors")
        if uncertain:
            reasons.append("uncertain_lines")
        if confidence < self.review_confidence_threshold:
            reasons.append("low_confidence")
        similarity = difflib.SequenceMatcher(None, draft.render(), code, autojunk=False).ratio()
        if similarity < MIN_DRAFT_SIMILARITY:
            reasons.append("large_llm_changes")
        reasons += [f"llm: {reason}" for reason in transcription.review_reasons]
        if transcription.needs_review and not transcription.review_reasons:
            reasons.append("llm: needs_review")

        return PipelineResult(
            language=language,
            code=code,
            syntax_ok=check.ok if check else None,
            syntax_errors=check.errors if check else [],
            uncertain_lines=sorted(uncertain),
            confidence=confidence,
            need_review=bool(reasons),
            review_reasons=reasons,
        )
