"""Orchestrates Paddle OCR -> layout/normalize -> sandbox syntax check -> (LLM fix -> re-check).

The LLM is only called when the OCR text does not compile, and only once: an error
it cannot attribute to OCR may be the student's own and is left for the teacher.

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
    ReviewItem,
    SandboxResult,
    SyntaxIssue,
)
from app.services.srv_layout import build_layout
from app.services.srv_normalize import normalize_layout
from app.utils.enums import CodeStage, OcrStatus

logger = logging.getLogger(__name__)

CallLogger = Callable[[LlmCallRecord], None]

# Below this similarity between the OCR draft and the final code, the LLM has
# probably rewritten rather than transcribed, so a human should look at it.
MIN_DRAFT_SIMILARITY = 0.5


class OcrEngine(Protocol):
    def predict(self, image: bytes) -> list[OcrBox]: ...


class CodeLlm(Protocol):
    def transcribe(
        self, image_jpeg: bytes, draft: LayoutResult, syntax_errors: list[SyntaxIssue], log: CallLogger
    ) -> LlmTranscription: ...


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


def _tag(corrections: list[Correction], by: str) -> list[Correction]:
    return [c.model_copy(update={"by": by}) for c in corrections]


class OcrPipeline:
    def __init__(
        self,
        ocr: OcrEngine,
        llm: CodeLlm,
        sandbox: Sandbox,
        language: str = "python",
        low_confidence_threshold: float = 0.8,
        review_confidence_threshold: float = 0.75,
        llm_image_fn: Callable[[bytes], bytes] = _default_llm_image,
    ):
        self.ocr = ocr
        self.llm = llm
        self.sandbox = sandbox
        self.language = language
        self.low_confidence_threshold = low_confidence_threshold
        self.review_confidence_threshold = review_confidence_threshold
        self.llm_image_fn = llm_image_fn

    def run(self, image: bytes, recorder: PipelineRecorder) -> PipelineResult:
        language = self.language

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
        code = draft.render()
        version_id = recorder.on_code_version(
            CodeStage.SERVICE_NORMALIZED, code, normalize_fixes, language=language
        )
        corrections = _tag(normalize_fixes, "normalize")

        # 3. Syntax check of the OCR text
        check, sandbox_unavailable = self._check(code, language, version_id, recorder)

        # 4. Only if it does not compile: the LLM fixes OCR misreads once, then check again
        transcription: Optional[LlmTranscription] = None
        if check is not None and not check.ok:
            recorder.on_status(OcrStatus.LLM_CORRECTING)
            transcription = self.llm.transcribe(
                self.llm_image_fn(image), draft, check.errors, log=recorder.on_llm_call
            )
            code = transcription.code
            llm_corrections = _tag(transcription.corrections, "llm")
            version_id = recorder.on_code_version(
                CodeStage.LLM_CORRECTED, code, llm_corrections, language=language
            )
            corrections += llm_corrections
            check, sandbox_unavailable = self._check(code, language, version_id, recorder)

        # 5. What the teacher must look at
        review_items: list[ReviewItem] = []
        if transcription is None:
            # No LLM compared the text with the photo: every unsure OCR line is worth a look.
            review_items += [
                ReviewItem(line=line.line_no, reason=f"PaddleOCR không chắc (độ tin cậy {line.confidence:.2f})", source="ocr")
                for line in draft.lines
                if line.low_confidence
            ]
        else:
            review_items += transcription.uncertain
        if check is not None and not check.ok:
            review_items += [
                ReviewItem(line=e.line, reason=f"Lỗi cú pháp còn lại (do OCR hoặc do sinh viên): {e.message}", source="sandbox")
                for e in check.errors
            ]

        confidence = sum(box.confidence for box in boxes) / len(boxes)
        reasons = []
        if check is None:
            reasons.append("sandbox_unavailable" if sandbox_unavailable else "language_not_checked")
        elif not check.ok:
            reasons.append("syntax_errors")
        if any(item.source != "sandbox" for item in review_items):
            reasons.append("uncertain_lines")
        if confidence < self.review_confidence_threshold:
            reasons.append("low_confidence")
        if transcription is not None:
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
            llm_used=transcription is not None,
            corrections=corrections,
            review_items=review_items,
            uncertain_lines=sorted({item.line for item in review_items if item.line and item.source != "sandbox"}),
            confidence=confidence,
            need_review=bool(reasons),
            review_reasons=reasons,
        )

    def _check(
        self, code: str, language: str, version_id: Optional[int], recorder: PipelineRecorder
    ) -> tuple[Optional[SandboxResult], bool]:
        """Return (result, sandbox_unavailable); the result is None when nothing was checked."""
        if not self.sandbox.supports(language):
            return None, False
        try:
            recorder.on_status(OcrStatus.SANDBOX_CHECKING)
            check = self.sandbox.check(code, language)
        except SandboxUnavailable as e:
            logger.error("Sandbox unavailable: %s", e)
            return None, True
        recorder.on_sandbox_check(version_id, check)
        return check, False
