import logging
import os
import time
import uuid
from functools import lru_cache
from typing import Optional

from fastapi import UploadFile
from fastapi_sqlalchemy import db
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import SessionLocal
from app.engines.ocr_paddle import PaddleOcrEngine
from app.models.model_ocr import (
    OcrCodeVersion,
    OcrLlmCallLog,
    OcrSandboxCheck,
    OcrSubmission,
)
from app.schemas.sche_ocr_pipeline import (
    Correction,
    LlmCallRecord,
    OcrBox,
    PipelineResult,
    SandboxResult,
)
from app.services.srv_base import BaseService
from app.services.srv_llm_code import OpenAICodeLlm
from app.services.srv_ocr_pipeline import OcrPipeline
from app.services.srv_sandbox import LANGUAGES, DockerSandbox, SandboxConfig
from app.utils.enums import CodeStage, OcrStatus
from app.utils.exception_handler import CustomException, ExceptionType
from app.utils.image import InvalidImage, decode_image
from app.utils.storage import LocalStorage

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

storage = LocalStorage(settings.UPLOAD_DIR)


@lru_cache(maxsize=1)
def get_pipeline() -> OcrPipeline:
    return OcrPipeline(
        ocr=PaddleOcrEngine(
            lang=settings.OCR_LANG,
            device=settings.OCR_DEVICE,
            det_model=settings.OCR_DET_MODEL,
            rec_model=settings.OCR_REC_MODEL,
        ),
        llm=OpenAICodeLlm(
            api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_MODEL,
            timeout=settings.OPENAI_TIMEOUT,
            supported_languages=tuple(LANGUAGES),
        ),
        sandbox=DockerSandbox(
            SandboxConfig(
                timeout_s=settings.SANDBOX_TIMEOUT,
                memory=settings.SANDBOX_MEMORY,
                cpus=settings.SANDBOX_CPUS,
                runtime=settings.SANDBOX_RUNTIME,
            )
        ),
        max_fix_rounds=settings.OCR_MAX_FIX_ROUNDS,
        low_confidence_threshold=settings.OCR_LOW_CONFIDENCE,
        review_confidence_threshold=settings.OCR_REVIEW_CONFIDENCE,
    )


class DbRecorder:
    """Persists every pipeline step so a wrong result can be traced back to its stage."""

    def __init__(self, session: Session, submission: OcrSubmission):
        self.session = session
        self.submission = submission
        self.run = submission.run
        self._started = time.monotonic()

    def on_status(self, status: OcrStatus) -> None:
        self.submission.status = status.value
        timings = dict(self.submission.timings or {})
        timings.setdefault(status.value, int((time.monotonic() - self._started) * 1000))
        self.submission.timings = timings
        self.session.commit()

    def on_raw_ocr(self, boxes: list[OcrBox]) -> None:
        self.submission.raw_ocr = [box.model_dump() for box in boxes]
        self.session.commit()

    def on_code_version(
        self,
        stage: CodeStage,
        code: str,
        corrections: list[Correction],
        language: Optional[str] = None,
        fix_round: int = 0,
    ) -> Optional[int]:
        version = OcrCodeVersion(
            submission_id=self.submission.id,
            run=self.run,
            stage=stage.value,
            fix_round=fix_round,
            language=language,
            code=code,
            corrections=[c.model_dump() for c in corrections],
        )
        self.session.add(version)
        self.session.commit()
        return version.id

    def on_sandbox_check(self, version_id: Optional[int], result: SandboxResult) -> None:
        self.session.add(
            OcrSandboxCheck(
                submission_id=self.submission.id,
                code_version_id=version_id,
                run=self.run,
                language=result.language,
                image=result.image,
                command=result.command,
                exit_code=result.exit_code,
                ok=result.ok,
                timed_out=result.timed_out,
                errors=[e.model_dump() for e in result.errors],
                output=result.output,
                duration_ms=result.duration_ms,
            )
        )
        self.session.commit()

    def on_llm_call(self, record: LlmCallRecord) -> None:
        self.session.add(OcrLlmCallLog(submission_id=self.submission.id, run=self.run, **record.model_dump()))
        self.session.commit()

    def finish(self, result: PipelineResult) -> None:
        submission = self.submission
        submission.language = result.language
        submission.final_code = result.code
        submission.syntax_ok = result.syntax_ok
        submission.syntax_errors = [e.model_dump() for e in result.syntax_errors]
        submission.uncertain_lines = result.uncertain_lines
        submission.confidence = result.confidence
        submission.need_review = result.need_review
        submission.review_reasons = result.review_reasons
        submission.error = None
        self.on_status(OcrStatus.NEED_REVIEW if result.need_review else OcrStatus.DONE)

    def fail(self, error: str) -> None:
        self.session.rollback()
        self.submission.error = error[:4000]
        self.on_status(OcrStatus.FAILED)


def process_submission(submission_id: uuid.UUID) -> None:
    """Runs in a background task, outside the request, so it opens its own DB session."""
    session = SessionLocal()
    try:
        submission = session.get(OcrSubmission, submission_id)
        if submission is None:
            logger.warning("OCR submission %s not found", submission_id)
            return
        recorder = DbRecorder(session, submission)
        try:
            image = storage.read(submission.image_path)
            result = get_pipeline().run(image, recorder)
            recorder.finish(result)
        except Exception as e:
            logger.exception("OCR submission %s failed", submission_id)
            recorder.fail(f"{type(e).__name__}: {e}")
    finally:
        session.close()


class OcrSubmissionService(BaseService[OcrSubmission]):

    def __init__(self):
        super().__init__(OcrSubmission)

    def create_from_upload(self, file: UploadFile, user_id: Optional[int] = None) -> OcrSubmission:
        extension = os.path.splitext(file.filename or "")[1].lower()
        if extension not in ALLOWED_EXTENSIONS:
            raise CustomException(
                http_code=400,
                message=f"Unsupported file type, allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
            )
        max_bytes = settings.OCR_MAX_UPLOAD_MB * 1024 * 1024
        data = file.file.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise CustomException(http_code=413, message=f"File is larger than {settings.OCR_MAX_UPLOAD_MB}MB")
        try:
            decode_image(data)
        except InvalidImage as e:
            raise CustomException(http_code=400, message=str(e))

        submission = OcrSubmission(
            user_id=user_id,
            image_path=storage.save(data, extension),
            original_filename=file.filename,
            status=OcrStatus.PENDING.value,
            run=1,
        )
        db.session.add(submission)
        db.session.commit()
        db.session.refresh(submission)
        return submission

    def prepare_retry(self, id: uuid.UUID) -> OcrSubmission:
        submission = self.get_by_id(id)
        if submission.status not in (OcrStatus.DONE.value, OcrStatus.NEED_REVIEW.value, OcrStatus.FAILED.value):
            raise CustomException(exception=ExceptionType.CONFLICT)
        submission.run += 1
        submission.status = OcrStatus.PENDING.value
        submission.timings = None
        submission.error = None
        db.session.commit()
        db.session.refresh(submission)
        return submission

    def get_trace(self, id: uuid.UUID, run: Optional[int] = None) -> dict:
        submission = self.get_by_id(id)
        run = run or submission.run
        return {
            "submission": submission,
            "versions": [v for v in submission.versions if v.run == run],
            "sandbox_checks": [c for c in submission.sandbox_checks if c.run == run],
            "llm_calls": [c for c in submission.llm_calls if c.run == run],
        }
