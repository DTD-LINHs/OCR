import uuid

from sqlalchemy import Boolean, Column, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship

from app.models.model_base import BareBaseModel
from app.utils.enums import OcrStatus


class OcrSubmission(BareBaseModel):
    """One uploaded photo of handwritten code and its final transcription."""

    __tablename__ = "ocr_submission"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # No FK: the users table is created outside of Alembic (create_all) in this base.
    user_id = Column(Integer, index=True, nullable=True)
    image_path = Column(String, nullable=False)
    original_filename = Column(String)
    status = Column(String(32), nullable=False, default=OcrStatus.PENDING.value, index=True)
    # Incremented on every (re)run so logs of different runs can be told apart.
    run = Column(Integer, nullable=False, default=0)
    language = Column(String(32))
    final_code = Column(Text)
    syntax_ok = Column(Boolean)
    syntax_errors = Column(JSONB)
    need_review = Column(Boolean, nullable=False, default=False, index=True)
    review_reasons = Column(JSONB)
    uncertain_lines = Column(JSONB)
    confidence = Column(Float)
    raw_ocr = Column(JSONB)
    timings = Column(JSONB)
    error = Column(Text)

    versions = relationship(
        "OcrCodeVersion", cascade="all, delete-orphan", order_by="OcrCodeVersion.id"
    )
    sandbox_checks = relationship(
        "OcrSandboxCheck", cascade="all, delete-orphan", order_by="OcrSandboxCheck.id"
    )
    llm_calls = relationship(
        "OcrLlmCallLog", cascade="all, delete-orphan", order_by="OcrLlmCallLog.id"
    )


class OcrCodeVersion(BareBaseModel):
    """Snapshot of the code after each pipeline stage, to trace who changed what."""

    __tablename__ = "ocr_code_version"

    submission_id = Column(
        UUID(as_uuid=True),
        ForeignKey("ocr_submission.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    run = Column(Integer, nullable=False, default=0)
    stage = Column(String(32), nullable=False)
    fix_round = Column(Integer, nullable=False, default=0)
    language = Column(String(32))
    code = Column(Text, nullable=False)
    corrections = Column(JSONB)


class OcrSandboxCheck(BareBaseModel):
    """One syntax check of a code version inside the Docker sandbox."""

    __tablename__ = "ocr_sandbox_check"

    submission_id = Column(
        UUID(as_uuid=True),
        ForeignKey("ocr_submission.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    code_version_id = Column(
        Integer, ForeignKey("ocr_code_version.id", ondelete="SET NULL"), nullable=True
    )
    run = Column(Integer, nullable=False, default=0)
    language = Column(String(32), nullable=False)
    image = Column(String)
    command = Column(Text)
    exit_code = Column(Integer)
    ok = Column(Boolean, nullable=False, default=False)
    timed_out = Column(Boolean, nullable=False, default=False)
    errors = Column(JSONB)
    output = Column(Text)
    duration_ms = Column(Integer)


class OcrLlmCallLog(BareBaseModel):
    """Every request sent to the LLM, for cost tracking and prompt debugging."""

    __tablename__ = "ocr_llm_call_log"

    submission_id = Column(
        UUID(as_uuid=True),
        ForeignKey("ocr_submission.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    run = Column(Integer, nullable=False, default=0)
    purpose = Column(String(32), nullable=False)
    model = Column(String(64))
    prompt = Column(Text)
    response = Column(Text)
    prompt_tokens = Column(Integer)
    completion_tokens = Column(Integer)
    latency_ms = Column(Integer)
    parse_ok = Column(Boolean, nullable=False, default=False)
    error = Column(Text)
