from typing import Optional

import pytest

from app.schemas.sche_ocr_pipeline import (
    Correction,
    LlmCallRecord,
    LlmTranscription,
    OcrBox,
    ReviewItem,
    SandboxResult,
    SyntaxIssue,
)
from app.services.srv_ocr_pipeline import OcrPipeline, PipelineError, SandboxUnavailable
from app.utils.enums import CodeStage, OcrStatus


def box(text, row, col=0, conf=0.95):
    x0, y0 = col * 10, row * 30
    x1, y1 = x0 + len(text) * 10, y0 + 20
    return OcrBox(text=text, polygon=[[x0, y0], [x1, y0], [x1, y1], [x0, y1]], confidence=conf)


GOOD = "def f(x):\n    return x"
BROKEN = "def f(x);\n    return x"
GOOD_BOXES = [box("def f(x):", 0), box("return x", 1, col=4)]
BROKEN_BOXES = [box("def f(x);", 0), box("return x", 1, col=4)]
FIX = Correction(line=1, old="def f(x);", new="def f(x):", reason="';' đọc nhầm từ ':'")


class FakeOcr:
    def __init__(self, boxes):
        self.boxes = boxes

    def predict(self, image):
        return self.boxes


class FakeLlm:
    def __init__(self, transcription=None):
        self.transcription = transcription
        self.calls = []

    def transcribe(self, image_jpeg, draft, syntax_errors, log):
        self.calls.append((draft.render(), syntax_errors))
        log(LlmCallRecord(purpose="TRANSCRIBE", model="fake", prompt=draft.render(), parse_ok=True))
        return self.transcription


class FakeSandbox:
    """Code equal to GOOD passes; anything else fails on line 1."""

    def __init__(self, available=True):
        self.available = available
        self.checked: list[str] = []

    def supports(self, language):
        return language == "python"

    def check(self, code, language):
        if not self.available:
            raise SandboxUnavailable("daemon down")
        self.checked.append(code)
        ok = code == GOOD
        return SandboxResult(
            language=language,
            image="fake",
            command="check",
            exit_code=0 if ok else 1,
            ok=ok,
            errors=[] if ok else [SyntaxIssue(line=1, message="invalid syntax")],
        )


class MemoryRecorder:
    def __init__(self):
        self.statuses = []
        self.versions = []
        self.checks = []
        self.llm_calls = []

    def on_status(self, status):
        self.statuses.append(status)

    def on_raw_ocr(self, boxes):
        self.raw = boxes

    def on_code_version(self, stage, code, corrections, language=None, fix_round=0) -> Optional[int]:
        self.versions.append((stage, code))
        return len(self.versions)

    def on_sandbox_check(self, version_id, result):
        self.checks.append((version_id, result))

    def on_llm_call(self, record):
        self.llm_calls.append(record)


def make_pipeline(llm=None, sandbox=None, boxes=GOOD_BOXES, language="python"):
    return OcrPipeline(
        ocr=FakeOcr(boxes),
        llm=llm or FakeLlm(),
        sandbox=sandbox or FakeSandbox(),
        language=language,
        llm_image_fn=lambda image: image,
    )


def fixed(code=GOOD, corrections=(FIX,), uncertain=(), review_reasons=()):
    return LlmTranscription(
        language="python",
        code=code,
        corrections=list(corrections),
        uncertain=list(uncertain),
        needs_review=bool(review_reasons),
        review_reasons=list(review_reasons),
    )


def test_code_that_compiles_skips_the_llm():
    llm, recorder = FakeLlm(), MemoryRecorder()
    result = make_pipeline(llm).run(b"img", recorder)

    assert llm.calls == []
    assert result.llm_used is False
    assert result.code == GOOD and result.syntax_ok is True
    assert result.need_review is False
    assert [v[0] for v in recorder.versions] == [CodeStage.OCR_RAW, CodeStage.SERVICE_NORMALIZED]
    # The check is linked to the SERVICE_NORMALIZED version.
    assert [c[0] for c in recorder.checks] == [2]
    assert recorder.statuses == [OcrStatus.OCR_RUNNING, OcrStatus.SANDBOX_CHECKING]


def test_syntax_error_is_fixed_by_llm_and_rechecked():
    llm, recorder = FakeLlm(fixed()), MemoryRecorder()
    result = make_pipeline(llm, boxes=BROKEN_BOXES).run(b"img", recorder)

    # The LLM gets the OCR text and the sandbox errors
    assert llm.calls == [(BROKEN, [SyntaxIssue(line=1, message="invalid syntax")])]
    assert result.llm_used is True
    assert result.code == GOOD and result.syntax_ok is True
    assert [c[0] for c in recorder.checks] == [2, 3]
    assert recorder.versions[-1] == (CodeStage.LLM_CORRECTED, GOOD)
    assert recorder.statuses == [
        OcrStatus.OCR_RUNNING, OcrStatus.SANDBOX_CHECKING, OcrStatus.LLM_CORRECTING, OcrStatus.SANDBOX_CHECKING,
    ]
    assert [(c.line, c.old, c.new, c.by) for c in result.corrections] == [(1, "def f(x);", "def f(x):", "llm")]
    assert result.need_review is False


def test_error_left_by_llm_is_reported_once_without_retry():
    llm = FakeLlm(fixed(code=BROKEN, corrections=[]))
    result = make_pipeline(llm, boxes=BROKEN_BOXES).run(b"img", MemoryRecorder())

    assert len(llm.calls) == 1
    assert result.code == BROKEN and result.syntax_ok is False
    assert "syntax_errors" in result.review_reasons
    assert [(i.line, i.source) for i in result.review_items] == [(1, "sandbox")]
    assert "invalid syntax" in result.review_items[0].reason


def test_normalize_fixes_are_listed():
    boxes = [box("def f(x):", 0), box("return  x", 1, col=4)]  # double space is normalized
    result = make_pipeline(boxes=boxes).run(b"img", MemoryRecorder())
    assert [(c.line, c.new, c.by) for c in result.corrections] == [(2, "return x", "normalize")]


def test_low_confidence_ocr_lines_are_listed_when_llm_is_skipped():
    boxes = [box("def f(x):", 0), box("return x", 1, col=4, conf=0.5)]
    result = make_pipeline(boxes=boxes).run(b"img", MemoryRecorder())

    assert result.llm_used is False
    assert [(i.line, i.source) for i in result.review_items] == [(2, "ocr")]
    assert result.uncertain_lines == [2]
    assert "uncertain_lines" in result.review_reasons


def test_llm_uncertain_spots_and_reasons_are_listed():
    unsure = ReviewItem(line=2, reason="LLM không chắc 'x': có thể là '×'", source="llm")
    llm = FakeLlm(fixed(uncertain=[unsure], review_reasons=["Có 1 chỗ LLM không chắc"]))
    result = make_pipeline(llm, boxes=BROKEN_BOXES).run(b"img", MemoryRecorder())

    assert result.review_items == [unsure]
    assert result.uncertain_lines == [2]
    assert {"uncertain_lines", "llm: Có 1 chỗ LLM không chắc"} <= set(result.review_reasons)


def test_llm_rewrite_is_flagged():
    rewritten = "import os\nclass Totally:\n    pass\n" * 3
    llm = FakeLlm(fixed(code=rewritten, corrections=[]))
    result = make_pipeline(llm, boxes=BROKEN_BOXES).run(b"img", MemoryRecorder())
    assert "large_llm_changes" in result.review_reasons


def test_low_mean_ocr_confidence_is_flagged():
    boxes = [box("def f(x):", 0, conf=0.6), box("return x", 1, col=4, conf=0.7)]
    result = make_pipeline(boxes=boxes).run(b"img", MemoryRecorder())
    assert result.confidence == pytest.approx(0.65)
    assert "low_confidence" in result.review_reasons


def test_unsupported_language_skips_sandbox_and_llm():
    llm, sandbox = FakeLlm(), FakeSandbox()
    result = make_pipeline(llm, sandbox=sandbox, language="kotlin").run(b"img", MemoryRecorder())
    assert result.syntax_ok is None
    assert sandbox.checked == [] and llm.calls == []
    assert result.review_reasons == ["language_not_checked"]


def test_sandbox_down_does_not_fail_the_job():
    result = make_pipeline(sandbox=FakeSandbox(available=False)).run(b"img", MemoryRecorder())
    assert result.syntax_ok is None
    assert "sandbox_unavailable" in result.review_reasons


def test_no_text_raises():
    with pytest.raises(PipelineError):
        make_pipeline(boxes=[]).run(b"img", MemoryRecorder())
