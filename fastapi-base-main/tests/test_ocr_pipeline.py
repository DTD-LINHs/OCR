from typing import Optional

import pytest

from app.schemas.sche_ocr_pipeline import (
    LlmCallRecord,
    LlmSyntaxFix,
    LlmTranscription,
    OcrBox,
    SandboxResult,
    SyntaxDecision,
    SyntaxIssue,
)
from app.services.srv_ocr_pipeline import OcrPipeline, PipelineError
from app.services.srv_sandbox import SandboxUnavailable
from app.utils.enums import CodeStage, OcrStatus


def box(text, row, col=0, conf=0.95):
    x0, y0 = col * 10, row * 30
    x1, y1 = x0 + len(text) * 10, y0 + 20
    return OcrBox(text=text, polygon=[[x0, y0], [x1, y0], [x1, y1], [x0, y1]], confidence=conf)


DRAFT_BOXES = [box("def f(x):", 0), box("return x", 1, col=4)]
GOOD = "def f(x):\n    return x"
BROKEN = "def f(x);\n    return x"


class FakeOcr:
    def __init__(self, boxes):
        self.boxes = boxes

    def predict(self, image):
        return self.boxes


class FakeLlm:
    def __init__(self, transcription, fixes=()):
        self.transcription = transcription
        self.fixes = list(fixes)
        self.fix_calls = 0

    def transcribe(self, image_jpeg, draft, log):
        log(LlmCallRecord(purpose="TRANSCRIBE", model="fake", prompt=draft.render(), parse_ok=True))
        return self.transcription

    def fix_syntax(self, image_jpeg, code, language, errors, log):
        self.fix_calls += 1
        log(LlmCallRecord(purpose="FIX_SYNTAX", model="fake", prompt=code, parse_ok=True))
        return self.fixes.pop(0)


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
            errors=[] if ok else [SyntaxIssue(line=1, message="SyntaxError: invalid syntax")],
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
        self.versions.append((stage, code, fix_round))
        return len(self.versions)

    def on_sandbox_check(self, version_id, result):
        self.checks.append((version_id, result))

    def on_llm_call(self, record):
        self.llm_calls.append(record)


def make_pipeline(llm, sandbox=None, boxes=DRAFT_BOXES, max_fix_rounds=2):
    return OcrPipeline(
        ocr=FakeOcr(boxes),
        llm=llm,
        sandbox=sandbox or FakeSandbox(),
        max_fix_rounds=max_fix_rounds,
        llm_image_fn=lambda image: image,
    )


def transcription(code, language="python", confidence=0.9, uncertain=()):
    return LlmTranscription(language=language, code=code, confidence=confidence, uncertain_lines=list(uncertain))


def test_clean_transcription_is_done_without_review():
    recorder = MemoryRecorder()
    result = make_pipeline(FakeLlm(transcription(GOOD))).run(b"img", recorder)

    assert result.syntax_ok is True
    assert result.need_review is False
    assert result.code == GOOD
    assert [v[0] for v in recorder.versions] == [
        CodeStage.OCR_RAW,
        CodeStage.SERVICE_NORMALIZED,
        CodeStage.LLM_CORRECTED,
    ]
    assert recorder.statuses[0] == OcrStatus.OCR_RUNNING
    assert len(recorder.checks) == 1


def test_ocr_misread_is_fixed_by_llm_and_rechecked():
    llm = FakeLlm(
        transcription(BROKEN),
        fixes=[LlmSyntaxFix(code=GOOD, decisions=[SyntaxDecision(line=1, cause="ocr", fix=":")])],
    )
    recorder = MemoryRecorder()
    result = make_pipeline(llm).run(b"img", recorder)

    assert result.syntax_ok is True
    assert result.code == GOOD
    assert recorder.versions[-1] == (CodeStage.LLM_FIX, GOOD, 1)
    # Second check is linked to the fixed version.
    assert [c[0] for c in recorder.checks] == [3, 4]
    assert [c.purpose for c in recorder.llm_calls] == ["TRANSCRIBE", "FIX_SYNTAX"]


def test_author_mistake_is_kept_and_flagged_for_review():
    llm = FakeLlm(
        transcription(BROKEN),
        fixes=[LlmSyntaxFix(code=BROKEN, decisions=[SyntaxDecision(line=1, cause="author")])],
    )
    result = make_pipeline(llm).run(b"img", MemoryRecorder())

    assert result.code == BROKEN
    assert result.syntax_ok is False
    assert result.need_review is True
    assert "syntax_errors" in result.review_reasons
    assert llm.fix_calls == 1


def test_fix_loop_stops_after_max_rounds():
    attempts = [f"def f(x){c}\n    return x" for c in "!?"]
    llm = FakeLlm(transcription(BROKEN), fixes=[LlmSyntaxFix(code=a) for a in attempts])
    sandbox = FakeSandbox()
    result = make_pipeline(llm, sandbox=sandbox, max_fix_rounds=2).run(b"img", MemoryRecorder())

    assert llm.fix_calls == 2
    assert len(sandbox.checked) == 3
    assert result.syntax_ok is False


def test_unsure_lines_are_reported():
    llm = FakeLlm(
        transcription(BROKEN, uncertain=[2]),
        fixes=[LlmSyntaxFix(code=BROKEN, decisions=[SyntaxDecision(line=1, cause="unsure")])],
    )
    result = make_pipeline(llm).run(b"img", MemoryRecorder())
    assert result.uncertain_lines == [1, 2]
    assert "uncertain_lines" in result.review_reasons


def test_unsupported_language_skips_sandbox():
    sandbox = FakeSandbox()
    result = make_pipeline(FakeLlm(transcription(GOOD, language="Kotlin")), sandbox=sandbox).run(
        b"img", MemoryRecorder()
    )
    assert result.language == "kotlin"
    assert result.syntax_ok is None
    assert sandbox.checked == []
    assert result.review_reasons == ["language_not_checked"]


def test_sandbox_down_does_not_fail_the_job():
    result = make_pipeline(FakeLlm(transcription(GOOD)), sandbox=FakeSandbox(available=False)).run(
        b"img", MemoryRecorder()
    )
    assert result.syntax_ok is None
    assert "sandbox_unavailable" in result.review_reasons


def test_low_confidence_and_rewrite_are_flagged():
    rewritten = "import os\nclass Totally:\n    pass\n" * 3
    llm = FakeLlm(transcription(rewritten, language="unknown", confidence=0.4))
    result = make_pipeline(llm).run(b"img", MemoryRecorder())
    assert {"low_confidence", "large_llm_changes"} <= set(result.review_reasons)


def test_no_text_raises():
    with pytest.raises(PipelineError):
        make_pipeline(FakeLlm(transcription(GOOD)), boxes=[]).run(b"img", MemoryRecorder())
