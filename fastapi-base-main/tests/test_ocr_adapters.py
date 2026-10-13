import base64

import pytest

from app.schemas.sche_ocr_pipeline import CodeLine, LayoutResult
from app.services.srv_ocr_adapters import LlmServiceAdapter, PrecheckSandbox
from app.utils.exception_handler import CustomException


def layout(*lines, low=()):
    return LayoutResult(
        lines=[
            CodeLine(line_no=i, text=text, indent_level=indent, confidence=0.5 if i in low else 0.99)
            for i, (indent, text) in enumerate(lines, start=1)
        ]
    )


class FakeClient:
    provider = "openai"
    model = "fake-model"
    last_raw = "not json"


class FakeService:
    def __init__(self, response=None, error=None):
        self.client = FakeClient()
        self.response, self.error = response, error
        self.requests = []

    def correct_ocr(self, request):
        self.requests.append(request)
        if self.error:
            raise self.error
        return self.response


def test_llm_adapter_maps_mock_correction():
    # conftest sets LLM_PROVIDER=mock: the real LLMService runs without calling any LLM.
    calls = []
    draft = layout((0, "DEF f(x):"), (1, "RETURN x"))
    result = LlmServiceAdapter().transcribe(b"jpeg", draft, log=calls.append)

    assert result.language == "python"
    assert result.code == "def f(x):\n    return x"
    assert [(c.line, c.old, c.new) for c in result.corrections] == [
        (1, "DEF f(x):", "def f(x):"),
        (2, "    RETURN x", "    return x"),
    ]
    assert len(calls) == 1 and calls[0].parse_ok and calls[0].purpose == "TRANSCRIBE"


def test_llm_adapter_sends_rendered_lines_scores_and_image():
    from app.schemas.sche_llm import CorrectOcrResponse, UncertainSpot

    service = FakeService(
        CorrectOcrResponse(
            code="x = 1", uncertain=[UncertainSpot(line=1, text="x", note="?")], syntax_ok=True,
            similarity=1.0, needs_review=True, review_reasons=["Có 1 chỗ LLM không chắc"], model="fake-model",
        )
    )
    draft = layout((0, "if x:"), (1, "y = l"), low=(2,))
    result = LlmServiceAdapter(service).transcribe(b"jpeg", draft, log=lambda r: None)

    request = service.requests[0]
    assert request.ocr_text == "if x:\n    y = l"
    assert [(line.text, line.score) for line in request.lines] == [("if x:", 0.99), ("    y = l", 0.5)]
    assert base64.b64decode(request.image_base64) == b"jpeg"
    assert result.uncertain_lines == [1]
    assert result.needs_review is True
    assert result.review_reasons == ["Có 1 chỗ LLM không chắc"]


def test_llm_adapter_logs_failed_call_and_reraises():
    calls = []
    service = FakeService(error=CustomException(http_code=502, message="LLM không trả về JSON hợp lệ"))
    with pytest.raises(CustomException):
        LlmServiceAdapter(service).transcribe(b"jpeg", layout((0, "x = 1")), log=calls.append)

    assert calls[0].parse_ok is False
    assert calls[0].error == "LLM không trả về JSON hợp lệ"
    assert calls[0].response == "not json"


def test_precheck_sandbox():
    sandbox = PrecheckSandbox()
    assert sandbox.supports("Python3")
    assert not sandbox.supports("cpp")

    assert sandbox.check("def f(x):\n    return x\n", "python").ok
    bad = sandbox.check("def f(x)\n    return x\n", "python")
    assert not bad.ok
    assert bad.errors[0].line == 1
