"""Test phần LLM, không cần database và không cần LLM thật.

Chạy:  pytest tests/test_llm.py -v
"""

import json

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1 import api_llm
from app.core.config import settings
from app.schemas.sche_llm import CorrectOcrRequest, ReviewRequest, TestCaseResult
from app.services import llm_prompts as P
from app.services import srv_llm
from app.services.srv_llm import LLMClient, LLMService, extract_json
from app.utils.exception_handler import CustomException, custom_error_handler


# ---------------------------------------------------------------- helpers
def make_app() -> TestClient:
    """App tối giản chỉ có router LLM (app chính cần Postgres mới khởi động được)."""
    app = FastAPI()
    app.include_router(api_llm.router, prefix="/api/v1")
    app.add_exception_handler(CustomException, custom_error_handler)
    return TestClient(app)


def fake_llm(monkeypatch, reply: dict | str | list, status_first: int | None = None, seen: list | None = None):
    """Thay httpx.Client bằng server giả trả về `reply` theo chuẩn OpenAI.

    `reply` là list -> mỗi lần gọi lấy 1 phần tử theo thứ tự (phần tử cuối dùng cho các lần sau).
    """
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        body = json.loads(request.content)
        if seen is not None:
            seen.append(body)
        if status_first and calls["n"] == 1:
            return httpx.Response(status_first, json={"error": "response_format not supported"})
        r = reply[min(calls["n"], len(reply)) - 1] if isinstance(reply, list) else reply
        content = r if isinstance(r, str) else json.dumps(r, ensure_ascii=False)
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    real_client = httpx.Client
    monkeypatch.setattr(srv_llm.httpx, "Client",
                        lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    return calls


def service(provider="openai") -> LLMService:
    c = LLMClient()
    c.provider, c.model = provider, "fake-model"
    return LLMService(client=c)


OCR_IMG0 = """num = int Cviput ("Enter a number ")
Ans =1
while num 21 :
    Ans *= num
    mum-=1
retusn Ans"""


# ---------------------------------------------------------------- extract_json
def test_extract_json_handles_fences_and_noise():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Here you go: {"a": 2} hope it helps') == {"a": 2}
    assert extract_json("not json") is None


# ---------------------------------------------------------------- mock provider qua API
def test_api_correct_ocr_mock():
    srv = service("mock")
    api_llm.llm_service = srv
    r = make_app().post("/api/v1/llm/correct-ocr", json={"ocr_text": "DEF MAIN():\n    MOUSE-X = 1"})
    assert r.status_code == 200, r.text
    data = r.json()["data"]
    assert data["code"] == "def MAIN():\n    MOUSE_X = 1"
    assert data["syntax_ok"] is True
    assert len(data["changes"]) == 2


def test_api_review_mock():
    api_llm.llm_service = service("mock")
    body = {"problem": "Tính giai thừa", "code": "print(1)", "max_score": 10,
            "test_results": [{"name": "t1", "passed": True}, {"name": "t2", "passed": False}]}
    r = make_app().post("/api/v1/llm/review", json=body)
    assert r.status_code == 200, r.text
    data = r.json()["data"]
    assert (data["passed_tests"], data["total_tests"]) == (1, 2)
    assert data["suggested_score"] == 5.0
    assert set(data) >= {"general", "solution_guide", "conclusion"}   # đủ 3 phần
    assert data["conclusion"]["verdict"] == "partially_correct"


def test_api_validation_error_on_missing_field():
    api_llm.llm_service = service("mock")
    r = make_app().post("/api/v1/llm/review", json={"code": "x"})   # thiếu problem
    assert r.status_code == 422


# ---------------------------------------------------------------- provider thật (server giả)
def test_correct_ocr_good_fix(monkeypatch):
    fixed = ('num = int(input("Enter a number: "))\nAns = 1\nwhile num > 1:\n'
             '    Ans *= num\n    num -= 1\nreturn Ans')
    fake_llm(monkeypatch, {"code": fixed, "changes": [
        {"line": 6, "before": "retusn Ans", "after": "return Ans", "reason": "OCR", "confidence": "high"}],
        "uncertain": []})
    res = service().correct_ocr(CorrectOcrRequest(ocr_text=OCR_IMG0))
    assert res.code == fixed
    assert res.similarity > 0.8
    # 'return' ngoài hàm là lỗi cú pháp của chính sinh viên -> báo lại, không tự sửa
    assert res.syntax_ok is False and res.needs_review is True


def test_correct_ocr_flags_rewrite(monkeypatch):
    """LLM viết lại cả bài -> phải bị đánh dấu needs_review."""
    monkeypatch.setattr(settings, "LLM_GUARD_MODE", "revert")
    rewritten = "import math\n\ndef factorial(n):\n    return math.factorial(n)\n\nprint(factorial(int(input())))"
    fake_llm(monkeypatch, {"code": rewritten, "changes": [], "uncertain": []})
    res = service().correct_ocr(CorrectOcrRequest(ocr_text=OCR_IMG0))
    assert res.needs_review is True
    assert any("viết lại" in r for r in res.review_reasons)
    # lớp chặn theo dòng trả lại nguyên các dòng OCR -> code cuối không còn bài viết lại
    assert res.code == OCR_IMG0
    assert res.reverted_changes and all(r.kind == "line" for r in res.reverted_changes)


def test_correct_ocr_auto_diff_when_llm_lists_no_changes(monkeypatch):
    fake_llm(monkeypatch, {"code": "x = 1\nreturn x", "changes": [], "uncertain": []})
    res = service().correct_ocr(CorrectOcrRequest(ocr_text="x = 1\nretusn x"))
    assert [(c.line, c.before, c.after) for c in res.changes] == [(2, "retusn x", "return x")]


def test_correct_ocr_retries_without_json_mode(monkeypatch):
    seen = []
    calls = fake_llm(monkeypatch, {"code": "x = 1", "changes": []}, status_first=400, seen=seen)
    monkeypatch.setattr(settings, "LLM_JSON_MODE", True)
    service().correct_ocr(CorrectOcrRequest(ocr_text="x = l"))
    assert calls["n"] == 2
    assert "response_format" in seen[0] and "response_format" not in seen[1]


def test_correct_ocr_sends_scores_and_example(monkeypatch):
    seen = []
    fake_llm(monkeypatch, {"code": "x = 1"}, seen=seen)
    service().correct_ocr(CorrectOcrRequest(ocr_text="x = l", lines=[{"text": "x = l", "score": 0.42}]))
    msgs = seen[0]["messages"]
    assert msgs[0]["role"] == "system" and "NOT" in msgs[0]["content"]
    assert len(msgs) == 4                      # system + ví dụ (user, assistant) + input thật
    assert "1 | 0.42 | x = l" in msgs[-1]["content"]


def test_llm_bad_json_gives_502(monkeypatch):
    fake_llm(monkeypatch, "Sorry, I cannot help with that.")
    with pytest.raises(CustomException) as e:
        service().correct_ocr(CorrectOcrRequest(ocr_text="x = 1"))
    assert e.value.http_code == 502


def test_llm_unreachable_gives_503(monkeypatch):
    def boom(request):
        raise httpx.ConnectError("refused")
    real_client = httpx.Client
    monkeypatch.setattr(srv_llm.httpx, "Client",
                        lambda **kw: real_client(transport=httpx.MockTransport(boom), **kw))
    with pytest.raises(CustomException) as e:
        service().correct_ocr(CorrectOcrRequest(ocr_text="x = 1"))
    assert e.value.http_code == 503


def test_review_clamps_score_and_coerces_fields(monkeypatch):
    fake_llm(monkeypatch, {"general": {"requirements_comment": "r"}, "solution_guide": {"approach": "a"},
                           "conclusion": {"summary": "ok"}, "suggested_score": 15,
                           "issues": [{"line": "3", "type": "Logic Error", "severity": "HIGH",
                                       "explanation": "e", "suggestion": "s"}]})
    res = service().review(ReviewRequest(problem="p", code="c", max_score=10,
                                         test_results=[TestCaseResult(passed=True)]))
    assert res.suggested_score == 10            # không vượt thang điểm
    assert res.general.issues[0].line == 3
    assert res.general.issues[0].type == "logic"        # giá trị lạ -> mặc định
    assert res.general.issues[0].severity == "high"


def test_review_prompt_hides_hidden_tests():
    prompt = P.build_review_user(
        "p", "c",
        [TestCaseResult(name="public", input="5", expected="120", actual="120", passed=True),
         TestCaseResult(name="secret", input="SECRET_IN", expected="SECRET_OUT", actual="1", passed=False, hidden=True)],
        None, None, 10)
    assert "public" in prompt and "120" in prompt
    assert "SECRET_IN" not in prompt and "SECRET_OUT" not in prompt
    assert "(1/2 passed)" in prompt


# ---------------------------------------------------------------- nhận xét 3 phần + kiểm tra của server
FACT_CODE = 'n = int(input("Nhap n: "))\nresult = 0\nfor i in range(1, n + 1):\n    result *= i\nprint(result)'

GOOD_REVIEW = {
    "general": {
        "meets_requirements": "no",
        "requirements_comment": "Kết quả luôn bằng 0.",
        "standard_comment": "Có chuỗi gợi ý trong input().",
        "optimization_comment": "O(n) là phù hợp.",
        "complexity": "O(n)",
        "issues": [
            {"line": 1, "code_snippet": 'input("Nhap n: ")', "type": "io_format", "severity": "medium",
             "explanation": "Chuỗi gợi ý bị in ra output", "suggestion": "Bỏ chuỗi trong input()"},
            {"line": 2, "code_snippet": "result = 0", "type": "logic", "severity": "high",
             "explanation": "Nhân với 0 luôn ra 0", "suggestion": "Khởi tạo bằng 1"},
        ],
    },
    "solution_guide": {
        "approach": "Nhân dồn từ 1 đến n",
        "steps": ["Đọc n", "Khởi tạo kết quả bằng 1", "Nhân dồn", "In kết quả"],
        "pseudocode": "ĐỌC n\nkq <- 1\nLẶP i TỪ 1 ĐẾN n: kq <- kq * i\nIN kq",
        "expected_complexity": "O(n)",
    },
    "conclusion": {"verdict": "incorrect", "summary": "Sai vì khởi tạo result = 0."},
    "suggested_score": 3, "score_explanation": "...", "feedback_for_student": "Xem lại giá trị khởi tạo.",
}


def review_req(**kw):
    base = dict(problem="Tính n!", code=FACT_CODE, max_score=10,
                test_results=[TestCaseResult(name="n=5", input="5", expected="120", actual="0", passed=False)])
    base.update(kw)
    return ReviewRequest(**base)


def test_review_three_parts_parsed(monkeypatch):
    fake_llm(monkeypatch, GOOD_REVIEW)
    res = service().review(review_req())
    assert res.general.meets_requirements == "no" and res.general.complexity == "O(n)"
    assert [i.type for i in res.general.issues] == ["io_format", "logic"]
    assert res.general.issues[1].code_snippet == "result = 0"
    assert len(res.solution_guide.steps) == 4
    assert res.conclusion.verdict == "incorrect"
    assert res.warnings == []                    # review tốt -> không có cảnh báo


def test_review_warns_when_guide_contains_real_code(monkeypatch):
    bad = json.loads(json.dumps(GOOD_REVIEW))
    bad["solution_guide"]["pseudocode"] = "n = int(input())\nr = 1\nfor i in range(1, n + 1):\n    r *= i\nprint(r)"
    fake_llm(monkeypatch, bad)
    res = service().review(review_req())
    assert any("mã giả" in w for w in res.warnings)


def test_review_warns_on_missed_input_prompt(monkeypatch):
    bad = json.loads(json.dumps(GOOD_REVIEW))
    bad["general"]["issues"] = bad["general"]["issues"][1:]   # LLM bỏ sót lỗi prompt trong input()
    fake_llm(monkeypatch, bad)
    res = service().review(review_req())
    assert any("input()" in w and "[1]" in w for w in res.warnings)


def test_review_warns_on_verdict_mismatch_and_hidden_leak(monkeypatch):
    bad = json.loads(json.dumps(GOOD_REVIEW))
    bad["conclusion"]["verdict"] = "correct"
    bad["feedback_for_student"] = "Em sai test có đầu vào 12345."
    fake_llm(monkeypatch, bad)
    res = service().review(review_req(test_results=[
        TestCaseResult(name="hidden1", input="12345", expected="x", actual="0", passed=False, hidden=True)]))
    assert any("correct" in w and "0/1" in w for w in res.warnings)
    assert any("test ẩn" in w for w in res.warnings)


def test_pseudocode_detector():
    assert not srv_llm.looks_like_real_code("LẶP i TỪ 1 ĐẾN n: kq <- kq * i")
    assert not srv_llm.looks_like_real_code("FOR i FROM 1 TO n:\n    result <- result * i")
    assert srv_llm.looks_like_real_code("for i in range(5):\n    print(i)")


# ---------------------------------------------------------------- gửi kèm ảnh + OpenRouter
def test_correct_ocr_sends_image_as_multipart(monkeypatch):
    seen = []
    fake_llm(monkeypatch, {"code": "x = 1"}, seen=seen)
    res = service().correct_ocr(CorrectOcrRequest(ocr_text="x = l", image_base64="QUJD"))
    content = seen[0]["messages"][-1]["content"]
    assert isinstance(content, list) and content[0]["type"] == "text"
    assert "PHOTO" in content[0]["text"] and "x = l" in content[0]["text"]
    assert content[1] == {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,QUJD"}}
    assert res.used_image is True


def test_correct_ocr_without_image_sends_plain_text(monkeypatch):
    seen = []
    fake_llm(monkeypatch, {"code": "x = 1"}, seen=seen)
    res = service().correct_ocr(CorrectOcrRequest(ocr_text="x = l"))
    assert isinstance(seen[0]["messages"][-1]["content"], str)
    assert res.used_image is False


def test_image_too_large_rejected():
    with pytest.raises(CustomException) as e:
        srv_llm.image_part("A" * (srv_llm.MAX_IMAGE_CHARS + 1), None)
    assert e.value.http_code == 413


def test_openrouter_key_env_fallback(monkeypatch):
    import importlib
    from app.core import config
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    assert config.Settings().LLM_API_KEY == "sk-or-test"
    monkeypatch.setenv("LLM_API_KEY", "sk-direct")
    assert config.Settings().LLM_API_KEY == "sk-direct"


# ---------------------------------------------------------------- test ẩn: không gửi tên test
def test_review_prompt_hides_hidden_test_name():
    prompt = P.build_review_user(
        "Tính n!", "c",
        [TestCaseResult(name="n=5", input="5", expected="120", actual="0", passed=False),
         TestCaseResult(name="n=0", input="0", expected="1", actual="0", passed=False, hidden=True)],
        None, None, 10)
    assert "n=5" in prompt                       # test công khai vẫn giữ tên
    assert "n=0" not in prompt                   # tên test ẩn đã lộ input -> không được gửi
    assert "- hidden test 2: FAIL" in prompt
    assert "hidden test N" in P.REVIEW_SYSTEM and "feedback_for_student" in P.REVIEW_SYSTEM


# ---------------------------------------------------------------- JSON thiếu trường -> gọi lại 1 lần -> 502
def test_review_retries_once_when_parts_missing(monkeypatch):
    calls = fake_llm(monkeypatch, [{"suggested_score": 5}, GOOD_REVIEW])
    res = service().review(review_req())
    assert calls["n"] == 2
    assert res.conclusion.verdict == "incorrect" and res.general.issues


def test_review_missing_parts_twice_gives_502_and_logs_raw(monkeypatch, caplog):
    calls = fake_llm(monkeypatch, {"general": {}, "suggested_score": 5, "note": "RAW_MARKER"})
    with caplog.at_level("WARNING", logger=srv_llm.logger.name), pytest.raises(CustomException) as e:
        service().review(review_req())
    assert calls["n"] == 2                       # chỉ thử lại đúng 1 lần
    assert e.value.http_code == 502
    assert all(k in e.value.message for k in ("general", "solution_guide", "conclusion"))
    assert "RAW_MARKER" in caplog.text           # có ghi output thô của LLM


def test_raw_log_is_truncated(monkeypatch, caplog):
    fake_llm(monkeypatch, {"code": "", "pad": "x" * 5000})
    with caplog.at_level("WARNING", logger=srv_llm.logger.name), pytest.raises(CustomException):
        service().correct_ocr(CorrectOcrRequest(ocr_text="x = 1"))
    assert len(caplog.records) == 2              # 1 dòng log cho mỗi lần gọi
    assert all(len(r.getMessage()) < LLMService.MAX_RAW_LOG + 200 for r in caplog.records)


def test_correct_ocr_retries_when_code_missing(monkeypatch):
    calls = fake_llm(monkeypatch, [{"changes": []}, {"code": "x = 1", "changes": []}])
    res = service().correct_ocr(CorrectOcrRequest(ocr_text="x = l"))
    assert calls["n"] == 2 and res.code == "x = 1"


def test_api_correct_ocr_missing_code_gives_502(monkeypatch):
    calls = fake_llm(monkeypatch, {"code": "   ", "changes": []})
    api_llm.llm_service = service()
    r = make_app().post("/api/v1/llm/correct-ocr", json={"ocr_text": "x = l"})
    assert calls["n"] == 2
    assert r.status_code == 502
    assert "code" in r.json()["message"] and "thử lại" in r.json()["message"]


# ---------------------------------------------------------------- correct-ocr: gửi kèm syntax_error
def test_correct_ocr_prompt_includes_syntax_error():
    prompt = P.build_correct_ocr_user("x = l", [], None, "Dòng 6: 'return' outside function", has_image=True)
    assert "A syntax check of the OCR text reported: Dòng 6: 'return' outside function." in prompt
    assert "OCR misreading" in prompt and "student's own mistake" in prompt
    assert "the photo shows" in prompt
    assert "keep it unchanged and list it in 'uncertain'" in prompt


def test_correct_ocr_syntax_note_without_image_does_not_mention_photo():
    prompt = P.build_correct_ocr_user("x = l", [], None, "Dòng 3: invalid syntax")
    assert prompt.startswith(P.CORRECT_OCR_SYNTAX_NOTE_NO_IMAGE.format(syntax_error="Dòng 3: invalid syntax"))
    assert "clearly a character-level OCR misreading" in prompt
    assert "photo" not in prompt


def test_correct_ocr_prompt_unchanged_without_syntax_error():
    old = "OCR output (line number | OCR score | text):\n1 | 0.42 | x = l"
    assert P.build_correct_ocr_user("x = l", [0.42], None) == old
    assert P.build_correct_ocr_user("x = l", [0.42], None, None) == old
    assert P.build_correct_ocr_user("x = l", [0.42], None, "  ") == old


def test_service_passes_syntax_error_to_prompt(monkeypatch):
    seen = []
    fake_llm(monkeypatch, {"code": "x = 1"}, seen=seen)
    service().correct_ocr(CorrectOcrRequest(ocr_text="x = l", syntax_error="Dòng 3: invalid syntax"))
    service().correct_ocr(CorrectOcrRequest(ocr_text="x = l", syntax_error="Dòng 3: invalid syntax",
                                            image_base64="QUJD"))
    no_img, with_img = seen[0]["messages"][-1]["content"], seen[1]["messages"][-1]["content"][0]["text"]
    assert "reported: Dòng 3: invalid syntax." in no_img and "character-level" in no_img
    assert "reported: Dòng 3: invalid syntax." in with_img and "the photo shows" in with_img


# ---------------------------------------------------------------- server hoàn tác thay đổi đáng ngờ của LLM
def _correct(monkeypatch, ocr: str, llm_code: str, changes: list | None = None, mode: str = "revert"):
    monkeypatch.setattr(settings, "LLM_GUARD_MODE", mode)
    fake_llm(monkeypatch, {"code": llm_code, "changes": changes or [], "uncertain": []})
    return service().correct_ocr(CorrectOcrRequest(ocr_text=ocr))


def _assert_reverted(res, kind: str, ocr_tok: str, llm_tok: str, line: int):
    assert [(r.kind, r.ocr_token, r.llm_token, r.line) for r in res.reverted_changes] == [(kind, ocr_tok, llm_tok, line)]
    assert any(u.line == line and u.text == ocr_tok and "server hoàn tác" in u.note for u in res.uncertain)
    assert any("hoàn tác" in r for r in res.review_reasons) and res.needs_review


def test_revert_operator_swap(monkeypatch):
    res = _correct(monkeypatch, "def is_even(n):\n    if n / 2 == O:\n        retunn True",
                   "def is_even(n):\n    if n % 2 == 0:\n        return True",
                   [{"line": 2, "before": "    if n / 2 == O:", "after": "    if n % 2 == 0:", "reason": "x"}])
    _assert_reverted(res, "operator", "/", "%", 2)
    assert res.code == "def is_even(n):\n    if n / 2 == 0:\n        return True"   # O->0 và retunn vẫn được sửa
    assert [c.after for c in res.changes if c.line == 2] == ["    if n / 2 == 0:"]   # "after" theo dòng đã hoàn tác


def test_revert_number_change(monkeypatch):
    res = _correct(monkeypatch, "total = 0\nfor x in nums:\n    total *= x",
                   "total = 1\nfor x in nums:\n    total *= x",
                   [{"line": 1, "before": "total = 0", "after": "total = 1", "reason": "x"}])
    _assert_reverted(res, "number", "0", "1", 1)
    assert res.code.startswith("total = 0\n")
    assert res.changes == []          # thay đổi duy nhất bị hoàn tác -> không còn trong changes


def test_revert_identifier_rename(monkeypatch):
    res = _correct(monkeypatch, "x = a + b\nprint(x)", "result = a + b\nprint(x)")
    _assert_reverted(res, "identifier", "x", "result", 1)
    assert res.code == "x = a + b\nprint(x)"


def test_genuine_ocr_fixes_are_kept(monkeypatch):
    fixed = ('num = int(input("Enter a number: "))\nAns = 1\nwhile num > 1:\n'
             '    Ans *= num\n    num -= 1\nreturn Ans')
    res = _correct(monkeypatch, OCR_IMG0, fixed)
    assert res.reverted_changes == []
    assert res.code == fixed          # retusn->return, Cviput->input, mum->num, "21"->"> 1" đều giữ
    assert not any("hoàn tác" in r for r in res.review_reasons)


def test_suspicious_kind_rules():
    sk = srv_llm.suspicious_kind
    assert sk("retusn", "return") is None and sk("Cviput", "input") is None and sk("mum", "num") is None
    assert sk("DEF", "def") is None                      # chỉ đổi hoa/thường
    assert sk("1O", "10") is None and sk("l5", "15") is None and sk("S", "5") is None
    assert sk("=", "==") == "operator" and sk("and", "or") == "operator"
    assert sk("2", "3") == "number" and sk("count", "total") == "identifier"


# ---------------------------------------------------------------- chặn theo dòng: dòng bị sửa quá nhiều
def test_line_rewrite_is_reverted(monkeypatch):
    ocr = "def f(nums):\n    for x in nums:\n        print(x)"
    res = _correct(monkeypatch, ocr, "def f(nums):\n    for idx, value in enumerate(sorted(nums)):\n        print(x)")
    assert res.code == ocr                                       # trả lại nguyên dòng OCR
    assert [(r.kind, r.line) for r in res.reverted_changes] == [("line", 2)]
    assert any(u.line == 2 and "dòng bị sửa quá nhiều" in u.note for u in res.uncertain)
    assert any("nguyên dòng OCR" in r for r in res.review_reasons)


def test_line_layer_keeps_genuine_ocr_fixes(monkeypatch):
    ocr = 'num = int Cviput ("Enter a number ")\nAns = 1\nretusn Ans'
    fixed = 'num = int(input("Enter a number "))\nAns = 1\nreturn Ans'
    res = _correct(monkeypatch, ocr, fixed)
    assert res.code == fixed and res.reverted_changes == []


def test_line_threshold_from_config(monkeypatch):
    monkeypatch.setattr(settings, "LLM_MIN_LINE_SIMILARITY", 0.95)   # ngưỡng rất cao -> retusn->return cũng bị trả lại
    res = _correct(monkeypatch, "retusn Ans", "return Ans")
    assert res.code == "retusn Ans" and res.reverted_changes[0].kind == "line"


def test_line_layer_does_not_catch_small_logic_edit(monkeypatch):
    """Giới hạn đã biết: thêm "+ 1" chỉ đổi 2 ký tự nên dòng vẫn giống OCR ~95%, lớp chặn theo dòng không bắt được
    (và lớp token cũng không, vì số token thay đổi). Nếu sau này thêm luật riêng cho off-by-one thì sửa test này."""
    ocr = "for i in range(len(nums)):\n    for j in range(i, len(nums)):\n        pass"
    llm = "for i in range(len(nums)):\n    for j in range(i + 1, len(nums)):\n        pass"
    assert srv_llm._line_ratio("    for j in range(i, len(nums)):", "    for j in range(i + 1, len(nums)):") > 0.9
    res = _correct(monkeypatch, ocr, llm)
    assert res.reverted_changes == [] and "range(i + 1" in res.code


# ---------------------------------------------------------------- LLM_GUARD_MODE=flag (mặc định)
def test_guard_mode_default_is_flag(monkeypatch):
    from app.core import config
    monkeypatch.delenv("LLM_GUARD_MODE", raising=False)
    assert config.Settings().LLM_GUARD_MODE == "flag"
    assert config.Settings().LLM_MIN_LINE_SIMILARITY == 0.5


def test_flag_mode_keeps_llm_code_and_marks_suspicious(monkeypatch):
    ocr = "def is_even(n):\n    if n / 2 == O:\n        retunn True"
    llm = "def is_even(n):\n    if n % 2 == 0:\n        return True"
    res = _correct(monkeypatch, ocr, llm, mode="flag")
    assert res.code == llm                                   # không đổi code của LLM
    assert res.reverted_changes == []
    assert [(s.kind, s.ocr_token, s.llm_token, s.line, s.reverted) for s in res.suspicious_changes] == \
        [("operator", "/", "%", 2, False)]
    assert any(u.line == 2 and u.text == "%" and "server nghi ngờ" in u.note for u in res.uncertain)
    assert any("nghi ngờ" in r for r in res.review_reasons) and res.needs_review


def test_flag_mode_marks_heavy_line_without_changing_it(monkeypatch):
    ocr = "def f(nums):\n    for x in nums:\n        print(x)"
    llm = "def f(nums):\n    for idx, value in enumerate(sorted(nums)):\n        print(x)"
    res = _correct(monkeypatch, ocr, llm, mode="flag")
    assert res.code == llm and res.reverted_changes == []
    assert [(s.kind, s.line) for s in res.suspicious_changes] == [("line", 2)]
    assert any("dòng bị sửa quá nhiều" in u.note for u in res.uncertain)


def test_revert_mode_marks_entries_as_reverted(monkeypatch):
    res = _correct(monkeypatch, "total = 0", "total = 1", mode="revert")
    assert res.code == "total = 0"
    assert res.suspicious_changes == res.reverted_changes and res.reverted_changes[0].reverted is True


# ---------------------------------------------------------------- ngoại lệ: các sửa OCR thật ở bước D không bị tính
def _sus(ocr: str, llm: str):
    _, found = srv_llm.guard_llm_code(ocr, llm, "flag", 0.5)
    return [(s.kind, s.ocr_token, s.llm_token) for s in found]


def test_keyword_restored_is_not_suspicious():                      # ảnh 50: "Son" -> "for"
    assert _sus("Son i in pange (len(nams)) :", "for i in range(len(nams)) :") == []


def test_consistent_rename_is_not_suspicious():
    assert _sus("for nam in dup:\n    out.append(cum)", "for nam in dup:\n    out.append(nam)") == []      # ảnh 51
    assert _sus("for xe in ns:\n    total*=se", "for x in ns:\n    total *= x") == []                    # ảnh 54
    # đổi tên một biến ở mọi chỗ (tên OCR xuất hiện nhiều lần) vẫn bị tính
    assert _sus("count = 0\ncount += 1", "total = 0\ntotal += 1") == [("identifier", "count", "total")] * 2


def test_trailing_dot_in_number_is_noise():                          # ảnh 45: "0." -> "0"
    assert _sus("    max= 0.", "    max = 0") == []
    assert _sus("    max= 0.", "    max = 1") == [("number", "0.", "1")]


def test_assignment_and_condition_operator_fixes():
    assert _sus("if(nam[P]= fanget):", "if(nam[P] == fanget):") == []                       # ảnh 50
    assert _sus("        ask -input (\"Yes/No: \")", "        ask = input(\"Yes/No: \")") == []   # ảnh 29
    assert _sus("x = 1", "x == 1") == [("operator", "=", "==")]          # ngoài điều kiện: vẫn tính
    assert _sus("y = a - b", "y = a = b") == [("operator", "-", "=")]    # không phải phép gán đầu câu: vẫn tính


def test_line_threshold_half_keeps_garbled_def_line():               # ảnh 53, ratio 0.591
    assert _sus("do upon-lowercont:", "def upper_lower_count(stro):") == []
