"""Service LLM: (1) sửa lỗi OCR, (2) nhận xét bài làm.

Luồng chung của mỗi chức năng:
    request -> dựng prompt -> gọi LLM -> đọc JSON -> kiểm tra / chuẩn hóa -> response

LLM được gọi qua chuẩn OpenAI Chat Completions, nên đổi model/nhà cung cấp chỉ cần sửa .env.
Khi LLM_PROVIDER=mock, không gọi LLM thật mà trả kết quả giả lập (để dev, test, demo khi chưa có key).
"""

import difflib
import json
import keyword
import logging
import re
from collections import Counter
from typing import Any, Optional

import httpx

from app.core.config import settings
from app.schemas.sche_llm import (
    ISSUE_TYPES,
    CodeChange,
    Conclusion,
    GeneralAssessment,
    SolutionGuide,
    CorrectOcrRequest,
    CorrectOcrResponse,
    ReviewIssue,
    ReviewRequest,
    ReviewResponse,
    SuspiciousChange,
    UncertainSpot,
)
from app.services import llm_prompts as P
from app.utils.exception_handler import CustomException

logger = logging.getLogger(__name__)


# ============================================================================
# Client gọi LLM
# ============================================================================


class LLMClient:
    """Gọi một endpoint chuẩn OpenAI `/chat/completions` và trả về dict JSON."""

    def __init__(self):
        self.provider = settings.LLM_PROVIDER.lower()
        self.base_url = settings.LLM_BASE_URL.rstrip("/")
        self.model = settings.LLM_MODEL if self.provider != "mock" else "mock"
        self.last_raw: str = ""   # output thô lần gọi gần nhất, để ghi log khi JSON thiếu trường

    def chat_json(self, messages: list[dict]) -> dict:
        body: dict[str, Any] = {
            "model": settings.LLM_MODEL,
            "messages": messages,
            "temperature": settings.LLM_TEMPERATURE,
        }
        if settings.LLM_JSON_MODE:
            body["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json"}
        if settings.LLM_API_KEY:
            headers["Authorization"] = f"Bearer {settings.LLM_API_KEY}"

        try:
            with httpx.Client(timeout=settings.LLM_TIMEOUT_SECONDS) as client:
                r = client.post(f"{self.base_url}/chat/completions", json=body, headers=headers)
                # Một số server không hỗ trợ response_format -> thử lại không có nó
                if r.status_code == 400 and "response_format" in body:
                    body.pop("response_format")
                    r = client.post(f"{self.base_url}/chat/completions", json=body, headers=headers)
                r.raise_for_status()
                content = r.json()["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            logger.error("LLM HTTP %s: %s", e.response.status_code, e.response.text[:500])
            raise CustomException(http_code=502, message=f"LLM trả lỗi HTTP {e.response.status_code}")
        except httpx.HTTPError as e:
            logger.error("Không kết nối được LLM: %s", e)
            raise CustomException(http_code=503, message=f"Không kết nối được LLM tại {self.base_url}")
        except (KeyError, IndexError, ValueError) as e:
            raise CustomException(http_code=502, message=f"Response của LLM không đúng định dạng: {e}")

        self.last_raw = content or ""
        data = extract_json(content)
        if data is None:
            logger.error("LLM không trả JSON hợp lệ: %s", content[:500])
            raise CustomException(http_code=502, message="LLM không trả về JSON hợp lệ")
        return data


def extract_json(text: str) -> Optional[dict]:
    """Lấy object JSON từ output của LLM, kể cả khi bị bọc trong ```json ... ``` hoặc có chữ thừa."""
    if not text:
        return None
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


# ============================================================================
# Các hàm kiểm tra dùng chung
# ============================================================================


def strip_code_fence(code: str) -> str:
    m = re.match(r"^\s*```[a-zA-Z]*\n(.*?)\n?```\s*$", code, re.DOTALL)
    return m.group(1) if m else code


def check_syntax(code: str) -> tuple[bool, Optional[str]]:
    # compile() bắt được nhiều lỗi hơn ast.parse (vd. 'return' nằm ngoài hàm). Không chạy code.
    try:
        compile(code, "<student_code>", "exec", dont_inherit=True)
        return True, None
    except SyntaxError as e:
        return False, f"Dòng {e.lineno}: {e.msg}"
    except ValueError as e:   # vd. code chứa ký tự null
        return False, str(e)


def similarity(a: str, b: str) -> float:
    """Độ giống nhau, bỏ qua khoảng trắng và hoa/thường (để không phạt việc sửa thụt lề, khôi phục chữ thường)."""
    na, nb = re.sub(r"\s+", "", a).lower(), re.sub(r"\s+", "", b).lower()
    if not na and not nb:
        return 1.0
    return difflib.SequenceMatcher(None, na, nb, autojunk=False).ratio()


def line_diff(before: str, after: str) -> list[CodeChange]:
    """Tự tính danh sách dòng bị thay đổi (dùng khi LLM không tự liệt kê)."""
    a, b = before.split("\n"), after.split("\n")
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        old, new = a[i1:i2], b[j1:j2]
        for k in range(max(len(old), len(new))):
            out.append(CodeChange(
                line=j1 + min(k, max(len(new) - 1, 0)) + 1,
                before=old[k] if k < len(old) else "",
                after=new[k] if k < len(new) else "",
                reason="(tự tính từ diff)",
                confidence="medium",
            ))
    return out


# ---- Phát hiện thay đổi đáng ngờ: LLM có đổi logic của sinh viên không?
# Token: số (bắt đầu bằng chữ số, gồm cả "1O" để bắt nhầm O/0) | tên | toán tử nhiều ký tự | 1 ký tự bất kỳ
TOKEN_RE = re.compile(r"\d[\w.]*|[^\W\d]\w*|\*\*=|//=|==|!=|<=|>=|\+=|-=|\*=|/=|%=|//|\*\*|\S")
# Đổi qua lại giữa các toán tử này là lựa chọn của sinh viên, không phải lỗi OCR
SWAPPABLE_OPS = {"/", "%", "<", "<=", ">", ">=", "=", "==", "+", "-", "and", "or"}
# Nhầm ký tự trong số mà OCR hay gặp (O->0, l->1, S->5) thì vẫn cho sửa
NUMBER_CONFUSIONS = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1", "S": "5"})
MIN_IDENT_RATIO = 0.5
CONDITION_RE = re.compile(r"^\s*(if|elif|while)\b")
ASSIGN_MINUS_RE = re.compile(r"^\s*[^\W\d]\w*\s*-(?!=)")   # "ask -input(...)": OCR đọc "=" thành "-"
FLAG_NOTE = "server nghi ngờ: có thể là thay đổi logic, không phải lỗi OCR"
FLAG_LINE_NOTE = "server nghi ngờ: dòng bị sửa quá nhiều"
REVERT_NOTE = "server hoàn tác: có thể là thay đổi logic, không phải lỗi OCR"
REVERT_LINE_NOTE = "server hoàn tác: dòng bị sửa quá nhiều"


def _token_kind(tok: str) -> str:
    if tok in SWAPPABLE_OPS:
        return "operator"
    if tok[0].isdigit():
        return "number"
    if tok[0].isalpha() or tok[0] == "_":
        return "identifier"
    return "other"


def suspicious_kind(ocr_tok: str, llm_tok: str, ctx: Optional[dict] = None) -> Optional[str]:
    """Thay 1 token bằng 1 token khác có phải thay đổi logic không? Trả về loại, hoặc None nếu là sửa OCR hợp lý.

    ctx (không bắt buộc): ocr_line, pos (vị trí token trong dòng OCR), ocr_counts / llm_counts (số lần mỗi token
    xuất hiện trong toàn bộ bản OCR / code LLM). Thiếu ctx thì chỉ áp các luật không cần ngữ cảnh.
    """
    if ocr_tok == llm_tok:
        return None
    ctx = ctx or {}
    a, b = _token_kind(ocr_tok), _token_kind(llm_tok)
    if a == b == "operator":
        line = ctx.get("ocr_line", "")
        if (ocr_tok, llm_tok) == ("=", "==") and CONDITION_RE.match(line):
            return None   # "=" trong điều kiện if/elif/while là lỗi cú pháp: OCR mất một dấu "="
        if (ocr_tok, llm_tok) == ("-", "=") and ctx.get("pos") == 1 and ASSIGN_MINUS_RE.match(line):
            return None   # phép gán đầu câu lệnh: OCR đọc "=" thành "-"
        return "operator"
    if a == b == "number":
        if llm_tok in (ocr_tok.translate(NUMBER_CONFUSIONS), ocr_tok.rstrip(".").translate(NUMBER_CONFUSIONS)):
            return None   # nhầm ký tự trong số, hoặc dấu "." thừa cuối số (nhiễu OCR)
        return "number"
    if a == b == "identifier":
        if keyword.iskeyword(llm_tok) and not keyword.iskeyword(ocr_tok):
            return None   # tên lạ -> từ khóa Python (vd. "Son" -> "for"): OCR đọc sai từ khóa
        # so chữ thường để không phạt việc khôi phục chữ thường (DEF -> def)
        ratio = difflib.SequenceMatcher(None, ocr_tok.lower(), llm_tok.lower(), autojunk=False).ratio()
        if ratio >= MIN_IDENT_RATIO:
            return None
        oc, lc = ctx.get("ocr_counts"), ctx.get("llm_counts")
        if oc is not None and lc is not None and oc[ocr_tok] == 1 and lc[llm_tok] >= 2:
            return None   # tên OCR chỉ xuất hiện 1 lần, đổi về tên LLM dùng thống nhất ở chỗ khác (vd. "cum" -> "nam")
        return "identifier"
    return None


def _pair_lines(ocr_lines: list[str], new_lines: list[str]) -> list[tuple[int, int]]:
    """Ghép dòng OCR với dòng đã sửa tương ứng: (chỉ số dòng OCR, chỉ số dòng mới)."""
    sm = difflib.SequenceMatcher(None, [ln.strip() for ln in ocr_lines], [ln.strip() for ln in new_lines],
                                 autojunk=False)
    pairs = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "replace":
            continue
        if i2 - i1 == j2 - j1:
            pairs += list(zip(range(i1, i2), range(j1, j2)))
            continue
        # Số dòng khác nhau (vd. LLM bỏ dòng nhiễu): ghép mỗi dòng mới với dòng OCR giống nhất
        for j in range(j1, j2):
            ratio, best = max((difflib.SequenceMatcher(None, ocr_lines[i].strip(), new_lines[j].strip(),
                                                       autojunk=False).ratio(), i) for i in range(i1, i2))
            if ratio >= 0.5:
                pairs.append((best, j))
    return pairs


def _tokens(text: str) -> list[str]:
    return [m.group() for m in TOKEN_RE.finditer(text)]


def find_suspicious_tokens(ocr_text: str, code: str) -> list[tuple[tuple[int, int], SuspiciousChange]]:
    """So từng dòng OCR với dòng LLM theo token. Chỉ xét thay 1-đổi-1 cùng vị trí; tách/gộp token
    (vd. "21" -> "> 1") là sửa OCR, không xét. Trả về [(span của token LLM trong dòng, thay đổi)]."""
    ocr_lines, new_lines = ocr_text.split("\n"), code.split("\n")
    counts = {"ocr_counts": Counter(_tokens(ocr_text)), "llm_counts": Counter(_tokens(code))}
    found = []
    for i, j in _pair_lines(ocr_lines, new_lines):
        old_toks = _tokens(ocr_lines[i])
        new_matches = list(TOKEN_RE.finditer(new_lines[j]))
        sm = difflib.SequenceMatcher(None, old_toks, [m.group() for m in new_matches], autojunk=False)
        for tag, a1, a2, b1, b2 in sm.get_opcodes():
            if tag != "replace" or a2 - a1 != b2 - b1:
                continue
            for a, b in zip(range(a1, a2), range(b1, b2)):
                new_tok = new_matches[b].group()
                kind = suspicious_kind(old_toks[a], new_tok, {"ocr_line": ocr_lines[i], "pos": a, **counts})
                if kind:
                    found.append((new_matches[b].span(), SuspiciousChange(
                        line=j + 1, kind=kind, ocr_token=old_toks[a], llm_token=new_tok,
                        ocr_line=ocr_lines[i], llm_line=new_lines[j])))
    return found


def _line_ratio(a: str, b: str) -> float:
    na, nb = re.sub(r"\s+", "", a).lower(), re.sub(r"\s+", "", b).lower()
    return difflib.SequenceMatcher(None, na, nb, autojunk=False).ratio()


def find_heavy_lines(ocr_text: str, code: str, min_ratio: float) -> list[SuspiciousChange]:
    """Dòng nào LLM sửa mà giống dòng OCR dưới `min_ratio` (bỏ khoảng trắng, chữ thường) thì không giống sửa
    ký tự OCR nữa. Chỉ xét các dòng ghép được với một dòng OCR (xem _pair_lines)."""
    ocr_lines, new_lines = ocr_text.split("\n"), code.split("\n")
    return [SuspiciousChange(line=j + 1, kind="line", ocr_token=ocr_lines[i].strip(), llm_token=new_lines[j].strip(),
                             ocr_line=ocr_lines[i], llm_line=new_lines[j])
            for i, j in _pair_lines(ocr_lines, new_lines)
            if _line_ratio(ocr_lines[i], new_lines[j]) < min_ratio]


def guard_llm_code(ocr_text: str, code: str, mode: str, min_line_ratio: float) -> tuple[str, list[SuspiciousChange]]:
    """Phát hiện thay đổi đáng ngờ của LLM (theo token, rồi theo dòng).
    mode="flag": giữ nguyên code LLM. mode="revert": đặt lại token / nguyên dòng OCR gốc.
    Trả về (code cuối, danh sách thay đổi đáng ngờ)."""
    revert = mode == "revert"
    llm_lines = code.split("\n")
    tokens = find_suspicious_tokens(ocr_text, code)
    if revert and tokens:
        lines = code.split("\n")
        for (s, e), ch in sorted(tokens, key=lambda t: (t[1].line, t[0][0]), reverse=True):  # phải -> trái
            ln = lines[ch.line - 1]
            lines[ch.line - 1] = ln[:s] + ch.ocr_token + ln[e:]
        code = "\n".join(lines)
    heavy = find_heavy_lines(ocr_text, code, min_line_ratio)
    for ch in heavy:   # luôn ghi dòng thô LLM trả về (ở chế độ revert, `code` lúc này đã hoàn tác theo token)
        ch.llm_line, ch.llm_token = llm_lines[ch.line - 1], llm_lines[ch.line - 1].strip()
    if revert and heavy:
        lines = code.split("\n")
        for ch in heavy:
            lines[ch.line - 1] = ch.ocr_line
        code = "\n".join(lines)
    whole = {ch.line for ch in heavy}   # dòng đã tính cả dòng thì bỏ các mục theo token của dòng đó
    found = [ch for _, ch in tokens if ch.line not in whole] + heavy
    for ch in found:
        ch.reverted = revert
    return code, found


MAX_IMAGE_CHARS = 10_000_000   # ~7.5 MB ảnh gốc sau khi mã hóa base64


def image_part(image_base64: Optional[str], image_url: Optional[str]) -> Optional[dict]:
    """Tạo phần ảnh cho tin nhắn gửi LLM (chuẩn OpenAI: {"type": "image_url", ...})."""
    if image_base64:
        data = image_base64.strip()
        if len(data) > MAX_IMAGE_CHARS:
            raise CustomException(http_code=413, message="Ảnh quá lớn, hãy resize (cạnh dài ~1024-1600px) trước khi gửi")
        if not data.startswith("data:"):
            data = "data:image/jpeg;base64," + data
        return {"type": "image_url", "image_url": {"url": data}}
    if image_url:
        return {"type": "image_url", "image_url": {"url": image_url.strip()}}
    return None


def missing_keys(data: dict, keys: tuple) -> list[str]:
    """Các trường bắt buộc mà LLM bỏ trống (không có, null, chuỗi rỗng, object rỗng)."""
    out = []
    for k in keys:
        v = data.get(k)
        if v is None or (isinstance(v, str) and not v.strip()) or (isinstance(v, dict) and not v):
            out.append(k)
    return out


def _pick(value: Any, allowed: tuple, default: str) -> str:
    v = str(value).lower().strip() if value is not None else ""
    return v if v in allowed else default


INPUT_PROMPT_RE = re.compile(r"""\binput\(\s*[fFrRbB]?["']""")
REAL_CODE_RE = re.compile(r"\b(def\s+\w+\s*\(|print\s*\(|input\s*\(|range\s*\(|import\s+\w+)|^\s*for\s+\w+\s+in\s+", re.M)


def looks_like_real_code(text: str) -> bool:
    """Mã giả có thực ra là code Python chạy được không? (compile được + có cú pháp Python đặc trưng)"""
    if not text or not REAL_CODE_RE.search(text):
        return False
    try:
        compile(text, "<pseudocode>", "exec", dont_inherit=True)
        return True
    except (SyntaxError, ValueError):
        return False


def review_warnings(req: ReviewRequest, verdict: str, pseudocode: str, feedback: str,
                    passed: int, total: int, issues: list) -> list[str]:
    """Các kiểm tra do server tự làm, không phụ thuộc LLM."""
    w = []
    prompt_lines = [i for i, ln in enumerate(req.code.split("\n"), 1) if INPUT_PROMPT_RE.search(ln)]
    if prompt_lines and not any(it.type == "io_format" for it in issues):
        w.append(f"Code có chuỗi gợi ý trong input() ở dòng {prompt_lines} (sẽ in ra output, sai format khi chấm "
                 "tự động) nhưng nhận xét chưa nêu")
    if looks_like_real_code(pseudocode):
        w.append("Phần hướng dẫn giải chứa code Python chạy được thay vì mã giả")
    if total:
        if passed == total and verdict == "incorrect":
            w.append(f"Kết luận 'incorrect' nhưng code đạt {passed}/{total} test")
        if passed == 0 and verdict == "correct":
            w.append(f"Kết luận 'correct' nhưng code không đạt test nào (0/{total})")
    if req.compile_error and verdict == "correct":
        w.append("Kết luận 'correct' nhưng code bị lỗi biên dịch")
    for t in req.test_results:
        if t.hidden:
            for v in (t.input, t.expected):
                if v and len(v.strip()) >= 3 and v.strip() in feedback:
                    w.append(f"Nhận xét cho sinh viên có thể làm lộ dữ liệu test ẩn '{t.name or '?'}'")
                    break
    return w


# ============================================================================
# Service
# ============================================================================


class LLMService:
    def __init__(self, client: Optional[LLMClient] = None):
        self.client = client or LLMClient()

    MAX_RAW_LOG = 2000

    def _chat_required(self, messages: list[dict], required: tuple) -> dict:
        """Gọi LLM; nếu JSON thiếu trường bắt buộc thì gọi lại 1 lần, vẫn thiếu thì trả 502.

        Một số model (vd. qua OpenRouter) thỉnh thoảng trả JSON hợp lệ nhưng thiếu trường,
        nếu không kiểm tra thì response sẽ là 200 với toàn bộ nội dung rỗng.
        """
        for attempt in (1, 2):
            data = self.client.chat_json(messages)
            missing = missing_keys(data, required)
            if not missing:
                return data
            logger.warning("LLM trả JSON thiếu %s (lần %d/2). Output thô: %s",
                           missing, attempt, self.client.last_raw[: self.MAX_RAW_LOG])
        raise CustomException(
            http_code=502,
            message=f"LLM trả về JSON thiếu trường bắt buộc: {', '.join(missing)} (đã thử lại 1 lần)")

    # ------------------------------------------------------------------ 1. Sửa lỗi OCR
    def correct_ocr(self, req: CorrectOcrRequest) -> CorrectOcrResponse:
        ocr_text = req.ocr_text.replace("\r\n", "\n")
        scores = [ln.score for ln in req.lines] if req.lines else []

        image = image_part(req.image_base64, req.image_url)

        if self.client.provider == "mock":
            data = self._mock_correct(ocr_text)
        else:
            user_text = P.build_correct_ocr_user(ocr_text, scores, req.problem, req.syntax_error,
                                                 has_image=bool(image))
            if image:
                # Định dạng tin nhắn nhiều phần (text + ảnh) theo chuẩn OpenAI, OpenRouter cũng dùng
                user_content: Any = [{"type": "text", "text": P.CORRECT_OCR_IMAGE_NOTE + "\n" + user_text}, image]
            else:
                user_content = user_text
            messages = [
                {"role": "system", "content": P.CORRECT_OCR_SYSTEM},
                {"role": "user", "content": P.CORRECT_OCR_EXAMPLE_USER},
                {"role": "assistant", "content": P.CORRECT_OCR_EXAMPLE_ASSISTANT},
                {"role": "user", "content": user_content},
            ]
            data = self._chat_required(messages, ("code",))

        code = strip_code_fence(str(data.get("code") or "")).rstrip()
        if not code.strip():
            raise CustomException(http_code=502, message="LLM trả về code rỗng")

        # ---- Thay đổi đáng ngờ (đổi toán tử / số / tên biến, dòng bị sửa quá nhiều), trước mọi kiểm tra khác.
        # LLM_GUARD_MODE=flag: chỉ đánh dấu; revert: đặt lại bản OCR gốc
        llm_lines = code.split("\n")
        code, suspicious = guard_llm_code(ocr_text.rstrip(), code, settings.LLM_GUARD_MODE,
                                          settings.LLM_MIN_LINE_SIMILARITY)
        reverted = [s for s in suspicious if s.reverted]
        final_lines = code.split("\n")

        changes = []
        for c in data.get("changes") or []:
            try:
                changes.append(CodeChange(
                    line=int(c.get("line") or 0),
                    before=str(c.get("before", "")),
                    after=str(c.get("after", "")),
                    reason=str(c.get("reason", "")),
                    confidence=_pick(c.get("confidence"), ("high", "medium", "low"), "medium"),
                ))
            except (TypeError, ValueError, AttributeError):
                continue
        if reverted:
            # Cập nhật "after" của các thay đổi LLM tự liệt kê theo dòng đã hoàn tác; hoàn tác hết thì bỏ
            reverted_llm = {llm_lines[r.line - 1].strip(): final_lines[r.line - 1] for r in reverted}
            for c in changes:
                if c.after.strip() in reverted_llm:
                    c.after = reverted_llm[c.after.strip()]
            changes = [c for c in changes if c.after.strip() != c.before.strip()]
        if not changes and code != ocr_text.rstrip():
            changes = line_diff(ocr_text.rstrip(), code)

        uncertain = []
        for u in data.get("uncertain") or []:
            try:
                uncertain.append(UncertainSpot(
                    line=int(u.get("line") or 0), text=str(u.get("text", "")), note=str(u.get("note", ""))))
            except (TypeError, ValueError, AttributeError):
                continue
        for s in suspicious:
            if s.reverted:
                note = REVERT_LINE_NOTE if s.kind == "line" else REVERT_NOTE
                uncertain.append(UncertainSpot(
                    line=s.line, text=s.ocr_token, note=f"{note} (LLM đề xuất '{s.ocr_token}' -> '{s.llm_token}')"))
            else:
                note = FLAG_LINE_NOTE if s.kind == "line" else FLAG_NOTE
                uncertain.append(UncertainSpot(
                    line=s.line, text=s.llm_token, note=f"{note} (LLM đã đổi '{s.ocr_token}' -> '{s.llm_token}')"))

        # ---- Kiểm tra an toàn: LLM có đang "viết lại" code thay vì chỉ sửa lỗi OCR không?
        syntax_ok, syntax_error = check_syntax(code)
        sim = similarity(ocr_text, code)
        reasons = []
        token_sus = [s for s in suspicious if s.kind != "line"]
        line_sus = [s for s in suspicious if s.kind == "line"]
        if token_sus:
            detail = ", ".join(f"dòng {s.line}: '{s.ocr_token}' -> '{s.llm_token}'" for s in token_sus)
            verb = "hoàn tác" if token_sus[0].reverted else "nghi ngờ"
            reasons.append(f"Server {verb} {len(token_sus)} thay đổi của LLM ({detail}): "
                           "có thể là thay đổi logic, không phải lỗi OCR")
        if line_sus:
            lines_ = [s.line for s in line_sus]
            what = "trả lại nguyên dòng OCR" if line_sus[0].reverted else "nghi ngờ"
            reasons.append(f"Server {what} ở dòng {lines_}: LLM sửa quá nhiều "
                           f"(giống bản OCR dưới {settings.LLM_MIN_LINE_SIMILARITY:.0%}), không giống sửa lỗi OCR")
        # Tính trên code LLM trả về (trước khi server hoàn tác) để vẫn báo được việc LLM đã cố viết lại bài
        sim_llm = similarity(ocr_text, "\n".join(llm_lines))
        if sim_llm < settings.LLM_MIN_SIMILARITY:
            reasons.append(f"Code LLM trả về chỉ giống bản OCR {sim_llm:.0%} (ngưỡng {settings.LLM_MIN_SIMILARITY:.0%}), "
                           "LLM có thể đã viết lại code")
        n_ocr = len([ln for ln in ocr_text.split("\n") if ln.strip()])
        n_new = len([ln for ln in code.split("\n") if ln.strip()])
        if n_new > n_ocr:
            hint = ("có thể là dòng OCR bỏ sót mà LLM đọc lại từ ảnh, hoặc LLM tự thêm code"
                    if image else "LLM có thể đã thêm code")
            reasons.append(f"Code sau khi sửa nhiều dòng hơn bản OCR ({n_new} so với {n_ocr}), {hint}")
        if any(c.confidence == "low" for c in changes):
            reasons.append("Có thay đổi độ tin cậy thấp, cần đối chiếu với ảnh gốc")
        if uncertain:
            reasons.append(f"Có {len(uncertain)} chỗ LLM không chắc là lỗi OCR hay lỗi của sinh viên")
        if not syntax_ok:
            reasons.append(f"Code vẫn lỗi cú pháp ({syntax_error}), có thể do OCR hoặc do sinh viên")

        return CorrectOcrResponse(
            code=code,
            changes=changes,
            uncertain=uncertain,
            syntax_ok=syntax_ok,
            syntax_error=syntax_error,
            similarity=round(sim, 4),
            needs_review=bool(reasons),
            review_reasons=reasons,
            suspicious_changes=suspicious,
            reverted_changes=reverted,
            used_image=bool(image) and self.client.provider != "mock",
            model=self.client.model,
        )

    # ------------------------------------------------------------------ 2. Nhận xét bài
    def review(self, req: ReviewRequest) -> ReviewResponse:
        total = len(req.test_results)
        passed = sum(t.passed for t in req.test_results)

        if self.client.provider == "mock":
            data = self._mock_review(req, passed, total)
        else:
            messages = [
                {"role": "system", "content": P.REVIEW_SYSTEM},
                {"role": "user", "content": P.build_review_user(
                    req.problem, req.code, req.test_results, req.compile_error, req.rubric, req.max_score)},
            ]
            data = self._chat_required(messages, ("general", "solution_guide", "conclusion"))

        g = data.get("general") or {}
        sg = data.get("solution_guide") or {}
        cc = data.get("conclusion") or {}

        issues = []
        for it in g.get("issues") or data.get("issues") or []:
            try:
                line = it.get("line")
                issues.append(ReviewIssue(
                    line=int(line) if line not in (None, "", "null") else None,
                    code_snippet=str(it["code_snippet"]) if it.get("code_snippet") else None,
                    type=_pick(it.get("type"), ISSUE_TYPES, "logic"),
                    severity=_pick(it.get("severity"), ("high", "medium", "low"), "medium"),
                    explanation=str(it.get("explanation", "")),
                    suggestion=str(it.get("suggestion", "")),
                ))
            except (TypeError, ValueError, AttributeError):
                continue

        general = GeneralAssessment(
            meets_requirements=_pick(g.get("meets_requirements"), ("yes", "partial", "no"), "partial"),
            requirements_comment=str(g.get("requirements_comment", "")),
            standard_comment=str(g.get("standard_comment", "")),
            optimization_comment=str(g.get("optimization_comment", "")),
            complexity=str(g["complexity"]) if g.get("complexity") else None,
            issues=issues,
        )
        guide = SolutionGuide(
            approach=str(sg.get("approach", "")),
            steps=[str(x) for x in (sg.get("steps") or [])],
            pseudocode=str(sg.get("pseudocode", "")),
            expected_complexity=str(sg["expected_complexity"]) if sg.get("expected_complexity") else None,
        )
        conclusion = Conclusion(
            verdict=_pick(cc.get("verdict"), ("correct", "partially_correct", "incorrect"), "partially_correct"),
            summary=str(cc.get("summary", "")),
        )

        try:
            score = float(data.get("suggested_score", 0))
        except (TypeError, ValueError):
            score = 0.0
        score = max(0.0, min(score, req.max_score))   # không cho LLM chấm vượt thang điểm
        feedback = str(data.get("feedback_for_student", ""))

        return ReviewResponse(
            general=general,
            solution_guide=guide,
            conclusion=conclusion,
            suggested_score=round(score, 2),
            max_score=req.max_score,
            score_explanation=str(data.get("score_explanation", "")),
            feedback_for_student=feedback,
            passed_tests=passed,
            total_tests=total,
            warnings=review_warnings(req, conclusion.verdict, guide.pseudocode, feedback, passed, total, issues),
            model=self.client.model,
        )

    # ------------------------------------------------------------------ Giả lập (LLM_PROVIDER=mock)
    KEYWORDS = {"def", "return", "if", "elif", "else", "for", "while", "in", "and", "or", "not",
                "import", "from", "as", "print", "input", "int", "str", "range", "len", "break",
                "continue", "pass", "class", "try", "except", "is", "none", "true", "false"}

    def _mock_correct(self, ocr_text: str) -> dict:
        """Không phải LLM: chỉ áp vài luật đơn giản để có output hợp lý khi dev/test."""
        out_lines, changes = [], []
        for i, line in enumerate(ocr_text.split("\n"), 1):
            new = re.sub(r"(?<=[A-Za-z0-9])-(?=[A-Za-z0-9])", "_", line)   # a-b -> a_b
            def fix_kw(m):
                w = m.group(0)
                if w.isupper() and w.lower() in self.KEYWORDS:
                    return {"true": "True", "false": "False", "none": "None"}.get(w.lower(), w.lower())
                return w
            new = re.sub(r"[A-Za-z_]+", fix_kw, new)
            if new != line:
                changes.append({"line": i, "before": line, "after": new,
                                "reason": "[mock] luật: '-'->'_', từ khóa viết hoa -> thường", "confidence": "medium"})
            out_lines.append(new)
        return {"code": "\n".join(out_lines), "changes": changes, "uncertain": []}

    def _mock_review(self, req: ReviewRequest, passed: int, total: int) -> dict:
        ratio = passed / total if total else 0.0
        issues = []
        if req.compile_error:
            issues.append({"line": None, "type": "syntax", "severity": "high",
                           "explanation": f"[mock] Code không chạy được: {req.compile_error}",
                           "suggestion": "Kiểm tra lại cú pháp ở dòng được báo lỗi."})
        for i, line in enumerate(req.code.split("\n"), 1):
            if INPUT_PROMPT_RE.search(line):
                issues.append({"line": i, "code_snippet": line.strip(), "type": "io_format", "severity": "medium",
                               "explanation": "[mock] Có chuỗi gợi ý trong input(), sẽ bị in ra output",
                               "suggestion": "Bỏ chuỗi trong input()."})
        for t in req.test_results:
            if not t.passed:
                issues.append({"line": None, "type": "runtime" if t.error else "logic", "severity": "medium",
                               "explanation": f"[mock] Test '{t.name or '?'}' thất bại" + (f": {t.error}" if t.error else ""),
                               "suggestion": "Xem lại logic xử lý cho trường hợp này."})
        verdict = "correct" if total and passed == total else ("incorrect" if passed == 0 else "partially_correct")
        return {
            "general": {
                "meets_requirements": {"correct": "yes", "incorrect": "no"}.get(verdict, "partial"),
                "requirements_comment": f"[mock] Đạt {passed}/{total} test case.",
                "standard_comment": "[mock] Chưa đánh giá (chế độ giả lập).",
                "optimization_comment": "[mock] Chưa đánh giá (chế độ giả lập).",
                "complexity": None,
                "issues": issues,
            },
            "solution_guide": {
                "approach": "[mock] Chưa có hướng dẫn (chế độ giả lập).",
                "steps": [],
                "pseudocode": "[mock] ĐỌC dữ liệu -> XỬ LÝ -> IN kết quả",
                "expected_complexity": None,
            },
            "conclusion": {"verdict": verdict, "summary": f"[mock] Bài đạt {passed}/{total} test case."},
            "suggested_score": round(req.max_score * ratio, 2),
            "score_explanation": f"[mock] Điểm = {req.max_score} x tỷ lệ test đạt ({ratio:.0%}).",
            "feedback_for_student": "[mock] Đây là nhận xét giả lập, chưa dùng LLM thật.",
        }
