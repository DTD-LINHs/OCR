from typing import List, Literal, Optional
from pydantic import BaseModel, Field


# ============================================================================
# 1. SỬA LỖI OCR  (POST /api/v1/llm/correct-ocr)
#    OCR service -> LLM -> code đã sửa lỗi nhận dạng
# ============================================================================


class OcrLine(BaseModel):
    """Một dòng do OCR trả về, kèm độ tin cậy (nếu OCR service có)."""

    text: str
    score: Optional[float] = Field(None, ge=0, le=1, description="Độ tin cậy 0..1 của OCR")


class CorrectOcrRequest(BaseModel):
    ocr_text: str = Field(..., description="Code thô do OCR trả về (có thể còn lỗi nhận dạng)")
    lines: Optional[List[OcrLine]] = Field(
        None, description="Không bắt buộc. Từng dòng + score, giúp LLM biết chỗ OCR không chắc"
    )
    language: Literal["python"] = "python"
    problem: Optional[str] = Field(
        None, description="Không bắt buộc. Đề bài, giúp LLM đoán đúng tên biến/hàm"
    )
    image_base64: Optional[str] = Field(
        None, description=("Không bắt buộc. Ảnh bài làm gốc (base64 hoặc data URL 'data:image/jpeg;base64,...'). "
                           "Gửi kèm để LLM đối chiếu với chữ viết thật. Chỉ dùng với model nhận ảnh.")
    )
    image_url: Optional[str] = Field(None, description="Không bắt buộc. URL ảnh bài làm (thay cho image_base64)")


class CodeChange(BaseModel):
    line: int = Field(..., description="Số thứ tự dòng (bắt đầu từ 1) trong code đã sửa")
    before: str
    after: str
    reason: str
    confidence: Literal["high", "medium", "low"] = "medium"


class UncertainSpot(BaseModel):
    line: int
    text: str
    note: str = Field(..., description="Vì sao không chắc: lỗi OCR hay lỗi của sinh viên")


class CorrectOcrResponse(BaseModel):
    code: str = Field(..., description="Code sau khi sửa lỗi nhận dạng")
    changes: List[CodeChange] = []
    uncertain: List[UncertainSpot] = Field(
        [], description="Chỗ LLM không chắc nên GIỮ NGUYÊN, cần giảng viên xem"
    )
    syntax_ok: bool = Field(..., description="Code sau khi sửa có parse được bằng Python không")
    syntax_error: Optional[str] = None
    similarity: float = Field(..., description="Độ giống nhau giữa OCR và code đã sửa (0..1)")
    needs_review: bool = Field(..., description="True nếu LLM sửa quá nhiều -> giảng viên nên kiểm tra")
    review_reasons: List[str] = []
    used_image: bool = Field(False, description="Có gửi ảnh gốc cho LLM đối chiếu không")
    model: str


# ============================================================================
# 2. NHẬN XÉT BÀI LÀM  (POST /api/v1/llm/review)
#    code + kết quả sandbox -> nhận xét + điểm gợi ý
# ============================================================================


class TestCaseResult(BaseModel):
    """Kết quả 1 test case do sandbox trả về."""

    __test__ = False  # để pytest không nhầm đây là class test

    name: Optional[str] = None
    input: Optional[str] = None
    expected: Optional[str] = None
    actual: Optional[str] = None
    passed: bool
    error: Optional[str] = Field(None, description="Traceback / lỗi runtime nếu có")
    hidden: bool = Field(False, description="Test ẩn: không đưa input/expected vào nhận xét cho SV")


class ReviewRequest(BaseModel):
    problem: str = Field(..., description="Đề bài (nên có cả format input/output và giới hạn n nếu có)")
    code: str = Field(..., description="Code cuối cùng (đã sửa lỗi OCR / giảng viên đã duyệt)")
    test_results: List[TestCaseResult] = []
    compile_error: Optional[str] = Field(None, description="Lỗi cú pháp/biên dịch nếu code không chạy được")
    rubric: Optional[str] = Field(None, description="Tiêu chí chấm của giảng viên (không bắt buộc)")
    max_score: float = 10
    language: Literal["python"] = "python"


ISSUE_TYPES = ("requirement", "logic", "syntax", "runtime", "io_format", "standard", "performance", "style")
IssueType = Literal["requirement", "logic", "syntax", "runtime", "io_format", "standard", "performance", "style"]


class ReviewIssue(BaseModel):
    """Một đoạn code cần sửa (dùng cho phần 'gợi ý đoạn cần sửa')."""

    line: Optional[int] = None
    code_snippet: Optional[str] = Field(None, description="Đoạn code hiện tại có vấn đề")
    type: IssueType = Field(..., description=(
        "requirement: sai/thiếu yêu cầu đề | logic | syntax | runtime | "
        "io_format: prompt trong input(), in thừa, sai format output | "
        "standard: không đúng chuẩn ngôn ngữ / lập trình thi đấu | performance: chưa tối ưu | style"))
    severity: Literal["high", "medium", "low"] = "medium"
    explanation: str
    suggestion: str = Field(..., description="Gợi ý hướng sửa (mô tả, không viết lời giải hoàn chỉnh)")


# ---- Phần 1: Đánh giá chung
class GeneralAssessment(BaseModel):
    meets_requirements: Literal["yes", "partial", "no"]
    requirements_comment: str = Field(..., description="Code có đúng yêu cầu đề bài không, thiếu gì")
    standard_comment: str = Field(..., description=(
        "Đúng chuẩn ngôn ngữ / lập trình thi đấu không: đọc stdin, in đúng format, "
        "không có prompt/comment thừa trong input(), không in debug"))
    optimization_comment: str = Field(..., description="Nhận xét về độ tối ưu")
    complexity: Optional[str] = Field(None, description="Độ phức tạp của code hiện tại, vd. O(n^2)")
    issues: List[ReviewIssue] = Field([], description="Các đoạn cần sửa, kèm dòng và gợi ý")


# ---- Phần 2: Hướng dẫn giải
class SolutionGuide(BaseModel):
    approach: str = Field(..., description="Phương pháp / ý tưởng thuật toán")
    steps: List[str] = Field([], description="Các bước giải")
    pseudocode: str = Field(..., description="Mã giả, KHÔNG phải code chạy được")
    expected_complexity: Optional[str] = None


# ---- Phần 3: Kết luận
class Conclusion(BaseModel):
    verdict: Literal["correct", "partially_correct", "incorrect"]
    summary: str = Field(..., description="Kết luận đúng/sai trong 1 đoạn ngắn")


class ReviewResponse(BaseModel):
    general: GeneralAssessment
    solution_guide: SolutionGuide
    conclusion: Conclusion
    suggested_score: float
    max_score: float
    score_explanation: str
    feedback_for_student: str
    passed_tests: int
    total_tests: int
    warnings: List[str] = Field([], description="Cảnh báo do server tự kiểm tra (vd. LLM đưa code thật thay vì mã giả)")
    model: str
