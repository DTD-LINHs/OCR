"""Data passed between the OCR pipeline stages (Paddle -> service -> LLM -> sandbox)."""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class OcrBox(BaseModel):
    """One text box detected by PaddleOCR."""

    text: str
    polygon: list[list[float]]
    confidence: float

    @property
    def x_min(self) -> float:
        return min(p[0] for p in self.polygon)

    @property
    def x_max(self) -> float:
        return max(p[0] for p in self.polygon)

    @property
    def y_min(self) -> float:
        return min(p[1] for p in self.polygon)

    @property
    def y_max(self) -> float:
        return max(p[1] for p in self.polygon)


class CodeLine(BaseModel):
    line_no: int
    text: str
    indent_level: int = 0
    confidence: float = 1.0
    low_confidence: bool = False


class LayoutResult(BaseModel):
    lines: list[CodeLine]
    indent_unit: str = "    "

    def render(self) -> str:
        return "\n".join(
            (self.indent_unit * line.indent_level + line.text) if line.text else ""
            for line in self.lines
        )

    @property
    def low_confidence_lines(self) -> list[int]:
        return [line.line_no for line in self.lines if line.low_confidence]


class Correction(BaseModel):
    line: Optional[int] = None
    old: str = ""
    new: str = ""
    reason: str = ""


class LlmTranscription(BaseModel):
    language: str = "unknown"
    code: str
    corrections: list[Correction] = []
    uncertain_lines: list[int] = []
    confidence: float = Field(default=0.5, ge=0, le=1)


class SyntaxDecision(BaseModel):
    line: Optional[int] = None
    message: str = ""
    cause: Literal["ocr", "author", "unsure"]
    fix: Optional[str] = None


class LlmSyntaxFix(BaseModel):
    code: str
    decisions: list[SyntaxDecision] = []
    corrections: list[Correction] = []


class SyntaxIssue(BaseModel):
    line: Optional[int] = None
    message: str


class SandboxResult(BaseModel):
    language: str
    image: str
    command: str
    exit_code: Optional[int] = None
    ok: bool
    timed_out: bool = False
    errors: list[SyntaxIssue] = []
    output: str = ""
    duration_ms: int = 0


class LlmCallRecord(BaseModel):
    purpose: str
    model: str
    prompt: str
    response: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    latency_ms: int = 0
    parse_ok: bool = False
    error: Optional[str] = None


class PipelineResult(BaseModel):
    language: str
    code: str
    syntax_ok: Optional[bool] = None
    syntax_errors: list[SyntaxIssue] = []
    uncertain_lines: list[int] = []
    confidence: float
    need_review: bool
    review_reasons: list[str] = []
