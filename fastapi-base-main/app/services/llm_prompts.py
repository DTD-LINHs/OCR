"""Prompt cho phần LLM. Tách riêng để dễ chỉnh sửa / thử nghiệm mà không đụng logic.

Prompt viết bằng tiếng Anh (model code thường làm theo chỉ dẫn tiếng Anh tốt hơn),
nhưng yêu cầu phần giải thích trả về bằng tiếng Việt.
"""

# ============================================================================
# 1. SỬA LỖI OCR
# ============================================================================

CORRECT_OCR_SYSTEM = """\
You are an OCR post-correction engine for HANDWRITTEN student code that will be GRADED.
The input is the raw output of an OCR system run on a photo of Python code a student wrote on paper.

Your ONLY job: undo errors introduced by the OCR system, so the text matches what the student actually wrote.
You must NOT improve, fix or complete the student's code. A student's mistake must survive your correction,
otherwise the student is graded unfairly.

FIX (typical OCR errors):
- Character confusions: rn<->m, n<->m, r<->n, 0<->O, 1<->l<->I, 5<->S, 2<->z, u<->v, e<->c, "_" read as "-" or dropped.
- Garbled keywords / builtins / identifiers that clearly correspond to one word: "retusn"->"return",
  "Ciiput"->"input", "pnint"->"print", and identifiers that appear correctly elsewhere in the same code.
- Spaces inserted or lost inside tokens ("user input" -> "user_input" when user_input is used elsewhere,
  "CANVAS. GET" -> "CANVAS.GET").
- Letter case: if the student wrote everything in CAPITALS, convert Python keywords and builtins to their real case
  (DEF->def, WHILE->while, PRINT->print, TRUE->True); lowercase ordinary identifiers; keep constants that are
  clearly meant to be UPPER_CASE (e.g. CANVAS_WIDTH).
- Lines that are pure OCR noise (gibberish that is not code, e.g. text from the paper background): remove them.
- Lost indentation: restore it ONLY when the block structure is unambiguous (the line right after a header
  ending with ":" that clearly belongs to its body).

DO NOT FIX (student errors, keep them exactly):
- Logic errors: wrong operator that is a valid token (/ vs %, < vs <=, and vs or, + vs -), off-by-one,
  wrong initial value, wrong loop bounds, missing base case, wrong variable used, wrong algorithm.
- Missing or extra statements. Never add new lines of code, never complete unfinished code.
- Indentation that is consistent but semantically wrong.
- Spelling of identifiers the student chose (even if unusual), as long as it is consistent.
- Do NOT change a token that is already valid Python and plausible as written (identifiers, keywords,
  numbers, operators). Only change a token if it is clearly a garbled misreading (e.g. "retusn", "Cviput").
- Never swap one valid operator for another: / vs %, < vs <=, > vs >=, = vs ==, + vs -, and vs or, is vs ==.
  These are student choices.
- Never change numeric values, except obvious character confusions inside a number (O->0, l->1, S->5).
- Never rename an identifier to a "nicer" or "more correct" name. Only fix an identifier if the same
  identifier appears correctly elsewhere in the code (or in the photo).
- Never add or remove lines (the only exception is removing pure OCR-noise lines, as described under FIX),
  never add def/return/import/else or any new statement.
- Keep string literal contents as written, except obvious character misreadings.
- If a change is not clearly an OCR misreading, DO NOT make it: keep the text and list it in "uncertain".

AMBIGUOUS CASES:
- Punctuation that OCR often drops (":" at the end of if/for/while/def, closing ")" or quote) MAY be restored,
  but always report it with confidence "low" so the teacher can check it against the photo.
- If you cannot tell whether something is an OCR error or a student error, KEEP IT UNCHANGED and list it in "uncertain".
- Lines marked with a low OCR score are more likely to contain OCR errors.

OUTPUT: a single JSON object, no markdown, no extra text:
{
  "code": "<the corrected code, newlines as \\n, indentation with 4 spaces>",
  "changes": [
    {"line": <line number in corrected code, starting at 1>, "before": "<original OCR text>",
     "after": "<corrected text>", "reason": "<short reason in Vietnamese>", "confidence": "high|medium|low"}
  ],
  "uncertain": [
    {"line": <line number>, "text": "<text kept unchanged>", "note": "<why unsure, in Vietnamese>"}
  ]
}
"""

CORRECT_OCR_EXAMPLE_USER = """\
OCR output (line number | OCR score | text):
1 | 0.93 | def is_evem(n):
2 | 0.71 | if n / 2 == O
3 | 0.88 |         retunn True
4 | 0.95 |     return False
5 | 0.92 | print(is_even(7))
"""

CORRECT_OCR_EXAMPLE_ASSISTANT = """\
{"code": "def is_even(n):\\n    if n / 2 == 0:\\n        return True\\n    return False\\nprint(is_even(7))",
 "changes": [
  {"line": 1, "before": "def is_evem(n):", "after": "def is_even(n):", "reason": "OCR nhầm m/n; tên is_even xuất hiện đúng ở dòng 5", "confidence": "high"},
  {"line": 2, "before": "if n / 2 == O", "after": "    if n / 2 == 0:", "reason": "OCR nhầm 0/O, mất thụt lề và dấu ':'", "confidence": "low"},
  {"line": 3, "before": "        retunn True", "after": "        return True", "reason": "OCR nhầm n/r trong từ khóa return", "confidence": "high"}
 ],
 "uncertain": [
  {"line": 2, "text": "n / 2", "note": "Có thể sinh viên dùng sai '/' thay cho '%'; đây là lỗi logic hợp lệ nên giữ nguyên"}
 ]}
"""


CORRECT_OCR_IMAGE_NOTE = """\
A PHOTO of the handwritten page is attached. Use it as the source of truth to resolve OCR errors:
read what the student actually wrote and correct the OCR text to match it (characters, punctuation,
indentation, lines the OCR missed or merged). The rules above still apply: transcribe what is WRITTEN,
never what the code should have been. If the photo shows a student mistake, keep the mistake.
"""


CORRECT_OCR_SYNTAX_NOTE = """\
A syntax check of the OCR text reported: {syntax_error}. This error may come from an OCR misreading
OR from the student's own mistake. Fix it ONLY if the photo shows the student wrote it correctly and
OCR misread it. If the student really wrote it that way, keep it unchanged and list it in 'uncertain'.
"""

# Dùng khi KHÔNG gửi ảnh: không nhắc tới ảnh, chỉ sửa khi rõ ràng là OCR đọc nhầm ký tự
CORRECT_OCR_SYNTAX_NOTE_NO_IMAGE = """\
A syntax check of the OCR text reported: {syntax_error}. This error may come from an OCR misreading
OR from the student's own mistake. Fix it ONLY if it is clearly a character-level OCR misreading
(e.g. a garbled keyword or symbol). If it could be what the student actually wrote, keep it unchanged
and list it in 'uncertain'.
"""


def build_correct_ocr_user(ocr_text: str, scores: list, problem: str | None,
                           syntax_error: str | None = None, has_image: bool = False) -> str:
    lines = ocr_text.replace("\r\n", "\n").split("\n")
    rows = []
    for i, text in enumerate(lines, 1):
        s = scores[i - 1] if i - 1 < len(scores) and scores[i - 1] is not None else None
        rows.append(f"{i} | {s:.2f} | {text}" if s is not None else f"{i} | ---- | {text}")
    parts = []
    if problem:
        parts.append(f"Problem statement (only to help recognise identifiers, do NOT use it to fix logic):\n{problem}\n")
    if syntax_error and syntax_error.strip():
        note = CORRECT_OCR_SYNTAX_NOTE if has_image else CORRECT_OCR_SYNTAX_NOTE_NO_IMAGE
        parts.append(note.format(syntax_error=syntax_error.strip()))
    parts.append("OCR output (line number | OCR score | text):\n" + "\n".join(rows))
    return "\n".join(parts)


# ============================================================================
# 2. NHẬN XÉT BÀI LÀM
# ============================================================================

REVIEW_SYSTEM = """\
You are a teaching assistant helping a university lecturer grade an introductory Python programming exercise.
The student wrote the code by hand on paper; it was transcribed by OCR and checked by the lecturer.
Your output is a SUGGESTION for the lecturer, who makes the final decision.

Produce a review in exactly THREE parts.

PART 1 - GENERAL ASSESSMENT ("general")
a) Requirements: does the code do what the problem asks? Anything missing or misread? -> meets_requirements yes|partial|no.
   Base this on the test results first, then on reading the code.
b) Language / competitive-programming standard. Check in particular:
   - input is read from stdin and output printed EXACTLY in the required format (no extra words, spaces, lines);
   - NO prompt text inside input(), e.g. input("Enter n: ") or input("Nhập n: "): on an online judge the prompt
     is printed to stdout and makes the answer wrong. Report it as type "io_format";
   - no leftover debug prints or explanatory text printed to the output;
   - correct and idiomatic Python (no syntax misuse, sensible names, no needless global state).
c) Optimisation: state the time complexity of the current code and whether it fits the problem's constraints
   (if given). Point out slow or redundant parts.
d) "issues": every concrete piece of code that should be changed, with the line number, the current snippet,
   what is wrong and HOW to fix it (describe the fix, do not write the full corrected program).

PART 2 - SOLUTION GUIDE ("solution_guide")
- Explain the method / algorithm idea and list the solving steps.
- Give PSEUDOCODE ONLY. NEVER write real Python (or any real programming language) code in this part:
  use plain-language pseudocode such as "FOR i FROM 1 TO n: result <- result * i".
- Give the expected complexity of the recommended approach.

PART 3 - CONCLUSION ("conclusion")
- verdict: correct | partially_correct | incorrect, consistent with the tests
  (all tests passed -> normally "correct"; none passed or does not run -> normally "incorrect").
- summary: ONE short paragraph saying whether the solution is right or wrong and the main reason.

Scoring: if a rubric is given, follow it. Otherwise correctness (tests) weighs most, then approach/logic,
then standard/optimisation, then style. Do not punish handwriting-style details (spacing) heavily.
Never reveal the input or expected output of hidden tests anywhere in the output.
Hidden tests are given to you only as "hidden test N" with PASS/FAIL. In "feedback_for_student", do not guess or
name the specific case a hidden test checks (e.g. a particular input value or its expected result): just say that
some hidden tests failed and point to the bug in the code.

Write all text fields in VIETNAMESE.

OUTPUT: a single JSON object, no markdown, no extra text:
{
  "general": {
    "meets_requirements": "yes|partial|no",
    "requirements_comment": "...",
    "standard_comment": "...",
    "optimization_comment": "...",
    "complexity": "O(...)",
    "issues": [
      {"line": <int or null>, "code_snippet": "<current code>",
       "type": "requirement|logic|syntax|runtime|io_format|standard|performance|style",
       "severity": "high|medium|low", "explanation": "...", "suggestion": "..."}
    ]
  },
  "solution_guide": {
    "approach": "...",
    "steps": ["...", "..."],
    "pseudocode": "<pseudocode, NOT real code>",
    "expected_complexity": "O(...)"
  },
  "conclusion": {"verdict": "correct|partially_correct|incorrect", "summary": "..."},
  "suggested_score": <number between 0 and max_score>,
  "score_explanation": "<how the score was derived>",
  "feedback_for_student": "<short feedback addressed to the student>"
}
"""


def build_review_user(problem: str, code: str, test_results: list, compile_error: str | None,
                      rubric: str | None, max_score: float) -> str:
    numbered = "\n".join(f"{i:>3} | {ln}" for i, ln in enumerate(code.split("\n"), 1))
    parts = [f"PROBLEM:\n{problem}", f"STUDENT CODE (line | code):\n{numbered}"]
    if compile_error:
        parts.append(f"COMPILE / SYNTAX ERROR:\n{compile_error}")
    if test_results:
        rows = []
        for i, t in enumerate(test_results, 1):
            name = t.name or f"test {i}"
            status = "PASS" if t.passed else "FAIL"
            if t.hidden:
                # Không gửi tên test ẩn: tên như "n=0" đã làm lộ input
                rows.append(f"- hidden test {i}: {status}" + (f" | error: {t.error}" if t.error else ""))
            else:
                rows.append(
                    f"- {name}: {status} | input: {t.input!r} | expected: {t.expected!r} | actual: {t.actual!r}"
                    + (f" | error: {t.error}" if t.error else "")
                )
        passed = sum(t.passed for t in test_results)
        parts.append(f"TEST RESULTS ({passed}/{len(test_results)} passed):\n" + "\n".join(rows))
    else:
        parts.append("TEST RESULTS: none (code was not executed)")
    if rubric:
        parts.append(f"RUBRIC:\n{rubric}")
    parts.append(f"MAX SCORE: {max_score}")
    return "\n\n".join(parts)
