"""Language-agnostic character clean-up applied before the LLM sees the draft.

Only substitutions that are safe in every programming language belong here;
context-dependent fixes (0/O, 1/l, ;/:) are left to the LLM, which sees the photo.
"""

import re
import unicodedata

from app.schemas.sche_ocr_pipeline import Correction, LayoutResult

_CHAR_MAP = {
    "“": '"', "”": '"', "„": '"', "‟": '"', "«": '"', "»": '"',
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "´": "'",
    "—": "-", "–": "-", "−": "-", "‐": "-", "‑": "-",
    "×": "*", "÷": "/",
    "≠": "!=", "≤": "<=", "≥": ">=",
    "→": "->", "⇒": "=>",
}
_ZERO_WIDTH = re.compile("[​‌‍⁠﻿]")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")


def normalize_text(text: str) -> str:
    # NFKC folds full-width letters/digits and ligatures into plain ASCII.
    text = unicodedata.normalize("NFKC", text)
    text = _ZERO_WIDTH.sub("", text)
    text = "".join(_CHAR_MAP.get(ch, ch) for ch in text)
    text = text.replace(" ", " ")
    return _MULTI_SPACE.sub(" ", text).strip()


def normalize_layout(layout: LayoutResult) -> tuple[LayoutResult, list[Correction]]:
    corrections: list[Correction] = []
    lines = []
    for line in layout.lines:
        new_text = normalize_text(line.text)
        if new_text != line.text:
            corrections.append(
                Correction(line=line.line_no, old=line.text, new=new_text, reason="normalize")
            )
        lines.append(line.model_copy(update={"text": new_text}))
    return LayoutResult(lines=lines, indent_unit=layout.indent_unit), corrections
