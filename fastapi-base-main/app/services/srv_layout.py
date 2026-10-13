"""Rebuild code lines and indentation from PaddleOCR boxes.

OCR returns loose boxes without leading whitespace, so the layout of the code has
to be recovered from geometry:
- boxes whose vertical spans overlap belong to the same line,
- the left edge of each line, measured in character widths, is clustered into
  indentation levels,
- a large vertical gap between two lines becomes a blank line.
"""

import statistics
from typing import Iterable

from app.schemas.sche_ocr_pipeline import CodeLine, LayoutResult, OcrBox

SAME_LINE_MIN_OVERLAP = 0.5
INDENT_TOLERANCE_CHARS = 1.5
BLANK_LINE_GAP_RATIO = 1.0
WORD_GAP_CHARS = 0.5


def build_layout(
    boxes: Iterable[OcrBox],
    indent_unit: str = "    ",
    low_confidence_threshold: float = 0.8,
) -> LayoutResult:
    items = [box for box in boxes if box.text.strip()]
    if not items:
        return LayoutResult(lines=[], indent_unit=indent_unit)

    median_height = statistics.median(box.y_max - box.y_min for box in items) or 1.0
    char_width = _estimate_char_width(items, median_height)
    rows = _group_rows(items)
    levels = _indent_levels([row[0].x_min for row in rows], char_width)
    # The first line of a program is never indented. Shifting to it keeps a stray box
    # further left (a margin note, a smudge) from pushing the whole program one level in.
    levels = [max(level - levels[0], 0) for level in levels]

    lines: list[CodeLine] = []
    prev_bottom = None
    for row, level in zip(rows, levels):
        top = min(box.y_min for box in row)
        if prev_bottom is not None and top - prev_bottom > BLANK_LINE_GAP_RATIO * median_height:
            lines.append(CodeLine(line_no=len(lines) + 1, text=""))
        confidence = min(box.confidence for box in row)
        lines.append(
            CodeLine(
                line_no=len(lines) + 1,
                text=_join_row(row, char_width),
                indent_level=level,
                confidence=confidence,
                low_confidence=confidence < low_confidence_threshold,
            )
        )
        prev_bottom = max(box.y_max for box in row)

    return LayoutResult(lines=lines, indent_unit=indent_unit)


def _estimate_char_width(boxes: list[OcrBox], median_height: float) -> float:
    widths = [
        (box.x_max - box.x_min) / len(box.text.strip())
        for box in boxes
        if len(box.text.strip()) >= 2
    ]
    if not widths:
        return median_height * 0.5
    return statistics.median(widths) or median_height * 0.5


def _vertical_overlap(box: OcrBox, top: float, bottom: float) -> float:
    overlap = min(box.y_max, bottom) - max(box.y_min, top)
    shortest = min(box.y_max - box.y_min, bottom - top) or 1.0
    return max(overlap, 0.0) / shortest


def _group_rows(boxes: list[OcrBox]) -> list[list[OcrBox]]:
    rows: list[list[OcrBox]] = []
    for box in sorted(boxes, key=lambda b: (b.y_min + b.y_max) / 2):
        if rows:
            last = rows[-1]
            top = statistics.mean(b.y_min for b in last)
            bottom = statistics.mean(b.y_max for b in last)
            if _vertical_overlap(box, top, bottom) >= SAME_LINE_MIN_OVERLAP:
                last.append(box)
                continue
        rows.append([box])
    return [sorted(row, key=lambda b: b.x_min) for row in rows]


def _indent_levels(lefts: list[float], char_width: float) -> list[int]:
    """Cluster left edges (in characters) and use the cluster rank as indent level.

    Ranking instead of dividing by a fixed width copes with handwriting, where
    one indentation step is rarely the same number of characters twice.
    """
    base = min(lefts)
    offsets = [(left - base) / char_width for left in lefts]

    clusters: list[list[float]] = []
    for offset in sorted(offsets):
        if clusters and offset - statistics.mean(clusters[-1]) <= INDENT_TOLERANCE_CHARS:
            clusters[-1].append(offset)
        else:
            clusters.append([offset])
    centers = [statistics.mean(cluster) for cluster in clusters]

    return [
        min(range(len(centers)), key=lambda i: abs(centers[i] - offset))
        for offset in offsets
    ]


def _join_row(row: list[OcrBox], char_width: float) -> str:
    text = row[0].text.strip()
    for prev, box in zip(row, row[1:]):
        gap = box.x_min - prev.x_max
        separator = " " if gap > WORD_GAP_CHARS * char_width else ""
        text += separator + box.text.strip()
    return text
