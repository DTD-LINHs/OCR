from app.schemas.sche_ocr_pipeline import OcrBox
from app.services.srv_layout import build_layout
from app.services.srv_normalize import normalize_layout, normalize_text

CHAR_W = 10
LINE_H = 20


def box(text, col, row, conf=0.95, y_shift=0.0):
    """Box for `text` starting at character column `col` on text row `row`."""
    x0 = col * CHAR_W
    x1 = x0 + len(text) * CHAR_W
    y0 = row * LINE_H * 1.5 + y_shift
    y1 = y0 + LINE_H
    return OcrBox(text=text, polygon=[[x0, y0], [x1, y0], [x1, y1], [x0, y1]], confidence=conf)


def test_rebuilds_python_indentation():
    boxes = [
        box("def f(x):", 0, 0),
        box("if x > 0:", 4, 1),
        box("return x", 8, 2),
        box("return -x", 4, 3),
    ]
    assert build_layout(boxes).render() == (
        "def f(x):\n"
        "    if x > 0:\n"
        "        return x\n"
        "    return -x"
    )


def test_stray_box_left_of_the_code_does_not_indent_it():
    boxes = [
        box("def f(x):", 4, 0),
        box("return x", 8, 1),
        box("~~", 0, 2),  # smudge in the margin, further left than the code
    ]
    assert build_layout(boxes).render() == "def f(x):\n    return x\n~~"


def test_handwritten_indent_drift_stays_on_same_level():
    boxes = [
        box("for i in range(3):", 0, 0),
        box("a = i", 3, 1),
        box("b = a", 4, 2),  # written a bit further right, same block
        box("print(b)", 0, 3),
    ]
    levels = [line.indent_level for line in build_layout(boxes).lines]
    assert levels == [0, 1, 1, 0]


def test_merges_boxes_on_same_line_and_orders_left_to_right():
    boxes = [
        box("= 5;", 6, 0, y_shift=3),
        box("int x", 0, 0),
    ]
    layout = build_layout(boxes)
    assert len(layout.lines) == 1
    assert layout.lines[0].text == "int x = 5;"


def test_large_vertical_gap_becomes_blank_line():
    boxes = [box("x = 1", 0, 0), box("y = 2", 0, 3)]
    assert build_layout(boxes).render() == "x = 1\n\ny = 2"


def test_flags_low_confidence_lines():
    boxes = [box("a = 1", 0, 0, conf=0.99), box("b = 0O", 0, 1, conf=0.4)]
    layout = build_layout(boxes, low_confidence_threshold=0.8)
    assert layout.low_confidence_lines == [2]


def test_empty_input():
    assert build_layout([]).lines == []


def test_normalize_text():
    assert normalize_text("print(“hi”)") == 'print("hi")'
    assert normalize_text("a ≠ b — c") == "a != b - c"
    assert normalize_text("ｘ = １") == "x = 1"  # full-width
    assert normalize_text("x  =   1​") == "x = 1"


def test_normalize_layout_records_corrections():
    layout = build_layout([box("s = ‘a’", 0, 0), box("ok = 1", 0, 1)])
    normalized, corrections = normalize_layout(layout)
    assert normalized.render() == "s = 'a'\nok = 1"
    assert [c.line for c in corrections] == [1]
