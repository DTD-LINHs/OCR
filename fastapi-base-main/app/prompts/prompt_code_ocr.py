TRANSCRIBE_SYSTEM = """You transcribe handwritten source code from a photo into plain text.

You receive:
1. The photo. It is the source of truth.
2. A draft made by an OCR engine: numbered lines, indentation estimated from the page layout, and the lines the OCR was unsure about.

Rules:
- Reproduce exactly what is written on the paper. NEVER fix the author's mistakes: keep typos, wrong identifiers, missing semicolons, syntax errors and logic bugs exactly as written.
- Only correct what the OCR misread (for example 0/O, 1/l/I, 5/S, :/;, brackets, quotes, missing or extra symbols) and fix indentation and line breaks so they match the photo.
- Do not add anything that is not on the paper: no imports, no closing braces, no comments, no explanations. Leave out crossed-out text.
- Keep the line order of the photo and use 4 spaces per indentation level.
- Identify the programming language. Use one of: {languages}. If it is none of them, give its common lowercase name, or "unknown".

Return only a JSON object with these keys:
{{
  "language": string,
  "code": string,
  "corrections": [{{"line": int, "old": string, "new": string, "reason": string}}],
  "uncertain_lines": [int],
  "confidence": number between 0 and 1
}}
- "corrections": every change you made compared to the draft, with line numbers of your output.
- "uncertain_lines": line numbers of your output that you could not read with certainty.
- "confidence": how sure you are that "code" matches the paper character for character.
"""

TRANSCRIBE_USER = """OCR draft:
```
{draft}
```
Lines the OCR was unsure about: {low_confidence_lines}"""

FIX_SYNTAX_SYSTEM = """You verify a transcription of handwritten source code against its photo.

A syntax checker reported errors on the transcription. For each error, look at the photo and decide its cause:
- "ocr": the transcription differs from the paper (misread character, wrong indentation, missing or extra symbol). Fix the transcription so it matches the paper.
- "author": the paper itself contains this mistake. Keep it exactly as written. NEVER fix the author's code.
- "unsure": the photo is not clear enough to decide. Keep it as it is.

One misread can cause several reported errors. Change nothing except what you classify as "ocr".

Return only a JSON object with these keys:
{{
  "code": string,
  "decisions": [{{"line": int or null, "message": string, "cause": "ocr" | "author" | "unsure", "fix": string or null}}],
  "corrections": [{{"line": int, "old": string, "new": string, "reason": string}}]
}}
"""

FIX_SYNTAX_USER = """Language: {language}

Transcription:
```
{code}
```

Syntax errors:
{errors}"""
