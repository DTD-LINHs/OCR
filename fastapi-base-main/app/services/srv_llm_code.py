"""GPT-4o (vision) calls used by the pipeline: transcribe the photo and arbitrate syntax errors."""

import base64
import json
import time
from typing import Callable, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

from app.prompts.prompt_code_ocr import (
    FIX_SYNTAX_SYSTEM,
    FIX_SYNTAX_USER,
    TRANSCRIBE_SYSTEM,
    TRANSCRIBE_USER,
)
from app.schemas.sche_ocr_pipeline import (
    LayoutResult,
    LlmCallRecord,
    LlmSyntaxFix,
    LlmTranscription,
    SyntaxIssue,
)
from app.utils.enums import LlmPurpose

T = TypeVar("T", bound=BaseModel)
CallLogger = Callable[[LlmCallRecord], None]

MAX_PARSE_ATTEMPTS = 2


class LlmError(Exception):
    pass


def number_lines(code: str) -> str:
    return "\n".join(f"{i:>3} | {line}" for i, line in enumerate(code.split("\n"), start=1))


class OpenAICodeLlm:
    def __init__(
        self,
        api_key: Optional[str],
        model: str = "gpt-4o",
        timeout: int = 120,
        supported_languages: tuple[str, ...] = (),
        client=None,
    ):
        self.model = model
        self.supported_languages = supported_languages
        self._api_key = api_key
        self._timeout = timeout
        self._client = client

    @property
    def client(self):
        if self._client is None:
            if not self._api_key:
                raise LlmError("OPENAI_API_KEY is not configured")
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key, timeout=self._timeout, max_retries=2)
        return self._client

    def transcribe(self, image_jpeg: bytes, draft: LayoutResult, log: CallLogger) -> LlmTranscription:
        system = TRANSCRIBE_SYSTEM.format(languages=", ".join(self.supported_languages) or "any")
        user = TRANSCRIBE_USER.format(
            draft=number_lines(draft.render()),
            low_confidence_lines=draft.low_confidence_lines or "none",
        )
        return self._call_json(LlmPurpose.TRANSCRIBE, system, user, image_jpeg, LlmTranscription, log)

    def fix_syntax(
        self,
        image_jpeg: bytes,
        code: str,
        language: str,
        errors: list[SyntaxIssue],
        log: CallLogger,
    ) -> LlmSyntaxFix:
        error_lines = "\n".join(
            f"- line {e.line if e.line is not None else '?'}: {e.message}" for e in errors
        )
        user = FIX_SYNTAX_USER.format(language=language, code=number_lines(code), errors=error_lines)
        return self._call_json(LlmPurpose.FIX_SYNTAX, FIX_SYNTAX_SYSTEM, user, image_jpeg, LlmSyntaxFix, log)

    def _call_json(
        self,
        purpose: LlmPurpose,
        system: str,
        user: str,
        image_jpeg: bytes,
        schema: Type[T],
        log: CallLogger,
    ) -> T:
        image_url = "data:image/jpeg;base64," + base64.b64encode(image_jpeg).decode("ascii")
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user},
                    {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}},
                ],
            },
        ]
        # The image is not stored in the log, only the text parts of the prompt.
        prompt_log = f"[system]\n{system}\n[user]\n{user}\n[image omitted]"

        last_error = None
        for _ in range(MAX_PARSE_ATTEMPTS):
            record = LlmCallRecord(purpose=purpose.value, model=self.model, prompt=prompt_log)
            started = time.monotonic()
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0,
                    response_format={"type": "json_object"},
                )
            except Exception as e:
                record.error = str(e)
                record.latency_ms = int((time.monotonic() - started) * 1000)
                log(record)
                raise LlmError(f"LLM request failed: {e}") from e

            record.latency_ms = int((time.monotonic() - started) * 1000)
            content = response.choices[0].message.content or ""
            record.response = content
            if response.usage:
                record.prompt_tokens = response.usage.prompt_tokens
                record.completion_tokens = response.usage.completion_tokens
            try:
                result = schema.model_validate(json.loads(content))
                record.parse_ok = True
                log(record)
                return result
            except (json.JSONDecodeError, ValidationError) as e:
                last_error = str(e)
                record.error = last_error
                log(record)
                messages.append({"role": "assistant", "content": content})
                messages.append(
                    {
                        "role": "user",
                        "content": f"Your reply was invalid: {last_error}. Return only the JSON object with the required keys.",
                    }
                )
        raise LlmError(f"LLM returned invalid JSON for {purpose.value}: {last_error}")
