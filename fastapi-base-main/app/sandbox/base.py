from dataclasses import dataclass
from enum import Enum
from typing import Protocol

class RunStatus(str, Enum):
    OK = "ok"
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error"
    TIMEOUT = "timeout"
    MEMORY_LIMIT = "memory_limit"
    OUTPUT_LIMIT = "output_limit"
    INPUT_LIMIT = "input_limit"
    SANDBOX_ERROR = "sandbox_error"
    LLM_ERROR = "llm_error"
    LLM_REJECTED = "llm_rejected"

@dataclass
class RunResult:
    status: RunStatus
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    duration_ms: int = 0
    error_type: str | None = None
    error_line: int | None = None

    @property
    def ok(self) -> bool:
        return self.status is RunStatus.OK

    def as_dict(self) -> dict:
        """Return a JSON/DB-friendly representation of the result."""
        return {
            "status": self.status.value,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
            "duration_ms": self.duration_ms,
            "error_type": self.error_type,
            "error_line": self.error_line,
        }

class SandboxRunner(Protocol):
    def run(self, code: str, stdin: str = "", timeout_s: float = 5) -> RunResult: ...

    def run_with_precheck(
        self, code: str, stdin: str = "", timeout_s: float = 5
    ) -> RunResult: ...
