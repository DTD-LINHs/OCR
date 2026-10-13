from .base import RunResult, RunStatus

def precheck(code: str) -> RunResult | None:
    if not isinstance(code, str):
        return RunResult(
            status=RunStatus.SANDBOX_ERROR,
            stderr="code must be a string",
            error_type="TypeError",
        )
    try:
        compile(code, "<sandbox>", "exec")
        return None
    except SyntaxError as e:
        return RunResult(
            status=RunStatus.SYNTAX_ERROR,
            stderr=f"{e.msg} (line {e.lineno})",
            error_type=type(e).__name__,
            error_line=e.lineno,
        )