from .base import RunResult, RunStatus, SandboxRunner
from .docker_runner import DockerRunner
from .factory import build_runner


def run_code(
    code: str,
    stdin: str = "",
    timeout_s: float = 5,
    runner: SandboxRunner | None = None,
) -> bool:
    """Return only whether the submission completed successfully.

    Detailed diagnostics remain available through ``runner.run_with_precheck``
    for logging or debugging, but callers that only need the pipeline decision
    can use this boolean API.
    """
    selected_runner = runner or DockerRunner()
    result = selected_runner.run_with_precheck(code, stdin=stdin, timeout_s=timeout_s)
    return result.ok


__all__ = [
    "DockerRunner",
    "RunResult",
    "RunStatus",
    "SandboxRunner",
    "build_runner",
    "run_code",
]
