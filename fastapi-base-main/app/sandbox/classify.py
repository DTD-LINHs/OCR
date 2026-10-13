import re

from .base import RunResult, RunStatus


_DOCKER_INFRA_PATTERNS = (
    "failed to connect to the docker api",
    "failed to connect to the docker daemon",
    "error response from daemon",
    "cannot connect to the docker daemon",
    "unable to find image",
    "is the docker daemon running",
    "docker: command not found",
)

_LAST_LINE_TYPE = re.compile(r"^([A-Za-z_][\w.]*)\s*(?::|$)")
_MAIN_FRAME = re.compile(r'File "[^"]*main\.py", line (\d+)')


def _parse_traceback(stderr: str) -> tuple[str | None, int | None]:
    lines = [line for line in stderr.strip().splitlines() if line.strip()]
    error_type = None
    if lines:
        match = _LAST_LINE_TYPE.match(lines[-1])
        error_type = match.group(1) if match else None
    frames = _MAIN_FRAME.findall(stderr)
    line = int(frames[-1]) if frames else None
    return error_type, line


def classify_process(
    rc: int | None,
    stdout: str,
    stderr: str,
    ms: int,
    timed_out: bool,
    output_exceeded: bool = False,
) -> RunResult:
    base = dict(stdout=stdout, stderr=stderr, exit_code=rc, duration_ms=ms)

    if output_exceeded:
        return RunResult(status=RunStatus.OUTPUT_LIMIT, **base)
    if timed_out:
        return RunResult(status=RunStatus.TIMEOUT, **base)
    if rc == 0:
        return RunResult(status=RunStatus.OK, **base)
    if rc is None:
        return RunResult(status=RunStatus.SANDBOX_ERROR, **base)

    low = (stderr or "").lower()
    if rc in (125, 126, 127) or any(pattern in low for pattern in _DOCKER_INFRA_PATTERNS):
        return RunResult(status=RunStatus.SANDBOX_ERROR, **base)
    if rc == 124:
        return RunResult(status=RunStatus.TIMEOUT, **base)
    # A SIGKILL after the timeout flag has been handled is most commonly the
    # cgroup memory limit. Docker reports it as 137; POSIX reports -9.
    if rc in (137, -9):
        return RunResult(status=RunStatus.MEMORY_LIMIT, **base)

    error_type, error_line = _parse_traceback(stderr or "")
    status = RunStatus.MEMORY_LIMIT if error_type == "MemoryError" else RunStatus.RUNTIME_ERROR
    return RunResult(
        status=status,
        error_type=error_type,
        error_line=error_line,
        **base,
    )


def classify_docker(*args, **kwargs) -> RunResult:
    return classify_process(*args, **kwargs)


# Compatibility name used by the subprocess backend.
classify = classify_process
