import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .base import RunResult, RunStatus
from .classify import classify
from .precheck import precheck


LAUNCHER = str(Path(__file__).with_name("_launcher.py"))
MAX_OUTPUT = 64 * 1024
MAX_INPUT = 64 * 1024


class SubprocessRunner:
    """Best-effort local runner; use DockerRunner for untrusted production code."""

    def __init__(self, mem_mb=128, cpu_s=5, fsize_mb=1):
        self.mem_mb, self.cpu_s, self.fsize_mb = mem_mb, cpu_s, fsize_mb

    def run(self, code: str, stdin: str = "", timeout_s: float = 5) -> RunResult:
        if len(stdin.encode("utf-8")) > MAX_INPUT:
            return RunResult(
                status=RunStatus.INPUT_LIMIT,
                stderr=f"stdin exceeds the {MAX_INPUT} byte limit",
            )

        with tempfile.TemporaryDirectory(prefix="sbx_") as tmp_name:
            tmp = Path(tmp_name)
            (tmp / "main.py").write_text(code, encoding="utf-8")
            (tmp / "stdin.txt").write_text(stdin, encoding="utf-8")

            env = {
                "PATH": os.defpath,
                "PYTHONNOUSERSITE": "1",
                "PYTHONUTF8": "1",
            }
            if os.name == "nt":
                env["SystemRoot"] = os.environ.get("SystemRoot", "")

            if os.name == "nt":
                t0 = time.monotonic()
                try:
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            LAUNCHER,
                            str(self.cpu_s),
                            str(self.mem_mb),
                            str(self.fsize_mb),
                            "main.py",
                        ],
                        cwd=tmp,
                        env=env,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                    )
                    timed_out = False
                    try:
                        stdout_bytes, stderr_bytes = process.communicate(
                            input=stdin.encode("utf-8"), timeout=timeout_s
                        )
                    except subprocess.TimeoutExpired:
                        subprocess.run(
                            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                            capture_output=True,
                            timeout=5,
                        )
                        stdout_bytes, stderr_bytes = process.communicate()
                        timed_out = True
                except OSError as exc:
                    return RunResult(
                        status=RunStatus.SANDBOX_ERROR,
                        stderr=str(exc),
                        error_type=type(exc).__name__,
                    )
                ms = int((time.monotonic() - t0) * 1000)
                stdout = stdout_bytes[:MAX_OUTPUT].decode("utf-8", "replace")
                stderr = stderr_bytes[:MAX_OUTPUT].decode("utf-8", "replace")
                return classify(
                    process.returncode,
                    stdout,
                    stderr,
                    ms,
                    timed_out,
                    len(stdout_bytes) > MAX_OUTPUT or len(stderr_bytes) > MAX_OUTPUT,
                )

            t0 = time.monotonic()
            with (
                open(tmp / "stdin.txt", encoding="utf-8") as fin,
                open(tmp / "out.txt", "wb") as fout,
                open(tmp / "err.txt", "wb") as ferr,
            ):
                try:
                    p = subprocess.Popen(
                        [
                            sys.executable,
                            LAUNCHER,
                            str(self.cpu_s),
                            str(self.mem_mb),
                            str(self.fsize_mb),
                            "main.py",
                        ],
                        cwd=tmp,
                        env=env,
                        stdin=fin,
                        stdout=fout,
                        stderr=ferr,
                        close_fds=True,
                        start_new_session=os.name != "nt",
                        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
                    )
                except OSError as exc:
                    return RunResult(
                        status=RunStatus.SANDBOX_ERROR,
                        stderr=str(exc),
                        error_type=type(exc).__name__,
                    )

                timed_out = False
                try:
                    p.wait(timeout=timeout_s)
                except subprocess.TimeoutExpired:
                    if os.name == "nt":
                        subprocess.run(
                            ["taskkill", "/PID", str(p.pid), "/T", "/F"],
                            capture_output=True,
                            timeout=5,
                        )
                    else:
                        os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
                    timed_out = True

            ms = int((time.monotonic() - t0) * 1000)
            stdout = (tmp / "out.txt").read_bytes()[:MAX_OUTPUT].decode("utf-8", "replace")
            stderr = (tmp / "err.txt").read_bytes()[:MAX_OUTPUT].decode("utf-8", "replace")
            output_exceeded = (tmp / "out.txt").stat().st_size > MAX_OUTPUT or (
                tmp / "err.txt"
            ).stat().st_size > MAX_OUTPUT
            return classify(p.returncode, stdout, stderr, ms, timed_out, output_exceeded)

    def run_with_precheck(self, code: str, stdin: str = "", timeout_s: float = 5) -> RunResult:
        result = precheck(code)
        return result if result is not None else self.run(code, stdin, timeout_s)

    def run_code(self, code: str, stdin: str = "", timeout_s: float = 5) -> bool:
        return self.run_with_precheck(code, stdin, timeout_s).ok
