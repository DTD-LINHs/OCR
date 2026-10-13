import base64
import logging
import subprocess
import threading
import time
import uuid

from .base import RunResult, RunStatus
from .classify import classify_docker
from .precheck import precheck


IMAGE = "sandbox-python:latest"
MAX_OUT = 64 * 1024
MAX_INPUT = 64 * 1024
MAX_CODE = 64 * 1024
logger = logging.getLogger(__name__)


class DockerRunner:
    """Run one Python submission in a disposable, networkless container."""

    def __init__(self, mem_mb=128, cpus=0.5, pids=64, image: str = IMAGE):
        self.mem_mb, self.cpus, self.pids, self.image = mem_mb, cpus, pids, image

    @staticmethod
    def _kill(name: str) -> None:
        try:
            subprocess.run(["docker", "kill", name], capture_output=True, timeout=10)
            # The --rm cleanup is asynchronous on some Docker versions.
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            logger.warning("Timed out while killing Docker container %s", name, exc_info=True)

    @staticmethod
    def _drain(stream, buf: bytearray, exceeded: threading.Event) -> None:
        while True:
            chunk = stream.read1(4096)
            if not chunk:
                return
            room = MAX_OUT - len(buf)
            if room > 0:
                buf += chunk[:room]
            if len(chunk) > room:
                exceeded.set()

    @staticmethod
    def _feed(stream, data: bytes) -> None:
        try:
            stream.write(data)
        except (BrokenPipeError, OSError):
            pass
        finally:
            try:
                stream.close()
            except OSError:
                pass

    def run(self, code: str, stdin: str = "", timeout_s: float = 5) -> RunResult:
        if not isinstance(code, str):
            return RunResult(
                status=RunStatus.SANDBOX_ERROR,
                stderr="code must be a string",
                error_type="TypeError",
            )
        code_bytes = code.encode("utf-8")
        if len(code_bytes) > MAX_CODE:
            return RunResult(
                status=RunStatus.SANDBOX_ERROR,
                stderr=f"code exceeds the {MAX_CODE} byte limit",
            )
        stdin_bytes = stdin.encode("utf-8")
        if len(stdin_bytes) > MAX_INPUT:
            return RunResult(
                status=RunStatus.INPUT_LIMIT,
                stderr=f"stdin exceeds the {MAX_INPUT} byte limit",
            )

        name = f"sbx-{uuid.uuid4().hex[:12]}"
        encoded_code = base64.b64encode(code_bytes).decode("ascii")
        command = [
            "docker",
            "run",
            "--rm",
            "-i",
            "--name",
            name,
            "--network",
            "none",
            "--memory",
            f"{self.mem_mb}m",
            "--memory-swap",
            f"{self.mem_mb}m",
            "--cpus",
            str(self.cpus),
            "--pids-limit",
            str(self.pids),
            "--read-only",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,nodev,size=16m,mode=1777",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            "65532:65532",
            "--ulimit",
            "nofile=32:32",
            "--ulimit",
            "fsize=1048576:1048576",
            "--env",
            "PYTHONIOENCODING=UTF-8",
            "--env",
            f"CODE_B64={encoded_code}",
            self.image,
            "sh",
            "-c",
            "python -c 'import base64, os; open(\"/tmp/main.py\", \"wb\").write(base64.b64decode(os.environ[\"CODE_B64\"]))' && exec python -I -B /tmp/main.py",
        ]

        exceeded = threading.Event()
        stdout_buf, stderr_buf = bytearray(), bytearray()
        t0 = time.monotonic()
        try:
            proc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except (FileNotFoundError, OSError) as exc:
            return RunResult(
                status=RunStatus.SANDBOX_ERROR,
                stderr=str(exc),
                error_type=type(exc).__name__,
            )

        threads = [
            threading.Thread(
                target=self._drain,
                args=(proc.stdout, stdout_buf, exceeded),
                daemon=True,
            ),
            threading.Thread(
                target=self._drain,
                args=(proc.stderr, stderr_buf, exceeded),
                daemon=True,
            ),
            threading.Thread(
                target=self._feed,
                args=(proc.stdin, stdin_bytes),
                daemon=True,
            ),
        ]
        for thread in threads:
            thread.start()

        timed_out = False
        deadline = t0 + max(float(timeout_s), 0.1) + 2
        while proc.poll() is None:
            if exceeded.is_set():
                self._kill(name)
                break
            if time.monotonic() > deadline:
                timed_out = True
                self._kill(name)
                break
            time.sleep(0.02)

        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        for thread in threads:
            thread.join(timeout=5)

        ms = int((time.monotonic() - t0) * 1000)
        return classify_docker(
            proc.returncode,
            stdout_buf.decode("utf-8", "replace"),
            stderr_buf.decode("utf-8", "replace"),
            ms,
            timed_out,
            output_exceeded=exceeded.is_set(),
        )

    def run_with_precheck(self, code: str, stdin: str = "", timeout_s: float = 5) -> RunResult:
        result = precheck(code)
        return result if result is not None else self.run(code, stdin, timeout_s)

    def run_code(self, code: str, stdin: str = "", timeout_s: float = 5) -> bool:
        return self.run_with_precheck(code, stdin, timeout_s).ok
