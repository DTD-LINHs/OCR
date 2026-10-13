"""Syntax-check transcribed code inside throw-away Docker containers.

The code comes from a photo, so it is untrusted: every check runs in a fresh
container with no network, a read-only root FS, a small tmpfs, no capabilities,
an unprivileged user and CPU/memory/pid limits. Nothing is executed, only parsed
or compiled, since the goal is to validate the transcription, not run it.
"""

import base64
import logging
import re
import time
from dataclasses import dataclass
from typing import Callable, Optional

from app.schemas.sche_ocr_pipeline import SandboxResult, SyntaxIssue

logger = logging.getLogger(__name__)

_FLAGS = re.MULTILINE | re.DOTALL
_MAX_OUTPUT_CHARS = 8000


class SandboxUnavailable(Exception):
    """Docker itself failed (daemon down, image missing...), not the checked code."""


@dataclass(frozen=True)
class LanguageSpec:
    image: str
    filename: str
    command: str
    error_pattern: str
    filename_fn: Optional[Callable[[str], str]] = None


def _java_filename(code: str) -> str:
    # javac requires a public class to live in a file of the same name.
    match = re.search(
        r"public\s+(?:(?:final|abstract|sealed)\s+)*(?:class|interface|enum|record)\s+(\w+)",
        code,
    )
    return f"{match.group(1)}.java" if match else "Main.java"


LANGUAGES: dict[str, LanguageSpec] = {
    "python": LanguageSpec(
        image="python:3.12-alpine",
        filename="main.py",
        command="python -c 'import sys; compile(open(sys.argv[1], encoding=\"utf-8\").read(), sys.argv[1], \"exec\")' \"$F\"",
        error_pattern=r'File "[^"\n]*", line (?P<line>\d+).*?^(?P<msg>\w*(?:Error|Exception)[^\n]*)$',
    ),
    "c": LanguageSpec(
        image="gcc:14",
        filename="main.c",
        command='gcc -fsyntax-only -x c "$F"',
        error_pattern=r"main\.c:(?P<line>\d+):(?:\d+:)? (?:fatal )?error: (?P<msg>[^\n]*)",
    ),
    "cpp": LanguageSpec(
        image="gcc:14",
        filename="main.cpp",
        command='g++ -fsyntax-only -std=c++17 -x c++ "$F"',
        error_pattern=r"main\.cpp:(?P<line>\d+):(?:\d+:)? (?:fatal )?error: (?P<msg>[^\n]*)",
    ),
    "java": LanguageSpec(
        image="eclipse-temurin:21-jdk-alpine",
        filename="Main.java",
        command='javac -J-Xmx128m -d /tmp/out "$F"',
        error_pattern=r"\.java:(?P<line>\d+): error: (?P<msg>[^\n]*)",
        filename_fn=_java_filename,
    ),
    "javascript": LanguageSpec(
        image="node:22-alpine",
        filename="main.js",
        command='node --check "$F"',
        error_pattern=r"main\.js:(?P<line>\d+)\n.*?^(?P<msg>\w*Error: [^\n]*)$",
    ),
    "go": LanguageSpec(
        image="golang:1.23-alpine",
        filename="main.go",
        command='gofmt -e "$F" > /dev/null',
        error_pattern=r"main\.go:(?P<line>\d+):\d+: (?P<msg>[^\n]*)",
    ),
    "php": LanguageSpec(
        image="php:8.3-cli-alpine",
        filename="main.php",
        command='php -l "$F"',
        error_pattern=r"error:\s*(?P<msg>[^\n]*?) in [^\n]* on line (?P<line>\d+)",
    ),
    "ruby": LanguageSpec(
        image="ruby:3.3-alpine",
        filename="main.rb",
        command='ruby -c "$F"',
        error_pattern=r"main\.rb:(?P<line>\d+): (?P<msg>[^\n]*)",
    ),
    "bash": LanguageSpec(
        image="bash:5.2",
        filename="main.sh",
        command='bash -n "$F"',
        error_pattern=r"main\.sh: line (?P<line>\d+): (?P<msg>[^\n]*)",
    ),
}

LANGUAGE_ALIASES = {
    "py": "python", "python3": "python",
    "c++": "cpp", "cxx": "cpp",
    "js": "javascript", "node": "javascript", "nodejs": "javascript",
    "golang": "go",
    "sh": "bash", "shell": "bash",
}


def canonical_language(language: Optional[str]) -> str:
    key = (language or "unknown").strip().lower()
    return LANGUAGE_ALIASES.get(key, key)


def parse_errors(spec: LanguageSpec, output: str) -> list[SyntaxIssue]:
    issues: list[SyntaxIssue] = []
    seen = set()
    for match in re.finditer(spec.error_pattern, output, _FLAGS):
        key = (int(match.group("line")), match.group("msg").strip())
        if key not in seen:
            seen.add(key)
            issues.append(SyntaxIssue(line=key[0], message=key[1]))
    return issues


@dataclass(frozen=True)
class SandboxConfig:
    timeout_s: int = 15
    memory: str = "256m"
    cpus: float = 0.5
    pids_limit: int = 64
    runtime: Optional[str] = None
    max_code_bytes: int = 64_000


class DockerSandbox:
    def __init__(self, config: SandboxConfig = SandboxConfig(), client=None):
        self.config = config
        self._client = client

    @property
    def client(self):
        if self._client is None:
            import docker

            try:
                self._client = docker.from_env()
            except docker.errors.DockerException as e:
                raise SandboxUnavailable(f"Cannot connect to Docker: {e}") from e
        return self._client

    def supports(self, language: str) -> bool:
        return canonical_language(language) in LANGUAGES

    def pull_images(self) -> None:
        for image in sorted({spec.image for spec in LANGUAGES.values()}):
            logger.info("Pulling sandbox image %s", image)
            self.client.images.pull(image)

    def check(self, code: str, language: str) -> SandboxResult:
        import docker
        import requests

        language = canonical_language(language)
        spec = LANGUAGES[language]
        encoded = code.encode("utf-8")
        if len(encoded) > self.config.max_code_bytes:
            raise ValueError(f"Code is larger than {self.config.max_code_bytes} bytes")

        filename = spec.filename_fn(code) if spec.filename_fn else spec.filename
        # The code travels as an env var and is decoded inside the container,
        # so nothing from the host file system is mounted.
        script = f'printf %s "$CODE_B64" | base64 -d > "$F" && {spec.command}'
        started = time.monotonic()
        try:
            container = self.client.containers.run(
                image=spec.image,
                command=["sh", "-c", script],
                environment={
                    "CODE_B64": base64.b64encode(encoded).decode("ascii"),
                    "F": f"/work/{filename}",
                },
                working_dir="/work",
                detach=True,
                network_disabled=True,
                read_only=True,
                tmpfs={"/work": "rw,size=16m,mode=1777", "/tmp": "rw,size=64m,mode=1777"},
                user="65534:65534",
                cap_drop=["ALL"],
                security_opt=["no-new-privileges"],
                mem_limit=self.config.memory,
                memswap_limit=self.config.memory,
                nano_cpus=int(self.config.cpus * 1e9),
                pids_limit=self.config.pids_limit,
                runtime=self.config.runtime,
                labels={"app": "ocr-sandbox"},
            )
        except docker.errors.DockerException as e:
            raise SandboxUnavailable(f"Cannot start sandbox ({spec.image}): {e}") from e

        exit_code, timed_out = None, False
        try:
            try:
                exit_code = container.wait(timeout=self.config.timeout_s)["StatusCode"]
            except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError):
                timed_out = True
                container.kill()
            output = container.logs(stdout=True, stderr=True).decode("utf-8", "replace")
        finally:
            try:
                container.remove(force=True)
            except docker.errors.DockerException:
                logger.warning("Could not remove sandbox container %s", container.id)

        output = output[-_MAX_OUTPUT_CHARS:]
        ok = exit_code == 0 and not timed_out
        errors = [] if ok else parse_errors(spec, output)
        if not ok and not errors:
            message = "Timed out" if timed_out else (output.strip().splitlines() or ["Unknown error"])[-1]
            errors = [SyntaxIssue(line=None, message=message)]

        return SandboxResult(
            language=language,
            image=spec.image,
            command=spec.command.replace('"$F"', filename),
            exit_code=exit_code,
            ok=ok,
            timed_out=timed_out,
            errors=errors,
            output=output,
            duration_ms=int((time.monotonic() - started) * 1000),
        )


if __name__ == "__main__":
    # python -m app.services.srv_sandbox  -> pre-pull every sandbox image
    logging.basicConfig(level=logging.INFO)
    DockerSandbox().pull_images()
