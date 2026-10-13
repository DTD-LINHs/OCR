import pytest

from app.sandbox.base import RunStatus
from app.sandbox.docker_runner import DockerRunner


CASES = (
    pytest.param("ok", "print('hi')", RunStatus.OK, id="ok"),
    pytest.param("timeout", "while True: pass", RunStatus.TIMEOUT, id="timeout"),
    pytest.param("memory", "x = [0] * 10**10", RunStatus.MEMORY_LIMIT, id="memory"),
    pytest.param(
        "output_spam",
        "while True: print('a' * 1000)",
        RunStatus.OUTPUT_LIMIT,
        id="output-spam",
    ),
    pytest.param("runtime", "1 / 0", RunStatus.RUNTIME_ERROR, id="runtime"),
    pytest.param("name_error", "pirnt('x')", RunStatus.RUNTIME_ERROR, id="name-error"),
    pytest.param("input_eof", "x = input()", RunStatus.RUNTIME_ERROR, id="input-eof"),
    pytest.param("env_leak", "import os; print(dict(os.environ))", RunStatus.OK, id="env-leak"),
    pytest.param(
        "read_file",
        "print(open('/etc/passwd').read()[:50])",
        RunStatus.OK,
        id="read-file",
    ),
    pytest.param(
        "sigterm_trap",
        "import signal; signal.signal(signal.SIGTERM, lambda *a: None)\nwhile True: pass",
        RunStatus.TIMEOUT,
        id="sigterm-trap",
    ),
    pytest.param(
        "write_file",
        "open('/sandbox/src/main.py', 'w').write('x')",
        RunStatus.RUNTIME_ERROR,
        id="write-file",
    ),
    pytest.param(
        "network",
        "import urllib.request; urllib.request.urlopen('http://example.com', timeout=2)",
        RunStatus.RUNTIME_ERROR,
        id="network",
    ),
    pytest.param(
        "fork",
        "import os\nfor _ in range(500): os.fork()",
        RunStatus.RUNTIME_ERROR,
        id="fork",
    ),
)


@pytest.mark.docker
@pytest.mark.parametrize(("name", "code", "expected_status"), CASES)
def test_docker_runner_cases(name, code, expected_status):
    result = DockerRunner().run_with_precheck(code, timeout_s=3)

    assert result.status == expected_status, (
        f"{name}: {result.status.value}, stderr={result.stderr!r}"
    )


@pytest.mark.docker
def test_run_with_precheck_does_not_start_docker(monkeypatch):
    runner = DockerRunner()

    def fail_if_called(*args, **kwargs):
        pytest.fail("Docker should not run for invalid Python syntax")

    monkeypatch.setattr(runner, "run", fail_if_called)

    result = runner.run_with_precheck("if True print('invalid')")

    assert result.status == RunStatus.SYNTAX_ERROR
    assert result.error_type == "SyntaxError"
