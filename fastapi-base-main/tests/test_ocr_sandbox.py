import pytest

from app.services.srv_sandbox import (
    LANGUAGES,
    DockerSandbox,
    SandboxUnavailable,
    _java_filename,
    canonical_language,
    parse_errors,
)


@pytest.mark.parametrize(
    "language, output, expected",
    [
        (
            "python",
            '  File "/work/main.py", line 2\n    if x\n        ^\nSyntaxError: expected \':\'\n',
            [(2, "SyntaxError: expected ':'")],
        ),
        (
            "c",
            "/work/main.c: In function 'main':\n/work/main.c:3:13: error: expected ';' before 'return'\n",
            [(3, "expected ';' before 'return'")],
        ),
        (
            "cpp",
            "/work/main.cpp:5:1: fatal error: conio.h: No such file or directory\n",
            [(5, "conio.h: No such file or directory")],
        ),
        (
            "java",
            "/work/Main.java:3: error: ';' expected\n        int x = 1\n                 ^\n1 error\n",
            [(3, "';' expected")],
        ),
        (
            "javascript",
            "/work/main.js:2\n  if (x {\n        ^\n\nSyntaxError: Unexpected token '{'\n    at wrapSafe\n",
            [(2, "SyntaxError: Unexpected token '{'")],
        ),
        ("go", "/work/main.go:4:2: expected '}', found 'EOF'\n", [(4, "expected '}', found 'EOF'")]),
        (
            "php",
            "PHP Parse error:  syntax error, unexpected end of file in /work/main.php on line 3\n"
            "\nParse error: syntax error, unexpected end of file in /work/main.php on line 3\n"
            "Errors parsing /work/main.php\n",
            [(3, "syntax error, unexpected end of file")],
        ),
        ("ruby", "/work/main.rb:3: syntax error, unexpected end-of-input\n", [(3, "syntax error, unexpected end-of-input")]),
        ("bash", "/work/main.sh: line 3: syntax error near unexpected token `fi'\n", [(3, "syntax error near unexpected token `fi'")]),
    ],
)
def test_parse_errors(language, output, expected):
    issues = parse_errors(LANGUAGES[language], output)
    assert [(i.line, i.message) for i in issues] == expected


def test_java_filename_follows_public_class():
    assert _java_filename("public class Solution {\n}") == "Solution.java"
    assert _java_filename("public final class A {}") == "A.java"
    assert _java_filename("class Hidden {}") == "Main.java"


def test_canonical_language():
    assert canonical_language("C++") == "cpp"
    assert canonical_language(" JS ") == "javascript"
    assert canonical_language(None) == "unknown"
    assert DockerSandbox().supports("Python3")
    assert not DockerSandbox().supports("cobol")


def test_rejects_oversized_code():
    sandbox = DockerSandbox(client=object())
    with pytest.raises(ValueError):
        sandbox.check("x" * 70_000, "python")


def _docker_sandbox():
    sandbox = DockerSandbox()
    try:
        sandbox.client.ping()
    except Exception:
        pytest.skip("Docker daemon not available")
    return sandbox


@pytest.mark.docker
def test_docker_python_syntax_check():
    sandbox = _docker_sandbox()
    assert sandbox.check("def f(x):\n    return x\n", "python").ok
    bad = sandbox.check("def f(x)\n    return x\n", "python")
    assert not bad.ok and bad.errors[0].line == 1


def test_unreachable_docker_raises_sandbox_unavailable(monkeypatch):
    import docker

    def boom():
        raise docker.errors.DockerException("daemon down")

    monkeypatch.setattr(docker, "from_env", boom)
    with pytest.raises(SandboxUnavailable):
        DockerSandbox().check("x = 1", "python")
