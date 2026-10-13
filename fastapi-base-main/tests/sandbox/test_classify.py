from app.sandbox.classify import classify_docker
from app.sandbox.base import RunStatus

TB = 'Traceback (most recent call last):\n  File "/sandbox/src/main.py", line 3, in <module>\n    1/0\nZeroDivisionError: division by zero\n'

def test_ok():
    assert classify_docker(0, "hi", "", 10, False).status == RunStatus.OK

def test_timeout():
    assert classify_docker(None, "", "", 3000, True).status == RunStatus.TIMEOUT

def test_oom():
    assert classify_docker(137, "", "", 10, False).status == RunStatus.MEMORY_LIMIT
def test_timeout_from_container():
    assert classify_docker(124, "", "", 3000, False).status == RunStatus.TIMEOUT
def test_runtime_parses_type_and_line():
    r = classify_docker(1, "", TB, 10, False)
    assert r.status == RunStatus.RUNTIME_ERROR
    assert r.error_type == "ZeroDivisionError"
    assert r.error_line == 3
def test_output_limit():
    assert classify_docker(137, "a", "", 10, False, output_exceeded=True).status == RunStatus.OUTPUT_LIMIT

def test_timeout_from_container():
    assert classify_docker(124, "", "", 3000, False).status == RunStatus.TIMEOUT
def test_docker_down_is_infra_not_student():
    err = "failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine"
    assert classify_docker(1, "", err, 10, False).status == RunStatus.SANDBOX_ERROR