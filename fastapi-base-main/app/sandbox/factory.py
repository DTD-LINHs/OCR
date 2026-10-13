from .base import SandboxRunner
from .docker_runner import DockerRunner
from .subprocess_runner import SubprocessRunner


def build_runner(
    backend: str = "docker",
    *,
    image: str = "sandbox-python:latest",
    memory_mb: int = 128,
    cpus: float = 0.5,
    pids: int = 64,
) -> SandboxRunner:
    backend = backend.lower().strip()
    if backend == "docker":
        return DockerRunner(memory_mb, cpus, pids, image=image)
    if backend == "subprocess":
        return SubprocessRunner(mem_mb=memory_mb)
    raise ValueError(f"Unsupported sandbox backend: {backend}")
