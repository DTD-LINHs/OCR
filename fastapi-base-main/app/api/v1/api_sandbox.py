"""Internal endpoint for testing the Python code sandbox."""

from typing import Any

from fastapi import APIRouter, status
from pydantic import BaseModel, Field

from app.core.config import settings
from app.sandbox.docker_runner import DockerRunner


router = APIRouter(prefix="/sandbox")


class SandboxRunRequest(BaseModel):
    """Python submission and optional stdin passed to the sandbox."""

    code: str = Field(..., min_length=1, max_length=64_000)
    stdin: str = Field(default="", max_length=64_000)
    timeout_s: float = Field(default=5, ge=0.1, le=60)


def _memory_to_mb(value: str) -> int:
    """Convert the configured Docker memory value, such as ``256m``, to MB."""

    normalized = value.strip().lower()
    if normalized.endswith("m"):
        return max(int(float(normalized[:-1])), 1)
    if normalized.endswith("g"):
        return max(int(float(normalized[:-1]) * 1024), 1)
    return max(int(float(normalized)), 1)


runner = DockerRunner(
    mem_mb=_memory_to_mb(settings.SANDBOX_MEMORY),
    cpus=settings.SANDBOX_CPUS,
    image=settings.SANDBOX_IMAGE,
)


@router.post(
    "/run",
    response_model=bool,
    status_code=status.HTTP_200_OK,
    summary="Run Python code in the isolated sandbox",
    description="Returns true only when the submitted program exits successfully.",
)
def run_sandbox(body: SandboxRunRequest) -> Any:
    return runner.run_code(
        body.code,
        stdin=body.stdin,
        timeout_s=body.timeout_s,
    )
