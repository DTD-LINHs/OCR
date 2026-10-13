"""Best-effort resource-limited launcher for local development.

The Docker backend is the production boundary for untrusted student code.
"""

import os
import runpy
import sys


def _set_limits(cpu_s: int, mem_mb: int, fsize_mb: int) -> None:
    try:
        import resource
    except ImportError:  # pragma: no cover - Windows
        return
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
    resource.setrlimit(resource.RLIMIT_AS, (mem_mb << 20, mem_mb << 20))
    resource.setrlimit(resource.RLIMIT_FSIZE, (fsize_mb << 20, fsize_mb << 20))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))


def main() -> None:
    if len(sys.argv) != 5:
        raise SystemExit("usage: _launcher.py CPU_SECONDS MEMORY_MB FSIZE_MB SCRIPT")
    cpu_s, mem_mb, fsize_mb = (int(value) for value in sys.argv[1:4])
    _set_limits(cpu_s, mem_mb, fsize_mb)
    script = sys.argv[4]
    if os.name != "nt":
        os.execv(sys.executable, [sys.executable, "-I", "-B", script])
    sys.argv = [script]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
