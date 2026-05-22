"""Capture host hardware and runtime context for benchmark runs."""

from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

import config


def _run_cmd(cmd: list[str], timeout: int = 10) -> str | None:
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def capture_hardware_snapshot(
    *,
    host_profile: str | None = None,
    embed_model: str | None = None,
    chat_model: str | None = None,
) -> dict[str, Any]:
    snap: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host_profile": host_profile or "unknown",
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "cpu_count": None,
        "ram_total_gb": None,
        "ram_available_gb": None,
        "gpu_nvidia_smi": None,
        "ollama_gpu_mode": config.OLLAMA_GPU_MODE,
        "ollama_base_url": config.OLLAMA_BASE_URL,
        "ollama_embed_model": embed_model or config.OLLAMA_EMBED_MODEL,
        "ollama_chat_model": chat_model or config.OLLAMA_CHAT_MODEL,
    }

    try:
        import psutil

        snap["cpu_count"] = psutil.cpu_count(logical=True)
        mem = psutil.virtual_memory()
        snap["ram_total_gb"] = round(mem.total / (1024**3), 2)
        snap["ram_available_gb"] = round(mem.available / (1024**3), 2)
    except ImportError:
        snap["psutil_note"] = "psutil not installed"

    snap["gpu_nvidia_smi"] = _run_cmd(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    docker_info = _run_cmd(["docker", "info"])
    snap["docker_has_nvidia"] = bool(
        docker_info and "nvidia" in docker_info.lower()
    )

    return snap
