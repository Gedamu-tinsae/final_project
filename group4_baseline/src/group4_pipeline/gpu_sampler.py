"""GPU sampling utilities for Group4 training metrics."""

from __future__ import annotations

import statistics
import subprocess
import threading
from typing import Any


def query_gpu_snapshot() -> dict[str, float] | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None

    if not out:
        return None

    first = out.splitlines()[0]
    parts = [p.strip() for p in first.split(",")]
    if len(parts) < 4:
        return None

    try:
        return {
            "gpu_util_pct": float(parts[0]),
            "gpu_mem_used_mb": float(parts[1]),
            "gpu_mem_total_mb": float(parts[2]),
            "gpu_power_w": float(parts[3]),
        }
    except Exception:
        return None


class GPUSampler:
    def __init__(self, interval_sec: float = 2.0) -> None:
        self.interval_sec = interval_sec
        self._samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        def loop() -> None:
            while not self._stop.is_set():
                s = query_gpu_snapshot()
                if s is not None:
                    self._samples.append(s)
                self._stop.wait(self.interval_sec)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def summary(self) -> dict[str, Any] | None:
        if not self._samples:
            return None
        util = [s["gpu_util_pct"] for s in self._samples]
        mem = [s["gpu_mem_used_mb"] for s in self._samples]
        pwr = [s["gpu_power_w"] for s in self._samples]
        return {
            "num_samples": len(self._samples),
            "gpu_util_avg_pct": statistics.fmean(util),
            "gpu_util_max_pct": max(util),
            "gpu_mem_used_avg_mb": statistics.fmean(mem),
            "gpu_mem_used_max_mb": max(mem),
            "gpu_power_avg_w": statistics.fmean(pwr),
            "gpu_power_max_w": max(pwr),
            "gpu_mem_total_mb": self._samples[-1]["gpu_mem_total_mb"],
        }
