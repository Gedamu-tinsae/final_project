"""Shared run tracking helpers for Group1/2/4 workflows."""

from __future__ import annotations

import csv
import json
import os
import re
import resource
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast


def _utc_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def _try_gpu_sample() -> dict[str, float] | None:
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
    parts = [p.strip() for p in out.splitlines()[0].split(",")]
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


_TPU_JAX_INIT = False
_TPU_AVAILABLE = False
_TPU_DEVICE = None


def _try_tpu_sample() -> dict[str, float] | None:
    global _TPU_JAX_INIT, _TPU_AVAILABLE, _TPU_DEVICE
    try:
        import jax  # type: ignore
    except Exception:
        return None

    if not _TPU_JAX_INIT:
        _TPU_JAX_INIT = True
        try:
            _TPU_AVAILABLE = jax.default_backend() == "tpu"
            if _TPU_AVAILABLE:
                devs = jax.devices()
                _TPU_DEVICE = devs[0] if devs else None
        except Exception:
            _TPU_AVAILABLE = False
            _TPU_DEVICE = None

    if not _TPU_AVAILABLE or _TPU_DEVICE is None:
        return None

    try:
        ms = _TPU_DEVICE.memory_stats()
        if not isinstance(ms, dict):
            return None
        bytes_in_use = float(ms.get("bytes_in_use", 0.0))
        bytes_limit = float(ms.get("bytes_limit", 0.0))
        peak_bytes_in_use = float(ms.get("peak_bytes_in_use", 0.0))
        out = {
            "tpu_bytes_in_use": bytes_in_use,
            "tpu_peak_bytes_in_use": peak_bytes_in_use,
            "tpu_bytes_limit": bytes_limit,
            "tpu_mem_used_mb": bytes_in_use / (1024.0 * 1024.0),
            "tpu_mem_peak_mb": peak_bytes_in_use / (1024.0 * 1024.0),
            "tpu_mem_limit_mb": bytes_limit / (1024.0 * 1024.0),
        }
        if bytes_limit > 0:
            out["tpu_mem_used_pct"] = 100.0 * (bytes_in_use / bytes_limit)
            out["tpu_mem_peak_pct"] = 100.0 * (peak_bytes_in_use / bytes_limit)
        return out
    except Exception:
        return None


def _safe_rss_kb_max() -> float:
    try:
        getrusage = cast(Any, getattr(resource, "getrusage", None))
        rusage_self = cast(Any, getattr(resource, "RUSAGE_SELF", None))
        if getrusage is None or rusage_self is None:
            return 0.0
        return float(getrusage(rusage_self).ru_maxrss)
    except Exception:
        return 0.0


@dataclass
class StageRecord:
    stage: str
    status: str
    wall_time_sec: float
    extra: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "stage": self.stage,
            "status": self.status,
            "wall_time_sec": self.wall_time_sec,
        }
        payload.update(self.extra)
        return payload


class ResourceSampler:
    def __init__(self, output_csv: Path, interval_sec: float = 2.0) -> None:
        self.output_csv = output_csv
        self.interval_sec = interval_sec
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._rows: list[dict[str, Any]] = []
        self._start = time.perf_counter()

    def start(self) -> None:
        def loop() -> None:
            while not self._stop.is_set():
                elapsed = time.perf_counter() - self._start
                rss_kb = _safe_rss_kb_max()
                row: dict[str, Any] = {
                    "t_sec": elapsed,
                    "rss_kb_max": float(rss_kb),
                }
                gpu = _try_gpu_sample()
                if gpu is not None:
                    row.update(gpu)
                tpu = _try_tpu_sample()
                if tpu is not None:
                    row.update(tpu)
                self._rows.append(row)
                self._stop.wait(self.interval_sec)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        if not self._rows:
            self.output_csv.write_text("t_sec,rss_kb_max\n", encoding="utf-8")
            return
        keys = sorted({k for r in self._rows for k in r.keys()})
        with self.output_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in self._rows:
                w.writerow(r)

    def summary(self) -> dict[str, Any]:
        if not self._rows:
            return {}
        rss_vals = [float(r.get("rss_kb_max", 0.0)) for r in self._rows]
        out: dict[str, Any] = {
            "samples": len(self._rows),
            "rss_kb_max": max(rss_vals),
        }
        if "gpu_util_pct" in self._rows[0]:
            util = [float(r.get("gpu_util_pct", 0.0)) for r in self._rows]
            mem = [float(r.get("gpu_mem_used_mb", 0.0)) for r in self._rows]
            pwr = [float(r.get("gpu_power_w", 0.0)) for r in self._rows]
            out.update(
                {
                    "gpu_util_avg_pct": sum(util) / max(1, len(util)),
                    "gpu_util_max_pct": max(util),
                    "gpu_mem_used_max_mb": max(mem),
                    "gpu_power_avg_w": sum(pwr) / max(1, len(pwr)),
                }
            )
        if "tpu_mem_used_mb" in self._rows[0]:
            tmem = [float(r.get("tpu_mem_used_mb", 0.0)) for r in self._rows]
            tpeak = [float(r.get("tpu_mem_peak_mb", 0.0)) for r in self._rows]
            tlim = [float(r.get("tpu_mem_limit_mb", 0.0)) for r in self._rows]
            tpct = [float(r.get("tpu_mem_used_pct", 0.0)) for r in self._rows]
            out.update(
                {
                    "tpu_mem_used_avg_mb": sum(tmem) / max(1, len(tmem)),
                    "tpu_mem_used_max_mb": max(tmem),
                    "tpu_mem_peak_max_mb": max(tpeak),
                    "tpu_mem_limit_mb": max(tlim) if tlim else 0.0,
                    "tpu_mem_used_avg_pct": sum(tpct) / max(1, len(tpct)),
                    "tpu_mem_used_max_pct": max(tpct) if tpct else 0.0,
                }
            )
        return out

    @property
    def rows(self) -> list[dict[str, Any]]:
        return list(self._rows)


class RunTracker:
    def __init__(
        self,
        *,
        group: str,
        output_root: Path,
        run_name: str,
        config: dict[str, Any],
    ) -> None:
        self.group = group
        self.output_root = output_root
        self.run_id = f"{_utc_ts()}_{run_name}"
        self.run_dir = output_root / group / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.stage_records: list[StageRecord] = []
        self.artifacts: list[dict[str, Any]] = []
        self._start = time.perf_counter()
        self.resource_sampler = ResourceSampler(self.run_dir / "resource_usage.csv", interval_sec=2.0)
        self.resource_sampler.start()
        self.stdout_log_path = self.run_dir / "stdout.log"
        self.write_json("run_config.json", config)
        self.write_json(
            "system_info.json",
            {
                "pid": os.getpid(),
                "python": os.environ.get("PYTHON", ""),
                "cwd": os.getcwd(),
                "jax_backend": _safe_jax_backend(),
                "jax_num_devices": _safe_jax_num_devices(),
            },
        )
        self.write_json("stage_meta.json", {"group": group, "run_id": self.run_id, "stages": []})
        self.write_json("artifacts_manifest.json", {"artifacts": []})
        self.write_json("timing.json", {"run_id": self.run_id, "stages": []})

    @contextmanager
    def stage(self, stage: str):
        t0 = time.perf_counter()
        extra: dict[str, Any] = {}
        status = "ok"
        try:
            yield extra
        except Exception:
            status = "error"
            raise
        finally:
            dt = max(1e-9, time.perf_counter() - t0)
            rec = StageRecord(stage=stage, status=status, wall_time_sec=dt, extra=extra)
            self.stage_records.append(rec)
            self.append_jsonl("stage_times.jsonl", rec.as_dict())
            stage_dir = self._stage_dir_for(stage)
            self._write_json_abs(stage_dir / "stage_meta.json", rec.as_dict())
            self._write_json_abs(
                stage_dir / "timing.json",
                {"stage": stage, "wall_time_sec": dt, "status": status},
            )

    def append_jsonl(self, rel_path: str, payload: dict[str, Any]) -> None:
        p = self.run_dir / rel_path
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.register_artifact(p, kind="jsonl")

    def write_json(self, rel_path: str, payload: dict[str, Any]) -> None:
        p = self.run_dir / rel_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self.register_artifact(p, kind="json")

    def write_csv(self, rel_path: str, rows: list[dict[str, Any]]) -> None:
        p = self.run_dir / rel_path
        p.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            p.write_text("", encoding="utf-8")
            self.register_artifact(p, kind="csv")
            return
        keys = sorted({k for r in rows for k in r.keys()})
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        self.register_artifact(p, kind="csv")

    def register_artifact(self, path: Path, *, kind: str, stage: str | None = None) -> None:
        rec = {
            "path": str(path),
            "kind": kind,
            "stage": stage or self._infer_stage_from_path(path),
            "exists": path.exists(),
        }
        self.artifacts.append(rec)

    def start_stdio_capture(self) -> "_StdIOTee":
        tee = _StdIOTee(self.stdout_log_path)
        tee.start()
        self.register_artifact(self.stdout_log_path, kind="log")
        return tee

    def _infer_stage_from_path(self, path: Path) -> str | None:
        rel = str(path).replace("\\", "/")
        m = re.search(r"(stage\d+)", rel)
        if m:
            return m.group(1)
        if "/peft/" in rel:
            return "peft"
        return None

    def _stage_dir_for(self, stage_name: str) -> Path:
        m = re.search(r"(stage\d+)", stage_name)
        stage_bucket = m.group(1) if m else "stage_misc"
        out = self.output_root / self.group / stage_bucket / self.run_id
        out.mkdir(parents=True, exist_ok=True)
        return out

    def _write_json_abs(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self.register_artifact(path, kind="json")

    def finalize(self) -> dict[str, Any]:
        self.resource_sampler.stop()
        wall = max(1e-9, time.perf_counter() - self._start)
        summary = {
            "group": self.group,
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "wall_time_sec": wall,
            "stages": [r.as_dict() for r in self.stage_records],
            "resources": self.resource_sampler.summary(),
        }
        self.write_json("metrics_summary.json", summary)
        self.write_json("stage_meta.json", {"group": self.group, "run_id": self.run_id, "stages": [r.as_dict() for r in self.stage_records]})
        self.write_json("timing.json", {"run_id": self.run_id, "wall_time_sec": wall, "stages": [r.as_dict() for r in self.stage_records]})
        self.write_json("artifacts_manifest.json", {"artifacts": self.artifacts})
        _write_plot_data_stage_timing(self.run_dir, self.stage_records)
        _write_plot_data_resource(self.run_dir, self.resource_sampler.rows)
        _try_make_stage_timing_plot(self.run_dir, self.stage_records)
        _try_make_memory_plot(self.run_dir, self.resource_sampler.rows)
        _try_make_io_counts_plot(self.run_dir, self.stage_records)
        _try_make_overview_dashboard(self.run_dir)
        self._materialize_stage_contract_dirs()
        return summary

    def _materialize_stage_contract_dirs(self) -> None:
        # Ensure stage contract files exist under outputs/<group>/stage<k>/<run_id>/...
        for rec in self.stage_records:
            stage_dir = self._stage_dir_for(rec.stage)
            self._write_json_abs(stage_dir / "run_config.json", {"group": self.group, "run_id": self.run_id})
            self._write_json_abs(stage_dir / "stage_meta.json", rec.as_dict())
            self._write_json_abs(stage_dir / "timing.json", {"stage": rec.stage, "wall_time_sec": rec.wall_time_sec})
            self._write_json_abs(stage_dir / "artifacts_manifest.json", {"artifacts": [a for a in self.artifacts if a.get("stage") in (None, rec.stage, self._infer_stage_from_path(Path(a["path"])))]})
            # Copy resource usage and stdout snapshots for stage-level contract
            if (self.run_dir / "resource_usage.csv").exists():
                (stage_dir / "resource_usage.csv").write_text((self.run_dir / "resource_usage.csv").read_text(encoding="utf-8"), encoding="utf-8")
                self.register_artifact(stage_dir / "resource_usage.csv", kind="csv", stage=rec.stage)
            if self.stdout_log_path.exists():
                (stage_dir / "stdout.log").write_text(self.stdout_log_path.read_text(encoding="utf-8"), encoding="utf-8")
                self.register_artifact(stage_dir / "stdout.log", kind="log", stage=rec.stage)


class _StdIOTee:
    def __init__(self, out_path: Path) -> None:
        self.out_path = out_path
        self._file = None
        self._orig_out = None
        self._orig_err = None

    def start(self) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.out_path.open("a", encoding="utf-8")
        self._orig_out = sys.stdout
        self._orig_err = sys.stderr
        sys.stdout = _TeeWriter(self._orig_out, self._file)
        sys.stderr = _TeeWriter(self._orig_err, self._file)

    def stop(self) -> None:
        if self._orig_out is not None:
            sys.stdout = self._orig_out
        if self._orig_err is not None:
            sys.stderr = self._orig_err
        if self._file is not None:
            self._file.flush()
            self._file.close()


class _TeeWriter:
    def __init__(self, a, b) -> None:
        self.a = a
        self.b = b

    def write(self, s):
        self.a.write(s)
        self.b.write(s)
        return len(s)

    def flush(self):
        self.a.flush()
        self.b.flush()


def _safe_jax_backend() -> str:
    try:
        import jax  # type: ignore

        return str(jax.default_backend())
    except Exception:
        return "unknown"


def _safe_jax_num_devices() -> int:
    try:
        import jax  # type: ignore

        return int(len(jax.devices()))
    except Exception:
        return 0


def _try_make_stage_timing_plot(run_dir: Path, stage_records: list[StageRecord]) -> None:
    if not stage_records:
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return

    labels = [r.stage for r in stage_records]
    vals = [r.wall_time_sec for r in stage_records]
    plt.figure(figsize=(10, 4))
    plt.bar(labels, vals)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("seconds")
    plt.title("Stage Timing")
    plt.tight_layout()
    out = run_dir / "fig_stage_timing.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    # alias for slide naming
    plt.savefig(run_dir / "fig_throughput_steps_per_sec.png", dpi=150)
    plt.close()


def _try_make_memory_plot(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return
    xs = [float(r.get("t_sec", 0.0)) for r in rows]
    ys_mb = [float(r.get("rss_kb_max", 0.0)) / 1024.0 for r in rows]
    if not xs or not ys_mb:
        return
    plt.figure(figsize=(10, 4))
    plt.plot(xs, ys_mb)
    plt.xlabel("seconds")
    plt.ylabel("RSS max (MB)")
    plt.title("Process Memory Over Time")
    plt.tight_layout()
    out = run_dir / "fig_stage_memory.png"
    plt.savefig(out, dpi=150)
    plt.savefig(run_dir / "fig_memory_usage.png", dpi=150)
    plt.close()


def _try_make_io_counts_plot(run_dir: Path, stage_records: list[StageRecord]) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return
    labels: list[str] = []
    vals: list[float] = []
    for rec in stage_records:
        count_val = None
        for k, v in rec.extra.items():
            if isinstance(v, (int, float)) and (
                "rows" in k.lower() or "files" in k.lower() or "count" in k.lower() or "jobs" in k.lower()
            ):
                count_val = float(v)
                break
        if count_val is not None:
            labels.append(rec.stage)
            vals.append(count_val)
    if not labels:
        return
    plt.figure(figsize=(10, 4))
    plt.bar(labels, vals)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("count")
    plt.title("Stage Count Metrics")
    plt.tight_layout()
    out = run_dir / "fig_io_counts.png"
    plt.savefig(out, dpi=150)
    plt.close()


def _write_plot_data_stage_timing(run_dir: Path, stage_records: list[StageRecord]) -> None:
    p = run_dir / "plots_data" / "stage_timing.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["stage", "status", "wall_time_sec"])
        w.writeheader()
        for r in stage_records:
            w.writerow({"stage": r.stage, "status": r.status, "wall_time_sec": r.wall_time_sec})


def _write_plot_data_resource(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    p = run_dir / "plots_data" / "resource_usage.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("", encoding="utf-8")
        return
    keys = sorted({k for r in rows for k in r})
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _try_make_overview_dashboard(run_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import matplotlib.image as mpimg  # type: ignore
    except Exception:
        return
    sources = [
        run_dir / "fig_stage_timing.png",
        run_dir / "fig_stage_memory.png",
        run_dir / "fig_io_counts.png",
        run_dir / "fig_memory_usage.png",
    ]
    existing = [p for p in sources if p.exists()]
    if not existing:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for i in range(4):
        ax = axes[i]
        if i < len(existing):
            img = mpimg.imread(str(existing[i]))
            ax.imshow(img)
            ax.set_title(existing[i].name)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(run_dir / "fig_overview_dashboard.png", dpi=150)
    plt.close()
