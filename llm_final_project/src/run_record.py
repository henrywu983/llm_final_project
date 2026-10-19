"""Everything needed to know exactly what produced a run.

Writes into the run folder:
    config.yaml       fully resolved config (after inherit + overrides)
    run.json          command, overrides, git commit, machine, start/end time, status
    pip_freeze.txt    exact package versions
    nvidia_smi.txt    GPU state at start
    git_diff.patch    uncommitted code changes, if any
"""

from __future__ import annotations

import datetime as dt
import json
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

from .config import save_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=120).stdout
    except Exception as e:  # noqa: BLE001
        return f"<failed: {e}>"


def _now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


class RunRecord:
    def __init__(self, run_dir: Path, cfg: dict, argv: list[str], experiment_overrides: list[str]):
        self.run_dir = Path(run_dir)
        self.cfg = cfg
        self.path = self.run_dir / "run.json"
        self.data = dict(command=" ".join([sys.executable, *argv]),
                         experiment_overrides=experiment_overrides)
        self._t0 = time.time()

    def start(self) -> None:
        save_config(self.cfg, self.run_dir / "config.yaml")
        commit = _run(["git", "rev-parse", "HEAD"]).strip()
        dirty = bool(_run(["git", "status", "--porcelain"]).strip())
        if dirty:
            (self.run_dir / "git_diff.patch").write_text(_run(["git", "diff", "HEAD"]), encoding="utf-8")
        (self.run_dir / "pip_freeze.txt").write_text(_run([sys.executable, "-m", "pip", "freeze"]), encoding="utf-8")
        (self.run_dir / "nvidia_smi.txt").write_text(_run(["nvidia-smi"]), encoding="utf-8")

        gpu = {}
        try:
            import torch

            gpu = dict(torch=torch.__version__, cuda=torch.version.cuda,
                       gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                       gpu_total_mem_gb=round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
                       if torch.cuda.is_available() else None)
        except Exception as e:  # noqa: BLE001
            gpu = dict(error=str(e))

        self.data.update(
            status="running", start_time=_now(), hostname=socket.gethostname(),
            python=sys.version.split()[0], platform=platform.platform(),
            git_commit=commit or None, git_dirty=dirty, **gpu,
        )
        self._write()

    def update(self, **fields) -> None:
        self.data.update(fields)
        self._write()

    def finish(self, status: str) -> None:
        self.data.update(status=status, end_time=_now(),
                         duration_hours=round((time.time() - self._t0) / 3600, 3))
        self._write()

    def _write(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2, default=str), encoding="utf-8")


class Tee:
    """Copy everything printed to the terminal into a log file."""

    def __init__(self, stream, file):
        self.stream, self.file = stream, file

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)
        return len(data)

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def isatty(self):
        return self.stream.isatty()

    def fileno(self):
        return self.stream.fileno()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def start_terminal_log(path: Path):
    f = open(path, "a", encoding="utf-8", buffering=1)
    f.write(f"\n===== {_now()} =====\n")
    sys.stdout = Tee(sys.__stdout__, f)
    sys.stderr = Tee(sys.__stderr__, f)
    return f
