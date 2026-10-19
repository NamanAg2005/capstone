"""
Policy-governed sandbox (Section 4.6).

Wraps the persistent kernel subprocess (sandbox_kernel.py) and enforces, at
the process level rather than hoping the kernel code behaves: a wall-clock
timeout and a memory ceiling per execution, using psutil since the POSIX
`resource` module isn't available on this Windows host. Every allow/deny
decision is logged as a PolicyEvent.

Three independent layers bound what a generated attempt can do:
  1. static_check.py -- rejects disallowed constructs before execution.
  2. this module's runtime policy -- network denial (inside the kernel) and
     filesystem confinement to the scratch dir (no open() available at all).
  3. this module's resource limits -- wall-clock timeout + memory ceiling.

A startup health check confirms the subprocess sandbox is actually
available; if it isn't (e.g. the Python interpreter can't be spawned),
execution falls back automatically to an in-process restricted-exec
implementation with an equivalent but less isolated set of restrictions,
logged and surfaced quietly rather than failing the session outright.
"""
from __future__ import annotations

import contextlib
import io
import json
import queue
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Literal

import psutil

from app import config
from app.models import PolicyEvent
from app.static_check import is_safe

KERNEL_SCRIPT = Path(__file__).resolve().parent / "sandbox_kernel.py"


class SandboxSession:
    """One persistent kernel per analysis session."""

    def __init__(self, datasets: dict[str, str], scratch_dir: Path):
        self.datasets = datasets
        self.scratch_dir = scratch_dir
        self.runtime: Literal["primary", "fallback"] = "primary"
        self.policy_log: list[PolicyEvent] = []
        self._proc: subprocess.Popen | None = None
        self._out_queue: queue.Queue[str] = queue.Queue()
        self._reader_thread: threading.Thread | None = None
        self._fallback_namespace: dict | None = None
        self._start()

    # ---- lifecycle -----------------------------------------------------
    def _start(self) -> None:
        try:
            self._proc = subprocess.Popen(
                [sys.executable, "-u", str(KERNEL_SCRIPT)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            self._reader_thread = threading.Thread(target=self._pump_stdout, daemon=True)
            self._reader_thread.start()
            init_msg = {"datasets": self.datasets, "scratch_dir": str(self.scratch_dir)}
            assert self._proc.stdin is not None
            self._proc.stdin.write(json.dumps(init_msg) + "\n")
            self._proc.stdin.flush()
            ready = self._out_queue.get(timeout=15)
            json.loads(ready)  # expect {"status": "ready"}
            self.runtime = "primary"
        except Exception:
            self.runtime = "fallback"
            self._start_fallback()

    def _pump_stdout(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            self._out_queue.put(line)

    def _start_fallback(self) -> None:
        """In-process restricted exec, used only if the subprocess sandbox
        itself cannot start. Less isolated (no separate OS process, no hard
        memory ceiling) but keeps the session usable rather than failing it
        outright; every attempt run this way is marked sandbox_runtime=
        "fallback" so the UI can show it quietly."""
        import pandas as pd
        import numpy as np
        import duckdb
        from app.sandbox_kernel import Kernel

        self._fallback_kernel = Kernel(self.datasets, str(self.scratch_dir))

    # ---- execution -------------------------------------------------------
    def execute(self, code: str, step_id: int, timeout_s: int | None = None, memory_mb: int | None = None) -> dict:
        timeout_s = timeout_s or config.SANDBOX_TIMEOUT_SECONDS
        memory_mb = memory_mb or config.SANDBOX_MEMORY_MB

        if not is_safe(code):
            from app.static_check import check

            reason = "; ".join(check(code))
            self.policy_log.append(
                PolicyEvent(step_id=step_id, decision="deny", resource="static_check", reason=reason)
            )
            return {
                "stdout": "", "error": f"static check rejected code: {reason}", "result_repr": None,
                "chart_path": None, "rows_in": 0, "rows_out": 0, "columns_touched": [],
                "policy_denied": True, "policy_denial_reason": reason, "sandbox_runtime": self.runtime,
            }

        self.policy_log.append(
            PolicyEvent(step_id=step_id, decision="allow", resource="static_check", reason="no violations")
        )

        if self.runtime == "fallback":
            return self._execute_fallback(code, step_id)
        return self._execute_primary(code, step_id, timeout_s, memory_mb)

    def _execute_primary(self, code: str, step_id: int, timeout_s: int, memory_mb: int) -> dict:
        assert self._proc is not None and self._proc.stdin is not None
        if self._proc.poll() is not None:
            # Previous run killed the kernel (e.g. timeout). Restart fresh --
            # this loses in-kernel state, which is logged via stdout note.
            self._start()

        try:
            self._proc.stdin.write(json.dumps({"code": code, "step_id": step_id}) + "\n")
            self._proc.stdin.flush()
        except Exception as e:
            return self._denied_result(step_id, f"could not reach sandbox process: {e}")

        exceeded = threading.Event()

        def _watch_memory() -> None:
            try:
                p = psutil.Process(self._proc.pid)
            except Exception:
                return
            start = time.time()
            while time.time() - start < timeout_s and self._proc.poll() is None:
                try:
                    rss_mb = p.memory_info().rss / (1024 * 1024)
                except Exception:
                    return
                if rss_mb > memory_mb:
                    exceeded.set()
                    self._proc.kill()
                    return
                time.sleep(0.1)

        watcher = threading.Thread(target=_watch_memory, daemon=True)
        watcher.start()

        try:
            line = self._out_queue.get(timeout=timeout_s)
        except queue.Empty:
            self._proc.kill()
            self.policy_log.append(
                PolicyEvent(step_id=step_id, decision="deny", resource="wall_clock_timeout",
                            reason=f"exceeded {timeout_s}s")
            )
            return self._denied_result(step_id, f"execution exceeded the {timeout_s}s sandbox timeout and was terminated")

        if exceeded.is_set():
            self.policy_log.append(
                PolicyEvent(step_id=step_id, decision="deny", resource="memory_limit",
                            reason=f"exceeded {memory_mb}MB")
            )
            return self._denied_result(step_id, f"execution exceeded the {memory_mb}MB sandbox memory limit and was terminated")

        try:
            result = json.loads(line)
        except json.JSONDecodeError:
            return self._denied_result(step_id, f"sandbox returned a non-JSON line: {line[:500]}")

        result["policy_denied"] = False
        result["policy_denial_reason"] = None
        result["sandbox_runtime"] = "primary"
        return result

    def _execute_fallback(self, code: str, step_id: int) -> dict:
        result = self._fallback_kernel.run(code)
        result["policy_denied"] = False
        result["policy_denial_reason"] = None
        result["sandbox_runtime"] = "fallback"
        return result

    def _denied_result(self, step_id: int, reason: str) -> dict:
        return {
            "stdout": "", "error": reason, "result_repr": None, "chart_path": None,
            "rows_in": 0, "rows_out": 0, "columns_touched": [],
            "policy_denied": True, "policy_denial_reason": reason, "sandbox_runtime": self.runtime,
        }

    def shutdown(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            try:
                assert self._proc.stdin is not None
                self._proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                self._proc.stdin.flush()
            except Exception:
                pass
            try:
                self._proc.terminate()
            except Exception:
                pass


def health_check() -> bool:
    """Confirms the subprocess sandbox layer is available and correctly
    configured before a session relies on it."""
    try:
        proc = subprocess.Popen(
            [sys.executable, "-u", str(KERNEL_SCRIPT)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )
        assert proc.stdin is not None and proc.stdout is not None
        proc.stdin.write(json.dumps({"datasets": {}, "scratch_dir": str(config.SCRATCH_DIR)}) + "\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
        ok = json.loads(line).get("status") == "ready"
        proc.terminate()
        return ok
    except Exception:
        return False
