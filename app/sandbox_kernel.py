"""
Persistent sandbox kernel -- runs as a child process.

One kernel process per session: the same dataframes and intermediate
variables survive across plan steps and across retries within a step,
exactly like one notebook kernel staying alive for the life of a session
(Section 4.6). Communicates with the parent (sandbox.py) via line-delimited
JSON over stdin/stdout.

Defense-in-depth network block: even though the static checker (static_check
.py) already rejects `import socket`/`requests`/etc. before code reaches
here, this process also monkeypatches socket.socket to refuse connections,
so a construct the static checker didn't anticipate still can't reach the
network. This process's only legitimate filesystem write target is its
scratch directory (for saved charts).
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import traceback
from pathlib import Path

import pandas as pd
import numpy as np
import duckdb


def _block_network() -> None:
    import socket

    def _blocked(*args, **kwargs):
        raise PermissionError("network access is denied inside the sandbox")

    socket.socket = _blocked  # type: ignore[assignment]
    socket.create_connection = _blocked  # type: ignore[assignment]


def _safe_builtins() -> dict:
    import builtins

    allowed_names = [
        "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter",
        "float", "format", "frozenset", "getattr", "hasattr", "hash", "hex",
        "int", "isinstance", "issubclass", "iter", "len", "list", "map",
        "max", "min", "next", "oct", "ord", "pow", "print", "range", "repr",
        "reversed", "round", "set", "slice", "sorted", "str", "sum", "tuple",
        "type", "zip", "True", "False", "None", "Exception", "ValueError",
        "TypeError", "KeyError", "IndexError", "StopIteration", "ZeroDivisionError",
        "RuntimeError", "AttributeError", "NotImplementedError", "sum",
        "object", "super", "classmethod", "staticmethod", "property",
        "StopAsyncIteration", "OverflowError", "FloatingPointError",
    ]
    safe = {}
    for name in allowed_names:
        if hasattr(builtins, name):
            safe[name] = getattr(builtins, name)
    # __import__ and __build_class__ must stay available at runtime for the
    # `import` statement and `class` statement to work at all (e.g. a step
    # that does `import time`, or duckdb's own internal lazy imports) --
    # static_check.py is the actual gate against *dynamic* use, since it
    # rejects both `import <disallowed module>` and a bare __import__(...)
    # call in user code before any of this ever runs.
    safe["__import__"] = builtins.__import__
    safe["__build_class__"] = builtins.__build_class__
    return safe


class Kernel:
    def __init__(self, datasets: dict[str, str], scratch_dir: str):
        _block_network()
        self.scratch_dir = Path(scratch_dir)
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        self.dfs: dict[str, pd.DataFrame] = {}
        self.duck = duckdb.connect(database=":memory:")
        for name, path in datasets.items():
            df = self._load(path)
            self.dfs[name] = df
            self.duck.register(name, df)
        self._chart_counter = 0
        self.namespace = self._build_namespace()

    def _load(self, path: str) -> pd.DataFrame:
        p = Path(path)
        suffix = p.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(p)
        if suffix in (".xlsx", ".xls"):
            return pd.read_excel(p)
        if suffix == ".parquet":
            return pd.read_parquet(p)
        if suffix == ".json":
            return pd.read_json(p)
        raise ValueError(f"unsupported dataset format: {suffix}")

    def _save_chart(self, fig, name: str = "chart") -> str:
        self._chart_counter += 1
        out_path = self.scratch_dir / f"{name}_{self._chart_counter}.png"
        fig.savefig(out_path)
        return str(out_path)

    def _build_namespace(self) -> dict:
        ns: dict = {
            "__builtins__": _safe_builtins(),
            "pd": pd,
            "np": np,
            "duckdb": self.duck,
            "dfs": self.dfs,
            "save_chart": self._save_chart,
        }
        # Convenience: expose each table directly by name too.
        ns.update(self.dfs)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            ns["plt"] = plt
        except Exception:
            pass
        return ns

    def run(self, code: str) -> dict:
        stdout_buf = io.StringIO()
        result_repr = None
        error = None
        rows_in = sum(len(df) for df in self.dfs.values())
        rows_out = 0
        columns_touched: list[str] = []

        try:
            with contextlib.redirect_stdout(stdout_buf):
                exec(compile(code, "<sandbox>", "exec"), self.namespace)
            result = self.namespace.get("result")
            if result is not None:
                result_repr = repr(result)[:4000]
                if isinstance(result, (pd.DataFrame, pd.Series)):
                    rows_out = len(result)
                    if isinstance(result, pd.DataFrame):
                        columns_touched = [str(c) for c in result.columns]
                else:
                    rows_out = 1
        except Exception:
            error = traceback.format_exc(limit=6)

        if not columns_touched:
            all_cols = set()
            for df in self.dfs.values():
                all_cols.update(str(c) for c in df.columns)
            columns_touched = [c for c in all_cols if c in code]

        chart_path = None
        # If the code called save_chart, its return value is normally
        # assigned into `result`; otherwise check for the most recent file.
        if isinstance(result_repr, str) and str(self.scratch_dir) in (result_repr or ""):
            chart_path = result_repr.strip("'\"")

        return {
            "stdout": stdout_buf.getvalue()[:4000],
            "error": error,
            "result_repr": result_repr,
            "chart_path": chart_path,
            "rows_in": rows_in,
            "rows_out": rows_out,
            "columns_touched": columns_touched,
        }


def main() -> None:
    init_line = sys.stdin.readline()
    init = json.loads(init_line)
    kernel = Kernel(init["datasets"], init["scratch_dir"])
    print(json.dumps({"status": "ready"}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if msg.get("cmd") == "shutdown":
            break
        code = msg.get("code", "")
        result = kernel.run(code)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
