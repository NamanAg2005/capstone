from pathlib import Path

import pandas as pd
import pytest

from app import config
from app.sandbox import SandboxSession, health_check

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = ROOT / "sample_data"


@pytest.fixture(scope="module")
def sandbox(tmp_path_factory):
    scratch = tmp_path_factory.mktemp("sandbox_scratch")
    sess = SandboxSession({"orders": str(SAMPLE_DIR / "orders.csv")}, scratch)
    yield sess
    sess.shutdown()


def test_health_check_passes():
    assert health_check() is True


def test_basic_execution(sandbox):
    result = sandbox.execute("result = len(orders)", step_id=1)
    assert result["error"] is None
    assert result["result_repr"] == "35"


def test_persistent_state_across_calls(sandbox):
    sandbox.execute("x = 10", step_id=2)
    result = sandbox.execute("result = x + 5", step_id=2)
    assert result["result_repr"] == "15"


def test_runtime_error_is_captured(sandbox):
    result = sandbox.execute("result = 1 / 0", step_id=3)
    assert result["error"] is not None
    assert "ZeroDivisionError" in result["error"]


def test_static_check_blocks_disallowed_code(sandbox):
    result = sandbox.execute("import os\nresult = os.getcwd()", step_id=4)
    assert result["policy_denied"] is True


def test_duckdb_backend_query(sandbox):
    result = sandbox.execute("result = duckdb.sql('select count(*) as n from orders').fetchone()[0]", step_id=5)
    assert result["error"] is None
    assert result["result_repr"] == "35"


def test_timeout_terminates_long_running_code(tmp_path):
    sess = SandboxSession({}, tmp_path)
    try:
        result = sess.execute("import time as _t\nwhile True:\n    pass\n", step_id=6, timeout_s=2, memory_mb=256)
        assert result["error"] is not None
        assert "timeout" in result["error"].lower() or result.get("policy_denied")
    finally:
        sess.shutdown()
