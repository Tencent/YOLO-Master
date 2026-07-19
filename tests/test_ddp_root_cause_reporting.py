from pathlib import Path
from types import SimpleNamespace
from ultralytics.utils import MACOS
from ultralytics.utils import dist as dist_utils
from ultralytics.utils.dist import collect_ddp_error_logs, ddp_launch_env, generate_ddp_command
from ultralytics.utils.torch_utils import TORCH_1_9


class D:
    def __init__(self, p):
        self.args = SimpleNamespace(model="dummy.pt", save_dir="", resume=False)
        self.hub_session = None
        self.resume = True
        self.world_size = 2
        self.save_dir = Path(p)


def test_worker_record_and_launch_logging(tmp_path):
    d = D(tmp_path)
    cmd, f = generate_ddp_command(d)
    try:
        c = Path(f).read_text()
        assert "@record\ndef main():" in c
        compile(c, f, "exec")
        if TORCH_1_9:
            assert "--log-dir" in cmd
            assert ("--tee" in cmd) is not MACOS
            assert d.ddp_log_dir
        else:
            assert "--log-dir" not in cmd and "--tee" not in cmd
            assert d.ddp_log_dir is None
    finally:
        Path(f).unlink(missing_ok=True)


def test_collect_error_logs(tmp_path):
    p = tmp_path / "attempt_0" / "1"
    p.mkdir(parents=True)
    (p / "error.json").write_text('{"message":"ROOT_MARKER"}')
    (p / "stderr.log").write_text("TRACE_MARKER")
    out = collect_ddp_error_logs(tmp_path)
    assert "ROOT_MARKER" in out and "TRACE_MARKER" in out


def test_windows_ddp_launch_disables_libuv(monkeypatch):
    monkeypatch.setattr(dist_utils, "WINDOWS", True)
    monkeypatch.setenv("USE_LIBUV", "1")

    assert ddp_launch_env()["USE_LIBUV"] == "0"
