import os
import sys
import types
from types import SimpleNamespace

from ultralytics.utils.torchrun import disable_libuv_rendezvous, disable_static_tcpstore_libuv


def test_disable_static_tcpstore_libuv_binds_legacy_backend():
    def tcp_store(*args, **kwargs):
        return args, kwargs

    rendezvous = SimpleNamespace(TCPStore=tcp_store)
    disable_static_tcpstore_libuv(rendezvous)

    _, kwargs = rendezvous.TCPStore("127.0.0.1", 12345)
    assert kwargs["use_libuv"] is False


def test_disable_libuv_rendezvous_patches_env_store(monkeypatch):
    calls = []

    def create_store(*args, **kwargs):
        calls.append((args, kwargs))
        return object()

    static = SimpleNamespace(TCPStore=lambda *args, **kwargs: (args, kwargs))
    elastic = types.ModuleType("torch.distributed.elastic.rendezvous")
    elastic.static_tcp_rendezvous = static
    rendezvous = types.ModuleType("torch.distributed.rendezvous")
    rendezvous.TCPStore = lambda *args, **kwargs: (args, kwargs)
    rendezvous._create_c10d_store = create_store
    for name, module in {
        "torch.distributed.elastic.rendezvous": elastic,
        "torch.distributed.rendezvous": rendezvous,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("USE_LIBUV", "1")

    disable_libuv_rendezvous()
    rendezvous._create_c10d_store("127.0.0.1", 12345, 0, 2, None, True)

    assert sys.modules["torch.distributed.rendezvous"] is rendezvous
    _, tcp_kwargs = rendezvous.TCPStore("127.0.0.1", 12345)
    assert tcp_kwargs["use_libuv"] is False
    assert calls[0][0][-1] is False
    assert calls[0][1] == {}
    assert os.environ["USE_LIBUV"] == "0"
