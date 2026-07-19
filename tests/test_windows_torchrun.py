from types import SimpleNamespace

from ultralytics.utils.torchrun import disable_static_tcpstore_libuv


def test_disable_static_tcpstore_libuv_binds_legacy_backend():
    def tcp_store(*args, **kwargs):
        return args, kwargs

    rendezvous = SimpleNamespace(TCPStore=tcp_store)
    disable_static_tcpstore_libuv(rendezvous)

    _, kwargs = rendezvous.TCPStore("127.0.0.1", 12345)
    assert kwargs["use_libuv"] is False
