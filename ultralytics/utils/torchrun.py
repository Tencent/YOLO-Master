"""Windows-compatible entry point for ``torch.distributed.run``."""

from functools import partial


def disable_static_tcpstore_libuv(rendezvous_module) -> None:
    """Force the legacy TCPStore backend when the Windows torch wheel omits libuv."""
    rendezvous_module.TCPStore = partial(rendezvous_module.TCPStore, use_libuv=False)


def main() -> None:
    """Patch the upstream static rendezvous backend, then delegate to torchrun."""
    from torch.distributed.elastic.rendezvous import static_tcp_rendezvous
    from torch.distributed.run import main as torchrun_main

    disable_static_tcpstore_libuv(static_tcp_rendezvous)
    torchrun_main()


if __name__ == "__main__":
    main()
