"""Windows-compatible entry point for ``torch.distributed.run``."""

import os
from functools import partial


def disable_static_tcpstore_libuv(rendezvous_module) -> None:
    """Force the legacy TCPStore backend when the Windows torch wheel omits libuv."""
    rendezvous_module.TCPStore = partial(rendezvous_module.TCPStore, use_libuv=False)


def disable_libuv_rendezvous() -> None:
    """Disable libuv in every rendezvous path used by the installed PyTorch version."""
    os.environ["USE_LIBUV"] = "0"

    from torch.distributed.elastic.rendezvous import static_tcp_rendezvous

    disable_static_tcpstore_libuv(static_tcp_rendezvous)

    # ``env://`` resolves through this private helper, which is not affected by
    # replacing ``static_tcp_rendezvous.TCPStore`` alone on recent PyTorch builds.
    import importlib

    rendezvous = importlib.import_module("torch.distributed.rendezvous")
    create_store = getattr(rendezvous, "_create_c10d_store", None)
    if create_store is None or getattr(create_store, "_ultralytics_libuv_disabled", False):
        return

    def create_store_without_libuv(*args, **kwargs):
        if len(args) >= 6:
            args = (*args[:5], False, *args[6:])
        else:
            kwargs["use_libuv"] = False
        return create_store(*args, **kwargs)

    create_store_without_libuv._ultralytics_libuv_disabled = True
    rendezvous._create_c10d_store = create_store_without_libuv


def main() -> None:
    """Patch the upstream static rendezvous backend, then delegate to torchrun."""
    from torch.distributed.run import main as torchrun_main

    disable_libuv_rendezvous()
    torchrun_main()


if __name__ == "__main__":
    main()
