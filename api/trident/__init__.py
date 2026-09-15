"""TRIDENT - marine pollution intelligence.

Detect -> localise -> attribute -> forecast -> dispatch -> verify.
"""

import os

__version__ = "0.1.0"


def _trust_os_certificates() -> None:
    """Route all TLS verification through the operating system trust store.

    Campus and corporate networks terminate TLS at an inspecting proxy whose
    root certificate is installed system-wide but is absent from the bundle
    shipped inside certifi. Without this, every library that fetches something
    fails verification independently -- the forcing API, the model weights
    download, dataset fetches -- and each one has to be patched separately.

    Applied here rather than per call site because it has to be in effect
    before any third-party library builds its own SSL context at import time.
    Set TRIDENT_NO_TRUSTSTORE=1 to opt out.
    """
    if os.environ.get("TRIDENT_NO_TRUSTSTORE"):
        return
    try:
        import truststore
    except ImportError:
        return
    truststore.inject_into_ssl()


_trust_os_certificates()
