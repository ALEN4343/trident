"""Resumable download for the CUDA torch wheel.

pip's own retry kept restarting from the same offset and never advanced, and
curl cannot complete a TLS handshake to this host on this machine (schannel
refuses without a revocation check). Python with the OS trust store reaches it
at full speed, so this drives the transfer directly and resumes with a Range
request after each drop.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import trident  # noqa: F401  -- OS trust store into ssl

import httpx

URL = (
    "https://download.pytorch.org/whl/cu126/"
    "torch-2.14.0%2Bcu126-cp313-cp313-win_amd64.whl"
)
CHUNK = 1 << 20


def fetch(url: str, dest: Path, attempts: int = 60) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = None

    for attempt in range(1, attempts + 1):
        have = dest.stat().st_size if dest.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}

        try:
            with httpx.stream(
                "GET", url, headers=headers, timeout=60.0, follow_redirects=True
            ) as r:
                if r.status_code not in (200, 206):
                    raise RuntimeError(f"HTTP {r.status_code}")

                if total is None:
                    if r.status_code == 206:
                        total = int(r.headers["content-range"].split("/")[-1])
                    else:
                        total = int(r.headers.get("content-length", 0))

                # A 200 to a Range request means the server ignored it and is
                # sending the whole file, so the partial has to be discarded
                # rather than appended to.
                mode = "ab" if r.status_code == 206 and have else "wb"
                if mode == "wb":
                    have = 0

                last = time.time()
                with open(dest, mode) as fh:
                    for chunk in r.iter_bytes(CHUNK):
                        fh.write(chunk)
                        have += len(chunk)
                        if time.time() - last > 10:
                            pct = 100.0 * have / total if total else 0.0
                            print(f"  {have/1e6:8.1f} / {total/1e6:.1f} MB  {pct:5.1f}%",
                                  flush=True)
                            last = time.time()

            if total and dest.stat().st_size >= total:
                print(f"complete: {dest} ({dest.stat().st_size/1e6:.1f} MB)", flush=True)
                return dest
            print(f"  short read, retrying (attempt {attempt})", flush=True)

        except Exception as exc:  # noqa: BLE001
            got = dest.stat().st_size if dest.exists() else 0
            print(f"  drop at {got/1e6:.1f} MB: {type(exc).__name__} -- "
                  f"resuming (attempt {attempt})", flush=True)
            time.sleep(min(5.0, attempt))

    raise SystemExit("could not complete download")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".wheels") / "torch_cu126.whl"
    fetch(URL, out)
