from __future__ import annotations

import asyncio
import base64
import mimetypes
from urllib.parse import urlparse

import httpx

MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB

_BLOCKED_HOSTS = frozenset({"localhost", "127.0.0.1", "0.0.0.0", "::1"})
_BLOCKED_PREFIXES = (
    "10.", "169.254.",
    "172.16.", "172.17.", "172.18.", "172.19.", "172.20.", "172.21.",
    "172.22.", "172.23.", "172.24.", "172.25.", "172.26.", "172.27.",
    "172.28.", "172.29.", "172.30.", "172.31.",
    "192.168.",
)


async def download_image(
    url: str,
    client: httpx.AsyncClient | None = None,
    max_bytes: int = MAX_IMAGE_BYTES,
) -> tuple[bytes, str]:
    """Return (raw_bytes, mime_type) from a `data:` or `http(s)://` URL."""
    if url.startswith("data:"):
        header, _, b64 = url.partition(",")
        if not b64:
            raise ValueError("data URL missing payload")
        mime = header.split(";")[0].removeprefix("data:") or "image/png"
        raw = base64.b64decode(b64)
        if len(raw) > max_bytes:
            raise ValueError(f"data image too large: {len(raw)} bytes (max {max_bytes})")
        return raw, mime

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Unsupported scheme: {parsed.scheme}")
    hostname = parsed.hostname or ""
    if hostname in _BLOCKED_HOSTS or any(hostname.startswith(p) for p in _BLOCKED_PREFIXES):
        raise ValueError(f"Blocked host: {hostname}")

    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=30.0)
    try:
        # Retry the prompt-image fetch: a single transient hiccup on the prompt host
        # (429/503/reset) otherwise fails the whole prompt at the prepare stage, which is
        # an automatic duel loss for that stem in a live round.
        last_exc: Exception | None = None
        for attempt in range(4):
            try:
                resp = await client.get(url)
                resp.raise_for_status()
                if len(resp.content) > max_bytes:
                    raise ValueError(f"Image too large: {len(resp.content)} bytes (max {max_bytes})")
                mime = (resp.headers.get("content-type", "").split(";")[0].strip()
                        or mimetypes.guess_type(url)[0]
                        or "image/png")
                return resp.content, mime
            except ValueError:
                raise  # oversize / bad payload: not worth retrying
            except Exception as exc:
                last_exc = exc
                if attempt < 3:
                    await asyncio.sleep(1.5 * (attempt + 1))
        raise last_exc if last_exc else RuntimeError("download_image failed")
    finally:
        if own_client:
            await client.aclose()
