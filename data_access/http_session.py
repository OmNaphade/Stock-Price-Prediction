"""One shared HTTP session for outbound market-data requests.

Tries curl_cffi (mimics a real browser TLS fingerprint, works around
datacenter IP blocks that Yahoo Finance applies to plain `requests`).
Falls back to a `requests.Session` with a browser User-Agent if curl_cffi
isn't installed.
"""

from __future__ import annotations

import requests

try:
    from curl_cffi import requests as curl_requests

    SESSION = curl_requests.Session(impersonate="chrome101")
    USING_CURL_CFFI = True
except Exception:
    SESSION = requests.Session()
    SESSION.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
    )
    USING_CURL_CFFI = False
