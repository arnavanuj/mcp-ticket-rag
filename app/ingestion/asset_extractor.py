"""Extract image links from markdown/plain URLs and apply extension filtering."""

from __future__ import annotations

import re
from urllib.parse import urlparse

_MARKDOWN_IMAGE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
_DIRECT_URL = re.compile(r"https?://[^\s)>\"']+", re.IGNORECASE)
SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def extract_image_urls(text: str) -> list[str]:
    """Extract candidate image URLs from markdown and direct links."""

    if not text:
        return []

    links = _MARKDOWN_IMAGE.findall(text)
    links.extend(_DIRECT_URL.findall(text))

    deduped: list[str] = []
    seen: set[str] = set()
    for link in links:
        clean = link.strip()
        if clean and clean not in seen:
            seen.add(clean)
            deduped.append(clean)
    return deduped


def is_supported_image_url(url: str) -> bool:
    """Return True when URL path ends with a supported image extension."""

    if not url:
        return False
    path = urlparse(url).path.lower()
    return path.endswith(SUPPORTED_IMAGE_EXTENSIONS)
