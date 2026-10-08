from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import requests

from ..models import Paper
from ._shared import BROWSER_HEADERS, TIMEOUT, _is_pdf_bytes


def _try_url_overrides(paper: Paper) -> Optional[bytes]:
    """User-curated DOI/arxiv -> URL overrides. Tried first.

    Format: ``$PAPER_LIBRARY_PATH/url_overrides.json``::

        {"<doi-or-arxiv-id>": "<full-PDF-url>", ...}

    Use case: author lab pages hosting their own PDFs, institutional
    repository links, any case where the operator already knows the
    exact PDF URL. Re-read on every call so manual edits take effect
    without restarting the daemon.
    """
    if not paper.doi and not paper.arxiv_id:
        return None
    # Resolve the vault dir through the SAME live-env + config path as the rest of the
    # library (services.concurrency._vault_path: PAPERVAULT_VAULT → PAPER_LIBRARY_PATH →
    # config.VAULT_PATH, expanduser'd) so a relocated vault's overrides are still read,
    # instead of a stale ~/paper-vault default that diverges from Library.root.
    from papervault.library.services.concurrency import _vault_path
    overrides_path = Path(_vault_path()) / "url_overrides.json"
    try:
        overrides = json.loads(overrides_path.read_text())
    except (FileNotFoundError, ValueError):
        return None
    url = overrides.get(paper.doi or "") or overrides.get(paper.arxiv_id or "")
    if not url:
        return None
    try:
        r = requests.get(url, timeout=TIMEOUT,
                         headers=BROWSER_HEADERS, allow_redirects=True)
        if r.ok and _is_pdf_bytes(r.content):
            return r.content
    except Exception:
        pass
    return None
