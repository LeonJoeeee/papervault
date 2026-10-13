"""Keep library unit tests independent of leases held by live OCR consumers."""

import pytest


@pytest.fixture(autouse=True)
def isolate_mineru_lease(monkeypatch, tmp_path):
    monkeypatch.setenv("PAPER_LIBRARY_MINERU_LEASE_FILE",
                       str(tmp_path / "paper-library-mineru.lease"))
