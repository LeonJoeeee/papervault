"""A5 compatibility matrix for the acquisition-v2 durable surfaces.

Versions belong to each surface, independently of the package release. Current
bytes stay unchanged: index/JSONL already carry version=1; absent individual
markers mean the slices 1–6 format below. A future format writer MUST stamp its
bumped surface in acquisition_versions before writing new data. Queued claims
and leases live in the index and claims events, not a separate persistent queue.
The politeness registry is process-local and has no durable format to migrate.
"""

from types import MappingProxyType

SCHEDULER_VERSION = 2
INDEX_FORMAT_VERSION = 1
IDENTITY_VERSION = 2
BUDGET_LEDGER_FORMAT_VERSION = 1
CLAIMS_LEDGER_FORMAT_VERSION = 1
LEASE_FORMAT_VERSION = 1
QUEUE_FORMAT_VERSION = 1

FORMAT_VERSIONS = MappingProxyType({
    "index": INDEX_FORMAT_VERSION,
    "identity": IDENTITY_VERSION,
    "budget": BUDGET_LEDGER_FORMAT_VERSION,
    "claims": CLAIMS_LEDGER_FORMAT_VERSION,
    "lease": LEASE_FORMAT_VERSION,
    "queue": QUEUE_FORMAT_VERSION,
})
COMPATIBILITY_MATRIX = MappingProxyType({
    surface: (version - 1, version) for surface, version in FORMAT_VERSIONS.items()
})


class CompatibilityError(ValueError):
    """Refuse unsupported data before a v2 writer can dispatch or publish."""


def check_readable(version, *, current_version: int = INDEX_FORMAT_VERSION,
                   surface: str = "format") -> None:
    """νN reads νN−1..νN; malformed, older and newer data fail safe."""
    if (type(current_version) is not int or current_version < 1
            or type(version) is not int
            or not current_version - 1 <= version <= current_version):
        raise CompatibilityError(
            f"unsupported {surface} version {version!r}; reader {current_version!r} "
            "requires its current or preceding format; refusing acquisition writes")


def check_versions(versions) -> None:
    """Check an optional per-surface header without altering its data."""
    if not isinstance(versions, dict):
        raise CompatibilityError("malformed acquisition_versions header")
    for surface, version in versions.items():
        if surface not in FORMAT_VERSIONS:
            raise CompatibilityError(f"unknown acquisition surface {surface!r}")
        check_readable(version, current_version=FORMAT_VERSIONS[surface], surface=surface)
