# 0004 — Grey-zone download tiers ship flag-gated, default OFF

Status: Accepted (2026-07-17)

## Context
The download cascade includes shadow-library fallbacks (Sci-Hub, Anna's Archive) and anti-bot scraping tiers (curl-cffi / cloudscraper / scrapling) that materially raise paywalled-paper retrieval but carry legal exposure once publicly distributed. In the originating lab deployment the Sci-Hub tier ran enabled. Because this repo is the single upstream (ADR-0001), deleting the code would also strip the capability from the lab instance.

## Decision
Keep the code; gate every grey tier behind an explicit env flag, disabled by default; the README states that enabling them is the operator's own decision and responsibility. Rejected: full deletion (lab loses the capability, paywalled recall drops for everyone with no recourse); shipping any of it enabled by default (indefensible in public distribution).

## Consequences
Out-of-the-box behavior is clean: open-access-only cascade. Download success on paywalled literature is honestly lower by default than the lab's numbers — documented, not hidden. The lab instance re-enables its tiers privately via env. The flags concentrate the legal posture in one greppable place.
