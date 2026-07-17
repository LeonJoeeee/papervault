"""paper-library CLI.

Subcommands are thin wrappers around the services + Library API. Keep each
handler focused (≤30 lines); business logic stays in the modules they call.

Conventions:
  - Output: human-readable to stdout by default; pass `--json` for machine output.
  - Errors / warnings: stderr; exit code 1 on failure.
  - Use argparse only (no extra deps).
  - Each handler signature: ``def cmd_<name>(args) -> int:``  return exit code.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from . import Library


# --------------------------------- helpers ---------------------------------


def _err(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)


def _json(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


# ------------------------------- handlers ----------------------------------


def cmd_add(args) -> int:
    """`paper-library add <ident>` — enqueue one paper for background ingest.

    D11: this only resolves + caches the metadata card and drops it onto the
    daemon's download queue (status ``queued``); the PDF download + OCR happen
    in the background daemon, not in this terminal.
    """
    from .services.add_service import AddService

    lib = Library()
    result = AddService(lib).add(args.identifier, force_refresh=args.force_refresh)

    if args.json:
        _json(result)
    else:
        status = result.get("status", "unknown")
        key = result.get("key") or "-"
        print(f"{status}: {key}")
        if result.get("message"):
            print(f"  {result['message']}")
        meta = result.get("metadata") or {}
        if meta.get("title"):
            print(f"  {meta['title']}")
        for cand in result.get("candidates") or []:
            print(f"  candidate: {cand.get('key')} — {cand.get('title','')[:80]}")

    return 0 if result.get("status") in ("queued", "exists") else 1


def cmd_add_batch(args) -> int:
    """`paper-library add-batch <file>` — one identifier per line, concurrent."""
    from .services.batch import BatchAddService

    if args.file == "-":
        idents = [line.strip() for line in sys.stdin if line.strip() and not line.startswith("#")]
    else:
        idents = [line.strip() for line in Path(args.file).read_text().splitlines()
                  if line.strip() and not line.startswith("#")]
    if not idents:
        _err("no identifiers provided")
        return 1

    lib = Library()
    svc = BatchAddService(lib, max_workers=args.workers)

    def progress(ident: str, result: dict) -> None:
        print(f"{result.get('status','?'):<18} {result.get('key') or '-':<22} {ident}",
              file=sys.stderr)

    results = svc.add_many(idents, on_result=progress)
    # D11/S7: AddService.add's success status is "queued" (never "added"); an
    # existing complete/terminal card resolves as "exists". Both are successes.
    failures = sum(1 for r in results if r.get("status") not in ("queued", "exists"))
    print(f"-- {len(results)} processed, {failures} failure(s)", file=sys.stderr)
    return 0 if failures == 0 else 1


def cmd_list(args) -> int:
    """`paper-library list [filters]` — tabular library view."""
    lib = Library()

    def keep(p) -> bool:
        if args.review and not p.is_review:
            return False
        if args.year_min is not None and (p.year is None or p.year < args.year_min):
            return False
        if args.year_max is not None and (p.year is None or p.year > args.year_max):
            return False
        if args.has_extract and not (lib.has_extract(p.key, "md") or lib.has_extract(p.key, "txt")):
            return False
        if args.citation_min and (p.citation_count or 0) < args.citation_min:
            return False
        return True

    papers = sorted(
        (p for p in lib.all_papers() if keep(p)),
        key=lambda p: (-(p.year or 0), -(p.citation_count or 0), p.key),
    )[: args.limit]

    if args.json:
        _json([{"key": p.key, "year": p.year, "title": p.title,
                "first_author": p.first_author(), "citation_count": p.citation_count,
                "is_review": p.is_review} for p in papers])
    else:
        for p in papers:
            tag = "R" if p.is_review else " "
            author = (p.first_author() or "?")[:18]
            print(f"{p.key:<22} {p.year or '?':<5} {tag} {author:<18} {(p.title or '')[:80]}")
        print(f"-- {len(papers)} shown", file=sys.stderr)
    return 0


def cmd_search(args) -> int:
    """`paper-library search <query>` — in-library search with optional LLM rerank."""
    from .services.search_service import SearchService

    lib = Library()
    out = SearchService(lib).search(
        args.query, limit=args.limit, rerank=not args.no_rerank,
    )

    if args.json:
        _json(out)
    else:
        results = out.get("results") or []
        for r in results:
            badge = "R" if r.get("is_review") else " "
            year = r.get("year") or "?"
            print(f"{r.get('key',''):<22} {year:<5} {badge} {(r.get('title') or '')[:80]}")
        print(f"-- {len(results)}/{out.get('total_in_library', 0)} matched", file=sys.stderr)
    return 0


def cmd_show(args) -> int:
    """`paper-library show <key>` — full metadata for one paper."""
    lib = Library()
    p = lib.get(args.key)
    if p is None:
        _err(f"no paper with key {args.key!r}")
        return 1

    info = p.model_dump()
    info["has_pdf"] = lib.has_pdf(p.key)
    info["has_extract_md"] = lib.has_extract(p.key, "md")
    info["has_extract_txt"] = lib.has_extract(p.key, "txt")
    _json(info)
    return 0


def cmd_read(args) -> int:
    """`paper-library read <key>` — print extracted text."""
    lib = Library()
    if lib.get(args.key) is None:
        _err(f"no paper with key {args.key!r}")
        return 1

    path = lib.md_path(args.key) if args.format == "md" else lib.txt_path(args.key)
    if not path.is_file():
        _err(f"no {args.format} extract for {args.key} ({path.relative_to(lib.root)})")
        return 1

    text = path.read_text()
    cap = args.max_chars
    if cap and len(text) > cap:
        half = cap // 2
        text = f"{text[:half]}\n\n... [truncated {len(text) - cap} chars] ...\n\n{text[-half:]}"
    print(text)
    return 0


def cmd_bibtex(args) -> int:
    """`paper-library bibtex` — export BibTeX (full library or subset)."""
    lib = Library()
    keys = None
    if args.keys:
        keys = [k.strip() for k in args.keys.split(",") if k.strip()]
    text, missing = lib.export_bibtex(keys)

    if args.output:
        Path(args.output).write_text(text)
    else:
        print(text, end="")

    if missing:
        _err(f"{len(missing)} key(s) not in library: {', '.join(missing)}")
        return 1
    return 0


def cmd_cite_check(args) -> int:
    """`paper-library cite-check <file.tex>` — validate \\cite keys."""
    from . import cite_check

    tex_path = Path(args.tex_file)
    if not tex_path.is_file():
        _err(f"file not found: {tex_path}")
        return 1

    allowed = None
    if args.allowed_keys:
        allowed = [k.strip() for k in args.allowed_keys.split(",") if k.strip()]

    lib = Library()
    report = cite_check.check(tex_path.read_text(), lib, allowed=allowed)

    if args.json:
        _json(report)
    else:
        verdict = "OK" if report["ok"] else "FAIL"
        print(f"{verdict}: {report['cited_count']} citation(s), "
              f"{len(report['unique_keys'])} unique, "
              f"{len(report['dangling'])} dangling "
              f"(legal pool: {report['legal_count']})")
        for k in report["dangling"]:
            print(f"  dangling: {k}")
    return 0 if report["ok"] else 1


def _audit_scan(lib: Library) -> dict:
    """Walk the library and return three lists of inconsistencies.

    - drift: index field is null but the file exists on disk (historical bug).
    - dangling: index field points to a file that's gone.
    - orphans: file on disk has no matching paper in the index.
    """
    rel = {"pdf": "pdfs", "md": "extracts/md", "txt": "extracts/txt"}
    drift, dangling, orphans = [], [], []
    keys = set(lib.keys())

    for p in lib.all_papers():
        on_disk = {
            "pdf": lib.has_pdf(p.key),
            "md": lib.has_extract(p.key, "md"),
            "txt": lib.has_extract(p.key, "txt"),
        }
        recorded = {"pdf": p.pdf_path, "md": p.md_path, "txt": p.txt_path}
        for kind, present in on_disk.items():
            field = recorded[kind]
            if present and not field:
                drift.append({"key": p.key, "kind": kind,
                              "expected": f"{rel[kind]}/{p.key}.{ 'pdf' if kind=='pdf' else kind }"})
            elif field and not present:
                dangling.append({"key": p.key, "kind": kind, "field": field})

    for d, kind, ext in [(lib.pdfs_dir, "pdf", "pdf"),
                         (lib.md_dir, "md", "md"),
                         (lib.txt_dir, "txt", "txt")]:
        for f in d.glob(f"*.{ext}"):
            if f.stem not in keys:
                orphans.append({"file": str(f.relative_to(lib.root)), "kind": kind})
    return {"drift": drift, "dangling": dangling, "orphans": orphans}


def _audit_queue_report(lib: Library) -> dict:
    """Bucket all papers by ``download_status`` and tally extract coverage.

    Returns a dict with:
      - ``status_counts``: status string -> count, plus ``other`` for empty / unknown.
      - ``ok_total``: count of papers with status ``ok`` (D7).
      - ``ok_has_md`` / ``ok_has_txt`` / ``ok_missing_both``: extract coverage
        for ``ok`` papers.
      - ``recent_failed``: up to 10 most-recent failed papers (sorted by
        ``added_at`` desc), each ``{key, doi, arxiv_id, added_at}``.
    """
    from .models import DOWNLOAD_STATUS_FAILED, DOWNLOAD_STATUS_OK

    status_counts: dict[str, int] = {}
    ok_total = ok_has_md = ok_has_txt = ok_missing_both = 0
    failed: list = []

    for p in lib.all_papers():
        s = p.download_status or ""
        bucket = s if s else "other"
        status_counts[bucket] = status_counts.get(bucket, 0) + 1
        if s == DOWNLOAD_STATUS_OK:
            ok_total += 1
            has_md = lib.has_extract(p.key, "md")
            has_txt = lib.has_extract(p.key, "txt")
            if has_md:
                ok_has_md += 1
            if has_txt:
                ok_has_txt += 1
            if not has_md and not has_txt:
                ok_missing_both += 1
        if s == DOWNLOAD_STATUS_FAILED:
            failed.append(p)

    failed.sort(key=lambda p: p.added_at or "", reverse=True)
    recent_failed = [
        {"key": p.key, "doi": p.doi, "arxiv_id": p.arxiv_id, "added_at": p.added_at}
        for p in failed[:10]
    ]
    return {
        "status_counts": status_counts,
        "ok_total": ok_total,
        "ok_has_md": ok_has_md,
        "ok_has_txt": ok_has_txt,
        "ok_missing_both": ok_missing_both,
        "recent_failed": recent_failed,
    }


def _print_queue_report(report: dict) -> None:
    counts = report["status_counts"]
    print("download_status breakdown:")
    # Stable display order (D7 enum): pending, ok, then the three terminal
    # states, other, then anything else (legacy values pre-migration).
    pinned = ["pending", "ok"]
    tail = ["extract_failed", "failed", "metadata_only", "other"]
    seen: set[str] = set()
    for k in pinned + tail:
        if k in counts:
            print(f"  {k+':':<22} {counts[k]:>5}")
            seen.add(k)
    for k in sorted(counts):
        if k not in seen:
            print(f"  {k+':':<22} {counts[k]:>5}")

    print()
    print('extract coverage (papers with download_status "ok"):')
    print(f"  has md:               {report['ok_has_md']} / {report['ok_total']}")
    print(f"  has txt:              {report['ok_has_txt']} / {report['ok_total']}")
    print(f"  missing both:         {report['ok_missing_both']}")

    recent = report["recent_failed"]
    if recent:
        print()
        print(f"recent failed (last {len(recent)}):")
        for item in recent:
            ident = (f"doi={item['doi']}" if item['doi']
                     else f"arxiv={item['arxiv_id']}" if item['arxiv_id']
                     else "id=?")
            print(f"  {item['key']:<22} {ident}")


def _metadata_only_filter_match(p, spec: str) -> bool:
    """Return True if paper ``p`` matches the ``--retry-metadata-only`` filter
    ``spec``. Supported specs (case-insensitive on the keyword):

      - ``has-arxiv``       — row carries a non-empty ``arxiv_id``.
      - ``has-doi``         — row carries a non-empty ``doi``.
      - ``10.<prefix>...``  — DOI prefix match (``p.doi`` startswith ``spec``),
                              e.g. ``10.3390`` for MDPI gold-OA.

    The filter is the safety gate against blanket-resetting all 548
    metadata_only rows (262 no-DOI citation stubs, 34 egusphere abstracts,
    ~18 fabricated arxiv ids have no retrievable PDF — resetting those just
    burns the cascade). An unrecognised spec raises ValueError so a typo
    fails loud instead of matching nothing silently.
    """
    s = (spec or "").strip()
    low = s.lower()
    if low == "has-arxiv":
        return bool((p.arxiv_id or "").strip())
    if low == "has-doi":
        return bool((p.doi or "").strip())
    if s.startswith("10."):
        return (p.doi or "").startswith(s)
    raise ValueError(
        f"unrecognised --filter spec {spec!r}; "
        "expected 'has-arxiv', 'has-doi', or a DOI prefix like '10.3390'")


def _is_nodoi_stub(p) -> bool:
    """A no-DOI in-domain searchable stub eligible for by-title DOI resolution:
    no doi, no usable arxiv_id, but has an abstract. ``all_papers()`` already
    excludes domain-quarantined rows, so in-domain is implied. Mirrors
    ``reconcile._needs_doi_resolve`` minus the resolve_attempted_at guard (the
    operator command re-probes on demand)."""
    from .fetch import looks_like_arxiv
    if (p.doi or "").strip():
        return False
    arxiv = (p.arxiv_id or "").strip()
    if arxiv and looks_like_arxiv(arxiv):
        return False
    return bool((p.abstract or "").strip())


def _stub_filter_match(p, spec: str) -> bool:
    """Scope filter for ``--resolve-stub-dois``. Supported specs:

      - empty                — all no-DOI stubs (no extra scoping).
      - ``metadata-only``    — only ``download_status == metadata_only`` stubs
                               (the permanently-stuck subset).
      - a substring          — case-insensitive substring of the paper key
                               (e.g. an author-year prefix like ``Strauss2012``).

    An unrecognised spec is treated as a key-substring match (no raise), since
    the stub population isn't DOI/arxiv-scopeable the way metadata_only retry is.
    """
    from .models import DOWNLOAD_STATUS_METADATA_ONLY
    s = (spec or "").strip()
    if not s:
        return True
    if s.lower() == "metadata-only":
        return (p.download_status or "") == DOWNLOAD_STATUS_METADATA_ONLY
    return s.lower() in (p.key or "").lower()


def cmd_audit(args) -> int:
    """`paper-library audit [--fix] [--queue] [--retry-failed] [--retry-low-quality] [--retry-metadata-only]` — index/disk + BG queue.

    Default (no action flag) runs the drift/dangling/orphan scan. Each of the
    action flags (``--fix``, ``--queue``, ``--retry-failed``,
    ``--retry-low-quality``, ``--retry-metadata-only``) can be combined.
    Passing any action flag alone skips the drift scan.
    """
    from .models import (
        DOWNLOAD_STATUS_EXTRACT_FAILED,
        DOWNLOAD_STATUS_FAILED,
        DOWNLOAD_STATUS_METADATA_ONLY,
        DOWNLOAD_STATUS_OK,
        DOWNLOAD_STATUS_PENDING,
    )

    lib = Library()
    payload: dict = {}
    do_drift_scan = args.fix or not (
        args.queue or args.retry_failed or args.retry_low_quality
        or args.retry_metadata_only or args.resolve_stub_dois)

    # ---- retry-failed: mutate first so --queue picks up the new pending count.
    if args.retry_failed:
        reset_count = 0
        for p in lib.all_papers():
            if (p.download_status or "") == DOWNLOAD_STATUS_FAILED:
                p.download_status = DOWNLOAD_STATUS_PENDING
                reset_count += 1
        if reset_count:
            lib.save()
        payload["reset_count"] = reset_count
        if not args.json:
            print(f"reset {reset_count} papers from failed → pending")
            print("they will be retried at next paper-library-mcp startup")

    # ---- retry-low-quality (D7/D9): reset extract_failed → ok with
    # extract_attempts cleared, so the extract_queue recovery scan
    # (classify → EXTRACT) re-runs them under the current cascade. Only
    # touches papers that still have a PDF on disk (the others have nothing
    # to re-extract). Useful when the cascade has been upgraded. The flag
    # name is kept for back-compat; the old ``extract_low_quality`` ghost
    # state no longer exists post-migration.
    if args.retry_low_quality:
        reset_count = 0
        for p in lib.all_papers():
            if ((p.download_status or "") == DOWNLOAD_STATUS_EXTRACT_FAILED
                    and lib.has_pdf(p.key)):
                p.download_status = DOWNLOAD_STATUS_OK
                p.extract_attempts = 0
                reset_count += 1
        if reset_count:
            lib.save()
        payload["retry_low_quality_count"] = reset_count
        if not args.json:
            if args.retry_failed:
                print()
            print(f"reset {reset_count} papers from extract_failed → ok "
                  f"(attempts cleared)")
            print("extract_queue recovery will re-run them at next "
                  "paper-library-mcp startup")

    # ---- retry-metadata-only: reset SCOPED metadata_only → pending so a
    # restart re-enqueues them through the (now Sci-Hub-on) download cascade.
    # metadata_only is a terminal state (DOWNLOAD_STATUS_TERMINAL) that
    # reconcile / recovery never auto-revives, so an operator reset is the
    # only path back. SAFETY: a --filter spec is REQUIRED (or an explicit
    # --all override) — a bare invocation must NOT mass-reset all 548 rows,
    # since most have no retrievable PDF anywhere and resetting them just
    # burns the cascade pointlessly.
    if args.retry_metadata_only:
        spec = (args.filter or "").strip()
        if not spec and not args.all:
            _err("audit --retry-metadata-only requires --filter <spec> "
                 "(e.g. --filter has-arxiv or --filter 10.3390), or pass "
                 "--all to reset every metadata_only row")
            return 2

        candidates = [
            p for p in lib.all_papers()
            if (p.download_status or "") == DOWNLOAD_STATUS_METADATA_ONLY
        ]
        if args.all:
            matched = candidates
        else:
            matched = [p for p in candidates
                       if _metadata_only_filter_match(p, spec)]

        for p in matched:
            p.download_status = DOWNLOAD_STATUS_PENDING
        if matched:
            lib.save()

        reset_keys = [p.key for p in matched]
        payload["retry_metadata_only_count"] = len(matched)
        payload["retry_metadata_only_keys"] = reset_keys
        if not args.json:
            if args.retry_failed or args.retry_low_quality:
                print()
            scope = "--all (every metadata_only row)" if args.all else f"--filter {spec!r}"
            print(f"reset {len(matched)} papers from metadata_only → pending "
                  f"[{scope}; {len(candidates)} metadata_only total]")
            for key in reset_keys[:20]:
                print(f"  {key}")
            if len(reset_keys) > 20:
                print(f"  (+{len(reset_keys) - 20} more)")
            print("they will be retried at next paper-library-mcp startup")

    # ---- resolve-stub-dois: give no-DOI in-domain stubs a DOI by title.
    # SAFETY: --dry-run is the DEFAULT (write only when --no-dry-run is passed).
    # A no-DOI stub (has abstract, no doi, no usable arxiv) can never be enriched
    # or downloaded; resolving its DOI (conservatively — the resolver abstains on
    # ambiguity) unblocks the existing DOI-keyed cascade. Dry-run REPORTS
    # title -> resolved-DOI without writing; a real run writes via
    # Library.set_resolved_doi + a single save().
    if args.resolve_stub_dois:
        from .fetch import resolve_doi_by_title
        spec = (args.filter or "").strip()
        stubs = [p for p in lib.all_papers()
                 if _is_nodoi_stub(p) and _stub_filter_match(p, spec)]
        resolved, abstained = [], 0
        outcomes = {"set": 0, "collision": 0, "has_doi": 0}
        for p in stubs:
            doi = resolve_doi_by_title(p.title, list(p.authors or []), p.year)
            if not doi:
                abstained += 1
                continue
            entry = {"key": p.key, "title": (p.title or "")[:80], "doi": doi}
            if not args.dry_run:
                outcome, detail = lib.set_resolved_doi(p.key, doi,
                                                       provenance="doi_resolved")
                entry["outcome"] = outcome
                entry["detail"] = detail
                outcomes[outcome] = outcomes.get(outcome, 0) + 1
            resolved.append(entry)
        if resolved and not args.dry_run:
            lib.save()
        payload["resolve_stub_dois"] = {
            "dry_run": args.dry_run,
            "scanned": len(stubs),
            "resolved": len(resolved),
            "abstained": abstained,
            "outcomes": outcomes if not args.dry_run else None,
            "matches": resolved,
        }
        if not args.json:
            if args.retry_failed or args.retry_low_quality or args.retry_metadata_only:
                print()
            mode = "DRY-RUN (no writes)" if args.dry_run else "WRITE"
            scope = f"--filter {spec!r}" if spec else "all no-DOI stubs"
            print(f"resolve-stub-dois [{mode}; {scope}]: "
                  f"{len(stubs)} stubs scanned, {len(resolved)} resolved, "
                  f"{abstained} abstained")
            for e in resolved[:30]:
                tail = f" -> {e.get('outcome')}" if not args.dry_run else ""
                print(f"  {e['key']:<22} {e['doi']}{tail}")
                print(f"      {e['title']}")
            if len(resolved) > 30:
                print(f"  (+{len(resolved) - 30} more)")
            if args.dry_run:
                print("dry-run: nothing written. Re-run with --no-dry-run to apply.")

    # ---- queue report
    if args.queue:
        queue_report = _audit_queue_report(lib)
        payload["queue"] = queue_report
        if not args.json:
            if (args.retry_failed or args.retry_low_quality
                    or args.retry_metadata_only or args.resolve_stub_dois):
                print()  # separator between reset notice and queue table
            _print_queue_report(queue_report)

    # ---- drift / dangling / orphan scan (default + --fix)
    drift_report: dict | None = None
    if do_drift_scan:
        drift_report = _audit_scan(lib)
        fixed = 0
        if args.fix and drift_report["drift"]:
            for item in drift_report["drift"]:
                p = lib.get(item["key"])
                if p is None:
                    continue
                setattr(p, f"{item['kind']}_path", item["expected"])
                fixed += 1
            lib.save()
        drift_report["fixed"] = fixed
        payload.update(drift_report)
        if not args.json:
            if (args.queue or args.retry_failed or args.retry_low_quality
                    or args.retry_metadata_only or args.resolve_stub_dois):
                print()
            print(f"drift (file exists, index field null): {len(drift_report['drift'])}")
            print(f"dangling (index points to missing file): {len(drift_report['dangling'])}")
            print(f"orphans (file has no matching paper): {len(drift_report['orphans'])}")
            if args.fix:
                print(f"fixed: {fixed} index field(s) backfilled")
            for item in drift_report["drift"][:20]:
                print(f"  drift  {item['kind']:<3} {item['key']}")
            for item in drift_report["dangling"][:20]:
                print(f"  dangle {item['kind']:<3} {item['key']} → {item['field']}")
            for item in drift_report["orphans"][:20]:
                print(f"  orphan {item['kind']:<3} {item['file']}")

    if args.json:
        _json(payload)

    if drift_report is not None and (drift_report["drift"]
                                     or drift_report["dangling"]
                                     or drift_report["orphans"]):
        return 1
    return 0


def cmd_status(args) -> int:
    """`paper-library status` — single-screen library + queue snapshot.

    Concise alternative to ``audit --queue``: focuses on "is the system
    healthy, and what just happened" rather than the full state-machine
    breakdown. Designed for ``watch -n 60 paper-library status`` style use.
    """
    lib = Library()
    papers = lib.all_papers()

    from .models import (
        DOWNLOAD_STATUS_EXTRACT_FAILED,
        DOWNLOAD_STATUS_OK,
        DOWNLOAD_STATUS_PENDING,
    )
    from .services.classify import DOWNLOAD, EXTRACT, classify

    ok_count = pending = failed_count = lq_count = 0
    has_pdf = has_md = 0
    # Routing eligibility per classify() (D7), so the operator can see what the
    # router would do on next restart without a live daemon. ✦ Phase 28 (route
    # B): the insight queue was removed; only download + extract remain.
    #
    # CAVEAT (until reconcile_once / D8 lands): only the EXTRACT number mirrors
    # what the queue's start()-recovery actually pulls — extract_queue.start()
    # enqueues exactly `classify(p)==EXTRACT` (extract_queue.py). The DOWNLOAD
    # number is *reconcile*-eligibility, NOT download-queue recovery: classify
    # fires DOWNLOAD for firecrawl-md `ok` papers (status=ok, ¬has_pdf, has md —
    # SDD §5 rule 3), but download_queue.start() only re-enqueues
    # `status==pending ∧ ¬has_pdf` (download_queue.py), so those firecrawl-md
    # papers are NOT picked up by the download queue — they wait for
    # reconcile_once (D8, not yet built) to route them back. So dl_q_eligible
    # may over-count what the download queue will truly pull on next start.
    dl_q_eligible = ex_q_eligible = 0
    failed_papers: list = []
    ok_papers: list = []
    for p in papers:
        s = p.download_status or ""
        p_has_pdf = lib.has_pdf(p.key)
        p_has_md = lib.has_extract(p.key, "md")
        if s == DOWNLOAD_STATUS_OK:
            ok_count += 1
            ok_papers.append(p)
        elif s == DOWNLOAD_STATUS_PENDING:
            pending += 1
        elif s == "failed":
            failed_count += 1
            failed_papers.append(p)
        elif s == DOWNLOAD_STATUS_EXTRACT_FAILED:
            lq_count += 1
        if p_has_pdf:
            has_pdf += 1
        if p_has_md:
            has_md += 1
        route = classify(p, lib)
        if route == DOWNLOAD:
            dl_q_eligible += 1
        elif route == EXTRACT:
            ex_q_eligible += 1

    failed_papers.sort(key=lambda p: p.added_at or "", reverse=True)
    ok_papers.sort(key=lambda p: p.added_at or "", reverse=True)

    info = {
        "library_root": str(lib.root),
        "total_papers": len(papers),
        "downloaded_ok": ok_count,
        "pending": pending,
        "failed": failed_count,
        "extract_failed": lq_count,
        "has_pdf": has_pdf,
        "has_extract_md": has_md,
        "queue_eligible": {
            "download": dl_q_eligible,
            "extract": ex_q_eligible,
        },
        "recent_ok": [{"key": p.key, "status": p.download_status,
                        "added_at": p.added_at} for p in ok_papers[:5]],
        "recent_failed": [{"key": p.key,
                            "doi": p.doi or None,
                            "arxiv_id": p.arxiv_id or None}
                           for p in failed_papers[:5]],
    }

    if args.json:
        _json(info)
        return 0

    print(f"library:       {info['library_root']}")
    print(f"total papers:  {info['total_papers']:,}")
    print()
    print(f"  downloaded:  {ok_count:>5}   (with PDF: {has_pdf}, with md: {has_md})")
    print(f"  pending:     {pending:>5}   (download queue work)")
    print(f"  failed:      {failed_count:>5}   (no OA copy / paywalled)")
    if lq_count:
        print(f"  extract-failed: {lq_count:>5}   (PDF on disk, extraction gave up)")
    print()
    print("classify() routing (what the router would do on next start):")
    print(f"  download:    {dl_q_eligible:>5}   (classify → DOWNLOAD: no PDF; "
          "reconcile-eligibility — firecrawl-md ok papers wait for reconcile_once, D8)")
    print(f"  extract:     {ex_q_eligible:>5}   (classify → EXTRACT: PDF, no md, ok)")

    if ok_papers:
        print()
        print("recent successful ingests:")
        for p in ok_papers[:5]:
            via = p.download_source or "?"
            print(f"  ✓ {p.key:<22} via {via:<10} {(p.added_at or '')[:10]}")

    if failed_papers:
        print()
        print("recent failed downloads:")
        for p in failed_papers[:5]:
            ident = p.doi or p.arxiv_id or "?"
            print(f"  ✗ {p.key:<22} {ident}")
        if failed_count > 5:
            print(f"  (+{failed_count - 5} more — see `paper-library audit --queue`)")

    return 0


def cmd_topics(args) -> int:
    """`paper-library topics list|show <slug>` — topic membership inspection."""
    lib = Library()
    if args.topic_command == "list":
        slugs = sorted(p.stem for p in lib.topics_dir.glob("*.json"))
        for s in slugs:
            print(s)
        print(f"-- {len(slugs)} topic(s)", file=sys.stderr)
        return 0

    path = lib.topics_dir / f"{args.slug}.json"
    if not path.is_file():
        _err(f"no topic with slug {args.slug!r}")
        return 1
    _json(json.loads(path.read_text()))
    return 0


def cmd_config(args) -> int:
    """`paper-library config` — print effective config."""
    lib = Library()
    info = {
        "library_root": str(lib.root),
        "library_root_env": "$PAPER_LIBRARY_PATH" if "PAPER_LIBRARY_PATH" in __import__("os").environ else "(default)",
        "papers_in_library": len(lib.all_papers()),
        "version": _version(),
    }
    if args.json:
        _json(info)
    else:
        for k, v in info.items():
            print(f"{k}: {v}")
    return 0


def cmd_zotero_sync(args) -> int:
    """`paper-library zotero-sync` — push metadata to Zotero user library.

    Reads ZOTERO_API_KEY + ZOTERO_USER_ID from env (or .env if loaded).
    Metadata-only: no file uploads (300 MB Zotero quota would saturate at
    scale). For PDFs, import library.bib via Zotero Desktop instead.
    """
    import os
    from .integrations.zotero import ZoteroSync

    api_key = os.environ.get("ZOTERO_API_KEY") or ""
    user_id_raw = os.environ.get("ZOTERO_USER_ID") or ""
    if not api_key or not user_id_raw:
        _err("ZOTERO_API_KEY and ZOTERO_USER_ID must be set in environment")
        return 2
    try:
        user_id = int(user_id_raw)
    except ValueError:
        _err(f"ZOTERO_USER_ID must be integer, got {user_id_raw!r}")
        return 2

    lib = Library()
    sync = ZoteroSync(lib, api_key=api_key, user_id=user_id)

    def _progress(stage: str, done: int, total: int) -> None:
        print(f"  [{stage}] {done}/{total}", file=__import__("sys").stderr)

    report = sync.sync_all(dry_run=args.dry_run, on_progress=_progress)
    if args.json:
        _json(report)
    else:
        print(f"linked (existing items matched by DOI/arxiv): {report['linked']}")
        print(f"created (new Zotero items): {report['created']}")
        print(f"skipped (already had zotero_key): {report['skipped']}")
        if report["errors"]:
            print(f"errors: {len(report['errors'])}")
            for e in report["errors"][:5]:
                print(f"  {e}")
    return 0 if not report["errors"] else 1


def _version() -> str:
    try:
        from importlib.metadata import version
        return version("paper-library")
    except Exception:
        return "unknown"


# -------------------------------- argparse ---------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="paper-library",
        description="Personal academic paper library — manage your local store",
    )
    p.add_argument("--library-path", help="override $PAPER_LIBRARY_PATH for this invocation")

    sp = p.add_subparsers(dest="command", required=True)

    # add
    pa = sp.add_parser("add", help="ingest one paper by DOI / arxiv id / freeform")
    pa.add_argument("identifier")
    pa.add_argument("--force-refresh", action="store_true",
                    help="re-download + re-extract even if already complete")
    pa.add_argument("--json", action="store_true")
    pa.set_defaults(func=cmd_add)

    # add-batch
    pab = sp.add_parser("add-batch", help="ingest many papers (one ident per line)")
    pab.add_argument("file", help="path to a file with one identifier per line, or - for stdin")
    pab.add_argument("--workers", type=int, default=4)
    pab.set_defaults(func=cmd_add_batch)

    # list
    pl = sp.add_parser("list", help="list papers in the library")
    pl.add_argument("--review", action="store_true", help="only review/survey papers")
    pl.add_argument("--year-min", type=int)
    pl.add_argument("--year-max", type=int)
    pl.add_argument("--has-extract", action="store_true")
    pl.add_argument("--citation-min", type=int, default=0)
    pl.add_argument("--limit", type=int, default=50)
    pl.add_argument("--json", action="store_true")
    pl.set_defaults(func=cmd_list)

    # search
    ps = sp.add_parser("search", help="search the library")
    ps.add_argument("query")
    ps.add_argument("--limit", "-k", type=int, default=10)
    ps.add_argument("--no-rerank", action="store_true")
    ps.add_argument("--json", action="store_true")
    ps.set_defaults(func=cmd_search)

    # show
    psh = sp.add_parser("show", help="show metadata for one paper")
    psh.add_argument("key")
    psh.set_defaults(func=cmd_show)

    # read
    pr = sp.add_parser("read", help="print extracted text for a paper")
    pr.add_argument("key")
    pr.add_argument("--md", action="store_const", const="md", dest="format", default="md")
    pr.add_argument("--txt", action="store_const", const="txt", dest="format")
    pr.add_argument("--max-chars", type=int, default=None)
    pr.set_defaults(func=cmd_read)

    # bibtex
    pb = sp.add_parser("bibtex", help="export BibTeX (full library or subset)")
    pb.add_argument("--keys", help="comma-separated keys; default = all")
    pb.add_argument("-o", "--output", help="write to file instead of stdout")
    pb.set_defaults(func=cmd_bibtex)

    # cite-check
    pc = sp.add_parser("cite-check", help="validate \\cite keys in a LaTeX file")
    pc.add_argument("tex_file")
    pc.add_argument("--allowed-keys", help="comma-separated subset to validate against")
    pc.add_argument("--json", action="store_true")
    pc.set_defaults(func=cmd_cite_check)

    # audit
    pau = sp.add_parser("audit", help="check / fix index ↔ disk consistency")
    pau.add_argument("--fix", action="store_true",
                     help="backfill index fields when files exist but the index is missing them")
    pau.add_argument("--queue", action="store_true",
                     help="show background ingest queue status (download_status breakdown)")
    pau.add_argument("--retry-failed", action="store_true",
                     help="reset all 'failed' papers to 'pending' so they get retried")
    pau.add_argument("--retry-low-quality", action="store_true",
                     help="reset 'extract_failed' papers (with a PDF) → 'ok' + clear attempts so the extract queue re-runs them under the current cascade")
    pau.add_argument("--retry-metadata-only", action="store_true",
                     help="reset SCOPED 'metadata_only' papers → 'pending' so a restart re-enqueues them through the download cascade; REQUIRES --filter (or --all)")
    pau.add_argument("--filter", metavar="SPEC",
                     help="scope for --retry-metadata-only ('has-arxiv', 'has-doi', DOI prefix like '10.3390') "
                          "OR for --resolve-stub-dois ('metadata-only', or a key substring like 'Strauss2012')")
    pau.add_argument("--all", action="store_true",
                     help="with --retry-metadata-only, override the --filter safety and reset EVERY metadata_only row")
    pau.add_argument("--resolve-stub-dois", action="store_true",
                     help="resolve DOIs by title for no-DOI in-domain stubs (has abstract, no doi, no usable arxiv) "
                          "so the DOI-keyed download cascade can fetch them; DRY-RUN by default (pass --no-dry-run to write)")
    pau.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True,
                     help="for --resolve-stub-dois: report matches WITHOUT writing (default). --no-dry-run applies them.")
    pau.add_argument("--json", action="store_true")
    pau.set_defaults(func=cmd_audit)

    # topics
    pt = sp.add_parser("topics", help="topic membership inspection")
    pts = pt.add_subparsers(dest="topic_command", required=True)
    pts.add_parser("list", help="list known topics")
    pts_show = pts.add_parser("show", help="show topic JSON contents")
    pts_show.add_argument("slug")
    pt.set_defaults(func=cmd_topics)

    # status
    pst = sp.add_parser("status",
                        help="quick library + BG queue dashboard")
    pst.add_argument("--json", action="store_true")
    pst.set_defaults(func=cmd_status)

    # config
    pcfg = sp.add_parser("config", help="print effective configuration")
    pcfg.add_argument("--json", action="store_true")
    pcfg.set_defaults(func=cmd_config)

    # zotero-sync
    pzs = sp.add_parser("zotero-sync",
                         help="push paper metadata + abstracts to Zotero (metadata-only, no files)")
    pzs.add_argument("--dry-run", action="store_true",
                     help="build payloads and report counts without posting")
    pzs.add_argument("--json", action="store_true")
    pzs.set_defaults(func=cmd_zotero_sync)

    # Phase 28 (2026-05-24, route B): the `paper-library insight` subcommand +
    # 5-Q digest pipeline were removed; knowledge synthesis lives in the
    # knowledge-system service now. Phase (2026-05-28): the legacy distill-tracking
    # MCP tools (list_undistilled / mark_distilled / flag_insight_invalid) were
    # deleted. Phase 1 post-ingest redesign (2026-05-31): books are out of
    # paper-library scope — the add-textbook / add-review CLI commands +
    # chapters / manual_ingest machinery were removed; review handling stays
    # via the bibliometric ``is_review`` field.

    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.library_path:
        import os
        os.environ["PAPER_LIBRARY_PATH"] = args.library_path
    try:
        return args.func(args)
    except NotImplementedError as exc:
        _err(f"{args.command} not yet implemented: {exc}")
        return 2
    except Exception as exc:
        _err(repr(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
