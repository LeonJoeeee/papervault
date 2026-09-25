"""LightRAG ingest layer (SDD §6.1): paper → graph.

- distill_batch(): ★ two-phase batch enqueue → single process. NOT 逐篇 await
  ainsert (that pins doc-level concurrency to 1 and defeats max_parallel_insert, §6.1.d).
  元数据态(fp==META)→ ledger done_meta, 不入图。done/PROCESSED 回写交 reconcile 读
  doc_status 终态(§6.6),这里只置 processing。
- remove_one(): adelete_by_doc_id with the 4-way DeletionResult.status handling (§6.1):
  success|not_found → 收口; not_allowed(403 busy) → pending_remove; fail → error.
- REDISTILL = remove_one(delete_ledger=False) 先删旧 doc(ainsert 按 doc_id 存在性去重,
  对已存在 id 直插会写 dup-<hash> FAILED 行,污染 doc_status + 饿池),再进 distill_batch。
- Insert guard (#131): distill_batch never enqueues over an existing doc. LightRAG 1.5 rejects an
  enqueue whose doc_id doc_status already holds, or whose canonical file_path basename ANY row
  already carries (regardless of status), and records the rejection as a FAILED `dup-*` marker
  with no content. So an existing `paper:<key>` is adopted (PROCESSED → done, in flight →
  processing), stale `dup-*` markers on the paper's file_path are purged first, and any other
  row still holding that file_path blocks the insert (ledger error) instead of minting a marker.
- Abstract-only docs (#144): an `ABSTRACT:` fingerprint inserts ONE doc built from the paper's
  metadata + abstract (ingest/abstract_doc.py) instead of reading an extract. Adoption of an
  existing `paper:<key>` is limited to a doc of the WANTED class (abstract vs full text): a doc
  of the other class is deleted and replaced, so an abstract doc never stands in for full text.
- rollback_abstract_docs(): the one-step undo — delete every abstract-only doc, ledger → done_meta.
"""
from __future__ import annotations

import hashlib
import logging
import re
from typing import Optional

from lightrag.base import DocStatus
from lightrag.utils import sanitize_text_for_encoding
from lightrag.utils_pipeline import normalize_document_file_path

from papervault.knowledge.ingest.abstract_doc import (
    build_abstract_doc,
    done_status_for,
    is_abstract_fp,
    is_abstract_text,
)
from papervault.knowledge.ingest.fingerprint import META
from papervault.knowledge.ingest.paper_library_client import _strip_references
from papervault.knowledge.ingest.vault import PaperRecord, read_extract_raw
from papervault.knowledge.ledger import store as ledger

log = logging.getLogger("ks.ingest.distill")


def doc_id(key: str) -> str:
    return f"paper:{key}"


def file_path(key: str) -> str:
    return f"paper/{key}"


def doc_status_of(st) -> object | None:
    """Pull `.status` out of an aget_docs_by_ids / doc_status entry (SDD §6.6 ★ contract).

    LightRAG's aget_docs_by_ids is type-hinted dict[str, DocProcessingStatus] but at RUNTIME
    returns {doc_id: plain dict}: it passes doc_status.get_by_id() straight through and both
    configured backends return a plain dict (PGDocStatusStorage.get_by_id `return dict(...)`;
    JsonDocStatusStorage.get_by_id `return self._data.get(id)`), whose `status` is a BARE STRING
    ('processed'/'failed'/…), not a DocProcessingStatus. So `getattr(st, "status")` on the dict
    is always None — which silently misclassifies every PROCESSED doc, trips stuck_guard, and
    flips successful docs to error (breaks §6.5 closure). Use dict-subscript; fall back to
    getattr only for an object-shaped st. DocStatus is a str-Enum, so the bare string compares
    equal to DocStatus.PROCESSED etc. downstream.
    """
    if st is None:
        return None
    if isinstance(st, dict):
        return st.get("status")
    return getattr(st, "status", None)


# LightRAG's enqueue-time duplicate-rejection rows (pipeline.py: `compute_mdhash_id(...,
# prefix="dup-")`, status FAILED, never any full_docs content) — #131.
_DUP_MARKER_PREFIX = "dup-"
# Upper bound on markers purged for ONE file_path per insert: each deletes one row, so the loop
# ends when none is left; the cap only stops a runaway if a backend delete silently no-ops.
_MAX_MARKER_PURGE = 64


async def _is_stale_dup_marker(rag, did: str, doc) -> bool:
    """A `dup-*` row that is FAILED and has no full_docs content: an audit record of a rejected
    enqueue, never a document. LightRAG never retries it (its consistency pass only *preserves*
    FAILED rows without content — the recurring "Preserving N failed document entries" log), yet
    its filename dedup still matches it, so it blocks every later insert of that file_path."""
    if not did.startswith(_DUP_MARKER_PREFIX) or doc_status_of(doc) != DocStatus.FAILED:
        return False
    return not await rag.full_docs.get_by_id(did)


async def _clear_insert_path(rag, key: str) -> tuple[Optional[str], int]:
    """Purge stale `dup-*` markers holding this paper's file_path (#131 Bound 2).

    Returns (blocking_doc_id, purged): blocking_doc_id is the id of a row that is NOT a stale
    marker yet still holds the file_path (LightRAG's filename dedup would reject the insert
    against it), or None when the path is clear. Only stale markers are ever deleted.
    `get_doc_by_file_basename` returns the oldest match, so this re-queries after each delete.
    """
    basename = normalize_document_file_path(file_path(key))
    purged = 0
    while True:
        match = await rag.doc_status.get_doc_by_file_basename(basename)
        if not match:
            return None, purged
        mid, mdoc = match
        if purged >= _MAX_MARKER_PURGE or not await _is_stale_dup_marker(rag, mid, mdoc):
            return mid, purged
        await rag.doc_status.delete([mid])
        purged += 1
        log.info("distill_batch: purged stale duplicate marker %s blocking %s (#131)", mid, basename)


# V7(SDD §6.9.7 / §6.1.c)— KS-side strip of trailing acknowledgements-class sections.
# These sections (致谢/funding/grant 号/email/利益声明) are the richest source of junk
# graph entities (person names, institutions, grant ids) that wear legitimate-ontology
# clothing → noise in the graph; cutting them also trims chunk count (吞吐). Build-time
# only (composed into clean(), which runs pre-ainsert at extraction time).
#
# CONSERVATIVE by design (§6.9.7): markdown-heading based ONLY (never an in-body keyword
# search, so "this work was funded by..." inside a methods paragraph is NOT a cut point); a
# bounded narrow cut (ack heading followed by another heading) is honored at any position while
# an unbounded cut-to-EOF is only honored in the trailing half; a body-marker gate keeps any
# removable segment that holds a floating table/figure/$$ block (genuine results body the
# narrow cut would otherwise lose); a char-count guard (_MIN_KEEP_RATIO) aborts the strip
# entirely if it would remove too much (mis-cut on a short/OCR-mangled doc); a doc with none of
# these sections is an exact no-op.
#
# 大小写不敏感(re.I). \b after the heading keeps "fundingsomething" / "conflictsxyz" from
# matching but allows "Funding", "Funding Information", "Conflicts of Interest", etc.
# ★ `(?:[\dIVXLC]{1,6}[.)]?\s+)?` allows an optional leading section number / roman numeral
# between the `#` markers and the keyword, so '## 5. Acknowledgments', '## VII. ACKNOWLEDGMENTS',
# '## 6 Acknowledgment' all match (drill fix 2026-06-02, SDD §6.9.7 / §11 F22): without it the
# regex required the keyword immediately after `#` and silently skipped every numbered ack
# heading, leaking author/grant/institution junk into the graph.
# ★ declaration-class variants (drill fix 2026-06-02c, SDD §6.9.7 / §11 F22 b3): two real
# journal heading styles leaked through — Liu2021 '## Compliance with ethical standard Conflict'
# (Springer; bare 'Conflict' under a 'Compliance...' heading, not 'conflicts of interest') and
# Liu2024a '## Disclosure statement' ('disclosure'/'declaration of interest' were absent). Added
# 'disclosure( statement)?' / 'declaration(s) of (competing )?interest(s)?' / 'compliance with
# ethical standard(s)?'. NOTE: this keyword list is hand-maintained and NOT exhaustive — a novel
# labelled declaration heading may still leak (accepted leak: a few ack names/grant ids reach the
# graph; the miss is conservative-direction, never body loss). Broaden only doc-first (SDD).
_ACK_HEADINGS = re.compile(
    r"(?im)^\s{0,3}#{1,4}\s*"
    r"(?:[\dIVXLC]{1,6}[.)]?\s+)?"  # optional section number / roman numeral prefix
    r"(?:"
    r"acknowledge?ments?|acknowledgement|"
    r"funding|"
    r"author\s+contributions?|"
    r"conflicts?\s+of\s+interest|competing\s+interests?|"
    r"declaration\s+of\s+competing\s+interest|"
    r"declarations?\s+of(?:\s+competing)?\s+interests?|"  # 'Declaration of interests'
    r"disclosure(?:\s+statement)?|"                        # Liu2024a '## Disclosure statement'
    r"compliance\s+with\s+ethical\s+standards?|"           # Liu2021 Springer 'Compliance...'
    r"data\s+availability(?:\s+statement)?"
    r")\b"
)
# Any markdown heading (used to find where the ack section ENDS — its scope stops at the next
# heading so a trailing Appendix is preserved).
_ANY_HEADING = re.compile(r"(?im)^\s{0,3}#{1,4}\s+\S")
_MIN_KEEP_RATIO = 0.5

# ★ body-marker probe (drill fix 2026-06-02b, SDD §6.9.7 / §11 F22 a/a2). The narrow-cut only
# protects an appendix that has its OWN markdown heading. OCR/markdown frequently emits a
# FLOATING data table / figure caption / display-math block between sections (or after the
# acknowledgements, before EOF) with no heading of its own — narrow-cut/EOF-cut would delete it
# = genuine results body lost (Adriani2009 lost 'TABLE I: positron fraction summary' + 'FIG. 5'
# + 13 $$ blocks, -24%; Cholis2020 lost 'TABLE II' 19-row fit constraints, -17%; Dong2020/
# Maurin2019 lost 'Figure N:' captions). So before removing an ack segment, scan it for these
# markers; if present the segment is NOT pure ack/ref prose and must be kept (gate ③ in
# _strip_acks). Conservative on purpose — when a removable segment shows any sign of body, keep
# it. Line-anchored (re.M) so a TABLE/FIG token must start a line (a caption/heading), not an
# in-prose mention ('see Table 2'); $$ and <table> are unambiguous block markers.
_BODY_MARKERS = re.compile(
    r"(?im)"
    r"(?:<table\b)"                              # HTML table (OCR'd data table)
    r"|(?:^\s*\|.*\|.*\|)"                       # markdown table row (>=2 pipes)
    r"|(?:^\s*TABLE\s+[IVXLC\d]+\b)"             # 'TABLE I' / 'TABLE II' caption at line start
    r"|(?:^\s*FIG(?:URE)?\.?\s*[IVXLC\d]+\b)"    # 'FIG. 5' / 'Figure 44:' caption at line start
    r"|(?:\$\$)"                                 # display-math block delimiter
)


def _strip_acks(text: str) -> str:
    """V7(SDD §6.9.7):剥 Acknowledg(e)ments / Funding / Author Contributions /
    Conflicts(Competing) Interests / Data Availability / Disclosure / Declaration of
    Interests / Compliance with Ethical Standards 等清晰标注的**尾部**声明章节
    (heading 词表人工维护, **非穷尽** —— 残余漏剥 = accepted leak, 方向保守=漏点噪声非丢正文)。

    保守口径(逐个 ack 候选标题从前到后过三道闸,删第一个全过的):
    ① **位置闸**:窄切(待删段有**下一标题** = 删除有界、只削单段)→ 不论位置一律认;
       无界 EOF 切(无下一标题)→ 仅当命中点落在后段(> 50%)才认(无界切只在文末安全)。
       (drill 2026-06-02b:旧"一律须 > 50%"对 appendix-heavy 论文过度限制 → 50% 前的真致谢段
        漏剥、人名/grant 号漏进图;放宽位置闸=修 Rathore2024@35% / Chen2020@44% 的 ack leak。)
    ② **markdown-heading based**(`^#{1,4}\\s*(?:编号)?\\s*<h>\\b`),不做正文内关键词搜杀
       (防 "this work was funded by..." 在正文被误当章节起点)。
    ③ ★ **body-marker 闸**(drill 2026-06-02b,§11 F22 a/a2):删段前扫 `_BODY_MARKERS`
       (`<table>` / md 表行 / `TABLE N` / `FIG N` / `$$`)。命中 = 待删段里混着 OCR 甩出的
       **floating 数据表/图注**(无自己标题,窄切口径保护不到)→ **保段不切**(跳到更靠后的纯
       ack 段)+ WARN 一行让 build-time 正文丢失可观测。Adriani2009(EOF 切删 TABLE I+FIG.5+
       13 公式块 -24%)、Cholis2020(窄切删 TABLE II 19 行拟合约束 -17%)即此闸救回。
    ④ ★ **窄切**:删到下一 markdown 标题为止(无下一标题=切到 EOF,已被闸①限定后段+闸②清过
       body);连续多个声明段(Acknowledgements 紧跟 Funding 紧跟 Data Availability)由递归收敛
       逐段剥净;⑤ char-count 护栏 `_MIN_KEEP_RATIO`:剥后 < 原文一半 → 判定误切,整体放弃、
       原样返回(防短文/OCR 异常把正文当尾巴砍掉);⑥ 无这些章节的论文 = no-op(原样返回)。

    NOT for pl's references — that stays in pl's `_strip_references` (KS 绝不改 pl 命名空间)."""
    if not text:
        return text
    half = len(text) * 0.5
    chosen: Optional[tuple[re.Match, Optional[re.Match]]] = None
    for cand in _ACK_HEADINGS.finditer(text):
        # Narrow cut ends at the NEXT markdown heading after this ack heading (so a heading-anchored
        # Appendix/Supplement survives); if none follows it is an unbounded cut-to-EOF.
        nxt = _ANY_HEADING.search(text, cand.end())
        # 闸①: bounded narrow cut honored at any position; unbounded EOF cut only in trailing half.
        if nxt is None and cand.start() <= half:
            continue
        seg = text[cand.start() : nxt.start()] if nxt is not None else text[cand.start() :]
        # 闸③: a removable segment that contains body markers is NOT pure ack/ref prose (floating
        # table/figure with no heading of its own) → keep it; try a later, cleaner ack section.
        if _BODY_MARKERS.search(seg):
            log.warning(
                "V7 _strip_acks: ack segment at %d%% (%d chars%s) holds body markers "
                "(table/fig/$$) → keep, not pure ack (would lose results body; SDD §6.9.7 F22 a/a2)",
                round(100 * cand.start() / len(text)),
                len(seg),
                ", cut-to-EOF" if nxt is None else "",
            )
            continue
        chosen = (cand, nxt)
        break  # earliest ack-class match that passes all gates
    if chosen is None:
        return text  # ⑥ no-op:无可安全删的尾部致谢段

    cand, nxt = chosen
    cut_start = cand.start()
    if nxt is not None:
        stripped = (text[:cut_start].rstrip() + "\n\n" + text[nxt.start() :].lstrip()).strip()
    else:
        stripped = text[:cut_start].rstrip()

    # Consecutive ack-class sections: if the head we kept still ends in an ack heading (or the
    # tail we re-joined leads with one), re-run until it stops changing (converges; still only
    # ever cuts ack sections, never an Appendix or a body-marker segment).
    if stripped != text:
        stripped = _strip_acks(stripped)

    # ★ 正文截断护栏(SDD §6.9.7 ④):护栏在**每个递归帧**比「本帧 stripped vs 本帧入参 text」。
    # 递归(line above)在本 guard 之前 return,故**最外层帧**比的是「全剥后结果 vs 原文」——
    # 用户面承诺(剥后 < 原文×0.5 → 放弃)在顶层成立,真实语料 0 触发。已知 benign false-keep:
    # 一篇极短且致谢占比 >50% 的合成/退化文档会在某帧触发护栏 → 放弃剥、保留致谢(把"合法的致谢
    # 主导短文"误判成误切)。真实论文绝不会 50%+ 是致谢,故语料里永不发生;方向保守(留致谢=多点
    # 噪声,而非砍正文=知识丢失),可接受(SDD §6.9.7 / §11 F22)。
    if len(stripped) < len(text) * _MIN_KEEP_RATIO:
        log.warning(
            "V7 _strip_acks would cut %d%% of doc (len %d → %d) → skip (likely mis-cut)",
            round(100 * (1 - len(stripped) / len(text))),
            len(text),
            len(stripped),
        )
        return text
    return stripped


def clean(text: str) -> str:
    """2a 删尾巴(SDD §6.1.c):`_strip_acks(_strip_references(text))`(组合)。
    ① `_strip_references`(pl 侧,**KS 不改**)剥 references/bibliography 段;
    ② V7 `_strip_acks`(KS 侧新增,§6.9.7)剥 acknowledgements/funding/利益声明等尾部章节。"""
    return _strip_acks(_strip_references(text))


def _dedup_key(cleaned_text: str) -> str:
    """与 LightRAG enqueue 同口径(sanitize_text_for_encoding 后)的内容指纹,用于批内预去重(F16)。

    LightRAG apipeline_enqueue_documents 按 sanitize_text_for_encoding(doc) 去重
    (lightrag.py:1404-1413):内容相同的第二个 doc_id 被静默丢弃、从不入 doc_status,
    其 ledger 行会永卡 processing。故 KS 在 enqueue 前用同一规范化口径自查碰撞。"""
    return hashlib.sha256(sanitize_text_for_encoding(cleaned_text).encode("utf-8")).hexdigest()


async def _existing_is_abstract(rag, did: str, st) -> bool:
    """Is the existing `did` doc an abstract-only doc (#144)? Read from its full_docs content (the
    text it was inserted with), falling back to doc_status content_summary. Anything that is not
    positively an abstract doc counts as full text — every doc built before #144 is full text."""
    doc = await rag.full_docs.get_by_id(did)
    content = doc.get("content") if isinstance(doc, dict) else None
    if content is None and isinstance(st, dict):
        content = st.get("content_summary")
    return is_abstract_text(content)


async def distill_batch(rag, items: list[tuple[PaperRecord, str]]) -> dict:
    """items = [(rec, fp)]. 两段式:批量 enqueue → 单次 process(§6.1)。"""
    inputs: list[str] = []
    ids: list[str] = []
    fpaths: list[str] = []
    queued: list[tuple[str, str]] = []     # (key, fp) — 批级异常回写 error 用(保留各自指纹,F17)
    seen_clean: dict[str, str] = {}        # dedup_key → 已入队的 key(F16 内容去重)
    counters = {"queued": 0, "meta": 0, "no_text": 0, "dup": 0, "errored": 0,
                "existing": 0, "blocked": 0, "purged_markers": 0,
                "abstract": 0, "replaced": 0, "replace_failed": 0}

    # #131 Bound 1: a doc_id doc_status already holds is never enqueued again (LightRAG would only
    # reject it and record a `dup-*` marker). One batched lookup for every would-be insert.
    want = [doc_id(rec.key) for rec, fp in items if fp != META]
    existing = await rag.aget_docs_by_ids(want) if want else {}

    for rec, fp in items:
        if fp == META:
            await ledger.upsert("paper", rec.key, doc_id=doc_id(rec.key), status="done_meta", fingerprint=META)
            counters["meta"] += 1
            continue
        want_abstract = is_abstract_fp(fp)
        st = existing.get(doc_id(rec.key))
        ds = doc_status_of(st)
        if ds is not None:
            if await _existing_is_abstract(rag, doc_id(rec.key), st) == want_abstract:
                # Adopt the existing doc: PROCESSED is done; anything else is still LightRAG's to
                # finish (or fail), so track it as processing and let reconcile_terminal record it.
                status = done_status_for(fp) if ds == DocStatus.PROCESSED else "processing"
                await ledger.upsert("paper", rec.key, doc_id=doc_id(rec.key), status=status, fingerprint=fp)
                counters["existing"] += 1
                log.info("distill_batch: %s already in doc_status (%s) → ledger %s, not re-enqueued",
                         rec.key, ds, status)
                continue
            # #144 Bound 2: the existing doc is the OTHER class (an abstract doc where full text is
            # wanted, or the reverse). Adopting it would record the wanted class while the graph
            # holds the other, so delete it and insert the wanted doc below. A delete that does not
            # land (busy pipeline / failure) leaves remove_one's pending_remove/error row → retried.
            r = await remove_one(rag, rec.key, delete_ledger=False)
            if r != "removed":
                counters["replace_failed"] += 1
                log.warning("distill_batch: %s holds a %s doc but %s is wanted; delete → %s, not enqueued",
                            rec.key, "full-text" if want_abstract else "abstract-only",
                            "abstract-only" if want_abstract else "full text", r)
                continue
            counters["replaced"] += 1
            log.info("distill_batch: %s replaced its %s doc (#144)", rec.key,
                     "full-text" if want_abstract else "abstract-only")
        if want_abstract:
            # One doc from title/authors/year/venue/ids/abstract; no extract, nothing to clean().
            cleaned = build_abstract_doc(rec)
            if not cleaned:
                # The abstract vanished since fingerprint() ran → plain metadata state.
                await ledger.upsert("paper", rec.key, doc_id=doc_id(rec.key), status="done_meta", fingerprint=META)
                counters["meta"] += 1
                continue
        else:
            text = read_extract_raw(rec)
            if not text or not text.strip():
                # 路径有但文件缺/空 → 当元数据态(待全文就绪)
                await ledger.upsert("paper", rec.key, doc_id=doc_id(rec.key), status="done_meta", fingerprint=META)
                counters["no_text"] += 1
                continue
            cleaned = clean(text)
        dk = _dedup_key(cleaned)
        if dk in seen_clean:
            # F16: 与本批先到的一篇正文(sanitize 后)字节相同 → LightRAG enqueue 会静默丢这个 doc_id。
            # 不留 processing(否则永卡):标 done_meta(dup),指向同内容已入队那篇。
            # ★指纹写 *真实 fp*(不是 META):本篇有全文,fingerprint(rec) 下轮恒返真实 hash;
            #   若这里写 META,下轮 diff(fp != META)会判 to_redistill 永久 churn(SDD §6.1 line 221 口径)。
            await ledger.upsert("paper", rec.key, doc_id=doc_id(rec.key), status="done_meta", fingerprint=fp)
            counters["dup"] += 1
            log.info("distill_batch: content-dup %s == %s → done_meta(dup)", rec.key, seen_clean[dk])
            continue
        blocker, purged = await _clear_insert_path(rag, rec.key)
        counters["purged_markers"] += purged
        if blocker is not None:
            # Another live row owns this file_path: an enqueue would be rejected as a duplicate and
            # leave a new marker. Record the failure instead (bounded by the #84 parking).
            await ledger.upsert("paper", rec.key, doc_id=doc_id(rec.key), status="error", fingerprint=fp)
            counters["blocked"] += 1
            log.warning("distill_batch: %s not enqueued — file_path %r is held by doc %s (#131)",
                        rec.key, normalize_document_file_path(file_path(rec.key)), blocker)
            continue
        seen_clean[dk] = rec.key
        inputs.append(cleaned)
        ids.append(doc_id(rec.key))
        fpaths.append(file_path(rec.key))
        await ledger.upsert("paper", rec.key, doc_id=doc_id(rec.key), status="processing", fingerprint=fp)
        queued.append((rec.key, fp))
        counters["queued"] += 1
        if want_abstract:
            counters["abstract"] += 1

    if inputs:
        try:
            # 批量入队(按 doc_id 去重),再单次 process —— Semaphore(max_parallel_insert) 在整批铺开。
            # ★前提: 调用点 pipeline idle(§6.6 删插互斥保证);否则 busy → request_pending 早返,本批仍 PENDING。
            await rag.apipeline_enqueue_documents(input=inputs, ids=ids, file_paths=fpaths)
            await rag.apipeline_process_enqueue_documents()
        except Exception as e:  # noqa: BLE001 — F17 批级兜底:不许击穿整 round
            # 批级 setup/校验/连接异常(enqueue 校验 ValueError、PG/Neo4j blip、pipeline 未 init):
            # 异常常发生在写 doc_status 之前,本批 key 会永卡 processing → 回写 error,下轮 diff(status) 重投。
            for key, fp in queued:
                await ledger.upsert("paper", key, doc_id=doc_id(key), status="error", fingerprint=fp)
            counters["errored"] = len(queued)
            counters["queued"] = 0
            log.warning("distill_batch enqueue/process failed (%d keys → error): %r", len(queued), e)
    # done 不在此写(enqueue 只回 track_id);reconcile 读 doc_status 终态回写 done/error(§6.6)
    return counters


def _result_status(r) -> Optional[str]:
    """DeletionResult.status(对象)或 dict 兜底。"""
    s = getattr(r, "status", None)
    if s is None and isinstance(r, dict):
        s = r.get("status")
    return s


async def remove_one(rag, key: str, *, delete_ledger: bool) -> str:
    """adelete_by_doc_id;4-way status(§6.1)。返回 removed|pending|error。
    delete_ledger=True(真删,REMOVE_PHASE)/ False(REDISTILL 先删旧 doc,留 ledger 行重插)。"""
    did = doc_id(key)
    r = await rag.adelete_by_doc_id(did)
    status = _result_status(r)
    if status in ("success", "not_found"):  # 已不在图 → 收口(404 不当失败,防无限重试)
        if delete_ledger:
            await ledger.delete("paper", key)
        return "removed"
    if status == "not_allowed":  # 403 pipeline busy
        await ledger.upsert("paper", key, doc_id=did, status="pending_remove")
        return "pending"
    await ledger.upsert("paper", key, doc_id=did, status="error")
    log.warning("remove %s failed: status=%s result=%r", key, status, r)
    return "error"


async def rollback_abstract_docs(rag, *, apply: bool) -> dict:
    """#144 Bound 4: delete every abstract-only doc and return its ledger row to done_meta.

    The class is every paper ledger row whose fingerprint is `ABSTRACT:` or whose status is
    done_abstract (any status: processing / error rows of the class are included). Dry run
    (apply=False) only counts. On apply, each row's doc is deleted (success / not_found both
    count) and the row written `done_meta` + `META`; a doc whose content is positively full text
    is never deleted (skipped and reported), and a delete LightRAG refuses leaves the row as it was.
    The caller must hold the pipeline idle — the CLI refuses while papervault.service runs.
    """
    led = await ledger.load("paper")
    rows = sorted((r for r in led.values()
                   if is_abstract_fp(r.fingerprint) or r.status == "done_abstract"),
                  key=lambda r: r.source_id)
    by_status: dict[str, int] = {}
    for r in rows:
        by_status[r.status] = by_status.get(r.status, 0) + 1
    result = {"abstract_rows": len(rows), "by_status": by_status,
              "keys": [r.source_id for r in rows], "reverted": 0,
              "failed": [], "skipped_not_abstract": []}
    if not apply:
        return result
    for r in rows:
        did = r.doc_id or doc_id(r.source_id)
        doc = await rag.full_docs.get_by_id(did)
        content = doc.get("content") if isinstance(doc, dict) else None
        if content and not is_abstract_text(content):
            result["skipped_not_abstract"].append(r.source_id)
            log.warning("rollback_abstract_docs: %s holds full text, not an abstract doc — kept",
                        r.source_id)
            continue
        status = _result_status(await rag.adelete_by_doc_id(did))
        if status not in ("success", "not_found"):
            result["failed"].append(r.source_id)
            log.warning("rollback_abstract_docs: delete %s → %s; ledger row kept", did, status)
            continue
        await ledger.upsert("paper", r.source_id, doc_id=did, status="done_meta", fingerprint=META)
        result["reverted"] += 1
    log.info("rollback_abstract_docs: %d reverted, %d failed, %d skipped (full text)",
             result["reverted"], len(result["failed"]), len(result["skipped_not_abstract"]))
    return result
