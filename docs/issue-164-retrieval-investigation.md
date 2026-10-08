# Identifierless paper retrieval investigation — issue 164

Investigated 2026-10-08; [issue 164](https://github.com/LeonJoeeee/papervault/issues/164). This is the requested investigation and recommendation record. Runtime behavior and live application state were not changed. The code base inspected was `2810a04d4a1b4e0aef0508bff3fae42579878aaf`.

The 3,175 records are a mixed population. There are real retrieval routes, published abstracts, withdrawn current arXiv versions, and unresolved access/discovery failures. Missing DOI/arXiv fields do **not** establish that no tier ever attempted them. A blocked request or absent aggregator link does **not** establish that no full text exists.

## Census and interpretation

Read `/data/paper-vault/index.json` directly as JSON, without constructing `Library`, calling MCP tools, or invoking a downloader. The captured census at `2026-10-08T09:42:35.455385+00:00` has SHA-256 `9e3126a39e428a6dc7ca6568deb1b35d93a3da002ef3a09ab1fccd10db157899`. Filter: neither `doi` nor `arxiv_id` is truthy; group by `urllib.parse.urlsplit(url).hostname`, retaining malformed/empty values separately.

- Library: **26,415** records; no DOI: **6,678**; no DOI and no arXiv ID: **3,175**.
- Status in this population: **2,678 metadata_only, 495 ok, 2 extract_failed**. There are **493 nonempty PDF files** at the existing canonical file paths; file presence does not prove identity or extract quality. No file was modified.
- Corrected URL partition: **2,394 ADS + 505 CORE + 40 OpenAlex + 9 arXiv + 214 empty + 1 malformed + 12 miscellaneous = 3,175**. The issue's approximate 13 miscellaneous records become 12 when its 214 empty strings are distinguished from the additional malformed string.

| Group | Count | Stored URL / identifier | Observed file yield | Verdict and next route |
| --- | ---: | --- | --- | --- |
| `ui.adsabs.harvard.edu` | 2,394 | ADS bibcode record pages | 12/15 non-abstract PDF candidates; 5/5 abstract PDFs, only 1–2 pages | **Retrievable in part; needs work** for typed classification, HTML sources and unresolved records. [#168](https://github.com/LeonJoeeee/papervault/issues/168) |
| `core.ac.uk` | 505 | 494 `/download/<id>.pdf`, 11 `/download/pdf/<id>.pdf` | Stored URL: 0/30; repository originals: 3/10 metadata successes | **Needs work; retrievable originals demonstrated**. [#167](https://github.com/LeonJoeeee/papervault/issues/167); constrained known-file route [#166](https://github.com/LeonJoeeee/papervault/issues/166) |
| Truly empty URL | 214 | No URL; 130 CORE-source and 84 INSPIRE-source records | INSPIRE title sample: 4/5 complete PDFs | **Needs work; retrievable in part**, not absence. [#171](https://github.com/LeonJoeeee/papervault/issues/171) and [#167](https://github.com/LeonJoeeee/papervault/issues/167) |
| Malformed URL, no parsed host | 1 | Otsuka2020 embeds a URL after `全文連結` | 1/1 complete PDF, but it is a meeting program/poster listing | **Needs work**: normalize and verify the target; do not claim a paper fetched. [#166](https://github.com/LeonJoeeee/papervault/issues/166) |
| `openalex.org` | 40 | OpenAlex work records | Bounded source-location probes: 13/40 complete PDFs | **Retrievable in part; needs work** for failed/stale IDs and remaining locations. [#169](https://github.com/LeonJoeeee/papervault/issues/169) |
| `arxiv.org` | 9 | Abstract pages; 7 explicitly versioned | Current unversioned PDF: 0/9; historical v1: 9/9 | **Needs work**: all current versions are withdrawn; historical files are retrievable, with a policy decision. [#170](https://github.com/LeonJoeeee/papervault/issues/170) |
| Other hosts, detailed below | 12 | 9 file endpoints and 3 NASA handles | 4/12 complete PDFs | **Retrievable in part; needs work** for blocked/dead targets. [#166](https://github.com/LeonJoeeee/papervault/issues/166); CORE-origin recovery [#167](https://github.com/LeonJoeeee/papervault/issues/167) |

A `needs work` verdict preserves uncertainty rather than turning it into false absence. The report assigns a follow-up route to every group; it does not claim to have acquired, reopened, or established universal full-text absence for every individual record. Those operations are outside this issue's read-only bounds.

## Probe method and limits

Sample order is ascending SHA-256 of the UTF-8 citation key. CORE samples are stratified: first 20 `metadata_only`, then first 10 other-status records in that order; existing successes are controls. All nine arXiv records, all 12 miscellaneous records, all 40 OpenAlex IDs and all 2,394 ADS metadata records were considered. Five empty-URL INSPIRE `metadata_only` records were selected in the same order. ADS file strata were selected independently from the complete metadata census.

File probes use HTTP GET, follow normal redirects, inspect `%PDF-`, and stream the body to completion in memory, with a 32 MiB cap and a 20-second socket timeout. Non-PDF landing reads are capped at 128 KiB; API metadata at 4 MiB. A success requires a 2xx response, PDF signature and a complete body, not a HEAD result, URL suffix, MIME label, or metadata flag. For HTTP source links a separately recorded HTTPS form was also probed after a non-file result. Redirect resolution was permitted. arXiv diagnosis also compared URL forms and user agents, retaining repeated 404 observations; no failed result was hidden by selecting a later success. Landing resolution was limited to explicit citation-PDF metadata or PDF/download links, at most three discovered files. No PDFs were saved, and no database, vault, or service write was made.

The resulting fractions are **observed complete-file yields under these bounds**, not population success estimates or a complete proof of each file's bibliographic identity. Production must still use its existing `_verify_pdf_matches_metadata` gate. ADS abstract file checks additionally parsed page counts and first-page title words: all five matched the title words and were one or two pages. A meeting abstract or program PDF is not a separate full research paper.

There was no retry-to-success policy. CORE metadata stopped at the first 429: ten successful lookups, an eleventh rate-limited response, and nineteen sample records not queried for metadata. One ADS file investigation was canceled while `Liu2006a` remained incomplete; its result is not counted as fetched. Other completed ADS observations were captured in stdout before cancellation. A socket timeout is not an overall deadline; future operational probes should bound elapsed time as well as bytes.

## ADS: split the publication kinds before retiring anything

The 48 authenticated GET batches, up to 50 stored bibcodes each, returned metadata matching all **2,394** source records, including their duplicate bibcodes. There are **2,392 distinct bibcodes**: `2022cosp...44.1519D` and `2019AGUFMSH22A..01P` each occur in two library records; request/result multiplicities agree. Counts in this report are library records, not deduplicated works. The query shape was `GET https://api.adsabs.harvard.edu/v1/search/query?q=bibcode:("<bibcode>" OR ...)&fl=bibcode,title,doi,identifier,doctype,pub,esources,property&rows=100`. Credentials came from the existing configuration, never copied or published. No record supplied a DOI.

| ADS doctype | Count | PDF esource | Other esource only | No esource |
| --- | ---: | ---: | ---: | ---: |
| abstract | 1517 | 277 | 327 | 913 |
| inproceedings | 555 | 239 | 41 | 275 |
| article | 217 | 125 | 3 | 89 |
| phdthesis | 66 | 15 | 37 | 14 |
| techreport | 19 | 5 | 0 | 14 |
| proposal | 6 | 0 | 1 | 5 |
| inbook | 5 | 1 | 0 | 4 |
| proceedings | 4 | 0 | 0 | 4 |
| book | 3 | 0 | 0 | 3 |
| mastersthesis | 2 | 1 | 1 | 0 |
| **Total** | **2,394** | **663** | **410** | **1,321** |

ADS defines `doctype` and `esources` as publication type and available electronic-source kinds. Neither the non-refereed flag nor the mere existence of an abstract page means abstract-only. An electronic link and ADS-indexed searchable text are also different concepts. See [ADS search fields and properties](https://ui.adsabs.harvard.edu/help/search/) and [ADS FAQ](https://prod.adsabs.harvard.edu/help/faq/).

- **913 abstracts with no esource:** candidate retirement class for the indexed abstract publication, with reason `abstract_only_no_ads_esource` after checking publication identity and existing assets. **34 already have local PDFs**, so this metadata alone cannot prove no full text exists anywhere. The other **879** must not all be relabeled absent solely on this field. Preserve existing assets and investigate any separate full article as a distinct bibliographic item.
- **277 abstracts with PDF esources:** sampled PDFs really downloaded, but the five checked files are one or two pages. A confirmed abstract-only document can be retired from the full-paper hunt with that explicit reason; keep its metadata or abstract document. Do not count those five as full-paper recovery. Do not extrapolate page-count evidence to all 277.
- **327 abstracts with other esources:** inspect the publisher/author HTML before deciding whether it is only an abstract, a poster, or fuller content. **Needs work**, not a retry-everything class.
- **877 non-abstract records:** 386 PDF candidates, 83 other-source-only, 408 without an esource. These include real articles, proceedings, theses and reports. **Needs work** where the bounded probe failed or no link is listed; there is no basis for blanket retirement.

File sample: **12/15 (80%)** non-abstract PDF candidates completed, two failed, one remained incomplete; **5/5 (100%)** abstract PDF candidates completed; **0/10** abstract/no-source and **0/5** non-abstract/no-source legacy probes completed. The negative legacy probes returned 403/timeouts, which are access failures, not proof of absence. The 12/15 denominator retains the incomplete record.

| Stratum | Key / ADS bibcode | Observed outcome |
| --- | --- | --- |
| abstract_no_source | `Wilber2004` / `2004AGUFMSH12A..01W` | TimeoutError: no complete PDF |
| abstract_no_source | `Richardson2013b` / `2013AGUFMSH24A..05R` | TimeoutError: no complete PDF |
| abstract_no_source | `Ghanbari2020` / `2020AGUFMSH0090018G` | TimeoutError: no complete PDF |
| abstract_no_source | `Wiedenbeck2016` / `2016AGUFMSH31B2596W` | 403: no complete PDF |
| abstract_no_source | `Lario2007a` / `2007AGUFMSH33A1077L` | 403: no complete PDF |
| abstract_no_source | `Jones2020` / `2020AGUFMP047.0009J` | 403: no complete PDF |
| abstract_no_source | `Hu2019` / `2019shin.confE.221H` | 403: no complete PDF |
| abstract_no_source | `Gomez2019a` / `2019EGUGA..2111017G` | 403: no complete PDF |
| abstract_no_source | `Jeunon2021` / `2021AGUFMSH33A..05J` | 403: no complete PDF |
| abstract_pdf | `Boden2014` / `2014EGUGA..16.5829B` | 200: complete PDF, 1 pages |
| abstract_pdf | `Cholis2021` / `2021cosp...43E1315C` | 200: complete PDF, 1 pages |
| abstract_pdf | `Guo2021` / `2021cosp...43E2418G` | 200: complete PDF, 2 pages |
| abstract_pdf | `Strauss2021a` / `2021cosp...43E.905S` | 200: complete PDF, 1 pages |
| abstract_pdf | `Ferreira2010` / `2010cosp...38.1640F` | 200: complete PDF, 1 pages |
| other_pdf | `Mutschler2019` / `2019amos.confE...8M` | 200: complete PDF, 11 pages |
| abstract_no_source | `Lario2011` / `2011AGUFMSH33D..02L` | TimeoutError: no complete PDF |
| other_pdf | `Heber1996a` / `1996A&A...316..538H` | 200: complete PDF, 9 pages |
| other_pdf | `Pyle1999` / `1999ICRC....7..386P` | TimeoutError: no complete PDF; 200: complete PDF, 4 pages |
| other_pdf | `Dorman2008a` / `2008ICRC....1..175D` | 200: complete PDF, 4 pages |
| other_pdf | `Duvernois1996` / `1996A&A...316..555D` | 200: complete PDF, 9 pages |
| other_pdf | `Potgieter1999a` / `1999ICRC....7...57P` | 200: complete PDF, 4 pages |
| other_pdf | `Ding2024b` / `2024PhDT........16D` | 200: no complete PDF; URLError: no complete PDF |
| other_pdf | `Dikpati1994` / `1994A&A...291..975D` | TimeoutError: no complete PDF; 200: complete PDF, 15 pages |
| other_pdf | `Tremblay2019` / `2019mlhp.confR..59T` | 200: complete PDF, 65 pages |
| other_pdf | `Florinski2008b` / `2008ASPC..385...18F` | 200: complete PDF, 7 pages |
| other_pdf | `Joshi2025` / `2025amos.conf..103J` | 404: no complete PDF; 403: no complete PDF |
| other_pdf | `von1997` / `1997ICRC....5...81V` | 200: complete PDF, 4 pages |
| other_pdf | `Fahr2000` / `2000A&A...357..268F` | 200: complete PDF, 15 pages |
| other_no_source | `Francis1974` / `1974PhDT.........7F` | 403: no complete PDF |
| other_pdf | `Ruderman1993` / `1993A&A...275..635R` | TimeoutError: no complete PDF; 200: complete PDF, 10 pages |
| other_no_source | `Sahu2026` / `2026cosp...46.1241S` | 403: no complete PDF |
| other_no_source | `Kucharek2022` / `2022cosp...44.1513K` | 403: no complete PDF |
| other_no_source | `Zubrin1993` / `1993JBIS...46R...3Z` | 403: no complete PDF |
| other_no_source | `von2014` / `2014lws..prop...94V` | TimeoutError: no complete PDF |
| other_pdf | `Liu2006a` / `2006PhDT........35L` | Incomplete; canceled, not counted as success |

Route and cost: [#168](https://github.com/LeonJoeeee/papervault/issues/168) should extract/validate a bibcode from the stored ADS URL (or source-qualified record identity), query the existing ADS member, and use its gateway/legacy path with the normal cascade verifier. A token is already configured on this host. A batched availability census costs metadata requests (48 in this audit); a usable candidate costs a gateway GET and redirects, with one bounded legacy fallback. Do not issue legacy requests for every abstract on every retry. No new source or separate pipeline is necessary.

## CORE: a download-shaped URL is not a cheap win until it answers

All 505 URLs have a PDF-download shape. The **30-record** sample returned Cloudflare challenge HTML with HTTP **403 for 30/30**, including ten previously successful controls: **0/30 (0%)** directly fetchable on this run. This is an access block, not missing credentials or proof that the papers do not exist. The group contains 293 `ok`, 211 `metadata_only`, and one `extract_failed` record; 293 nonempty PDFs already exist.

| Stratum | Sample citation keys, in selection order | Stored-URL outcome |
| --- | --- | --- |
| metadata_only (20) | `APnd`, `Ragot2020`, `Weihsnd`, `Li2023m`, `Freiherr2020`, `Hanzelka2022`, `Cukierman2020`, `Imgrund2016`, `Wilson2013`, `Wu2009`, `Pacioreknd`, `CHMIELOWIEC2024`, `Armstrong2014`, `Yang2025g`, `Hadandc`, `Charous2023`, `Aboudarham2020`, `Bamberger2007`, `Angelopoulos2008a`, `Mirand` | All HTTP 403 Cloudflare challenges; 0 complete PDFs |
| other-status controls (10) | `Vaisanen2023a`, `Mertsch2023`, `Dongari2010a`, `Masi2006`, `Russell2012`, `Borrajo1998`, `Rahmanifard2019`, `Barker2007`, `Davidson2021`, `Stone1995a` | All HTTP 403 Cloudflare challenges; 0 complete PDFs |

For the first ten selected records, `GET https://api.core.ac.uk/v3/outputs/<stored-output-id>` returned source metadata; the next, Pacioreknd, returned **429**, and metadata requests stopped. Seven of the ten say `fulltextStatus=enabled`, three `disabled`; disabled metadata can still contain origin links. At most two distinct candidates per metadata record were probed, including the API download URL when present. **3/10 (30%)** yielded a complete origin PDF. This is a lower bound for this small, terminal-record subsample; additional listed origins were left unprobed.

| Key | CORE output ID | Metadata state | Origin outcome |
| --- | --- | --- | --- |
| `APnd` | 475653135 | disabled | No complete PDF in bounded candidates |
| `Ragot2020` | 357359140 | enabled | No complete PDF in bounded candidates |
| `Weihsnd` | 6605866 | enabled | No complete PDF in bounded candidates |
| `Li2023m` | 568416448 | disabled | Complete PDF: 3748907 bytes via arxiv.org |
| `Freiherr2020` | 389617828 | enabled | No complete PDF in bounded candidates |
| `Hanzelka2022` | 528000741 | disabled | No complete PDF in bounded candidates |
| `Cukierman2020` | 357257381 | enabled | Complete PDF: 529049 bytes via physics.wm.edu |
| `Imgrund2016` | 79056167 | enabled | No complete PDF in bounded candidates |
| `Wilson2013` | 13120229 | enabled | Complete PDF: 6390910 bytes via research-repository.st-andrews.ac.uk |
| `Wu2009` | 71327306 | enabled | No complete PDF in bounded candidates |
| `Pacioreknd` | 6305146 | no metadata | 429; stopped; no origin probe |

The successful original routes were Li2023m → [arXiv 2306.12749](https://arxiv.org/abs/2306.12749), Cukierman2020 → William & Mary's author thesis PDF, and Wilson2013 → the St Andrews thesis PDF found from its handle landing page. API titles/source links identify candidates; production identity checking remains required.

Route and cost: [#167](https://github.com/LeonJoeeee/papervault/issues/167) should resolve source-qualified output IDs and prioritize `sourceFulltextUrls` within the existing CORE member, respecting 429 and bounding candidate/landing work. Cost is one metadata call per unresolved output plus a few source/landing/file GETs; service quotas may dominate. The API key is configured. Do not fill `url_overrides.json` with 505 machine-derived entries, add CAPTCHA retries, or treat `disabled` as universal absence. CORE's [API service](https://core.ac.uk/services/api) and [FAQ](https://core.ac.uk/faq) describe provider full texts and rate limits; this audit's 429 is the observed quota constraint, not a claimed fixed requests-per-second limit.

## OpenAlex: resolve the work, not just a DOI

All **40** work URLs were queried via `GET https://api.openalex.org/works/W...`: **37 metadata 200s, 3 404s** (Zintgraf2018/W2952003143, Plotnikov2025/W7118051217, Rouillard2025/W3098668023). The 37 successful responses contain **29 OA flags, 25 records with location PDF URLs, and 24 cached-PDF flags**. Only Kasim2021 now exposes a DOI, `10.1088/2632-2153/ac3ffa`; the other 36 do not.

Up to two distinct location/landing candidates per record were considered, with the bounded HTML link resolution described above. Counting all 40 records, including stale IDs/no URL, **13/40 (32.5%)** served complete PDFs; **7 of those 13** are `metadata_only`. A field named `pdf_url` sometimes contains a handle/record page, so those responses are not automatically file successes. The group has 18 existing PDFs, which must be preserved.

| Key | OpenAlex work ID | File probe outcome |
| --- | --- | --- |
| `Perko1987` | `W1625918965` | 200; no complete PDF observed |
| `Pei2007` | `W1657115388` | 200; HTTPS 200; no complete PDF observed; TimeoutError; HTTPS URLError; no complete PDF observed |
| `JohnstonHollitt2003` | `W2760744656` | 404; HTTPS 404; no complete PDF observed; 404; no complete PDF observed |
| `Nel2015` | `W2340140072` | Complete PDF, 5570284 bytes via repository.nwu.ac.za |
| `Palmerio2019` | `W2972185955` | 200; HTTPS 200; no complete PDF observed |
| `Steenkamp1995` | `W411256283` | Complete PDF, 8079281 bytes via repository.nwu.ac.za |
| `Kipf2018` | `W2963464736` | Complete PDF, 2221875 bytes via pure.uva.nl |
| `Farwa2026` | `W7140958148` | 403; no complete PDF observed; 403; no complete PDF observed |
| `Pioch2012` | `W2746884899` | 200; no complete PDF observed; 200; HTTPS 200; no complete PDF observed |
| `Nndanganeni2016a` | `W2581018661` | Complete PDF, 5161204 bytes via repository.nwu.ac.za |
| `Mukhoti2021b` | `W3169238351` | Complete PDF, 3058316 bytes via arxiv.org |
| `Kota2003` | `W1649363052` | Complete PDF, 181009 bytes via www-rccn.icrr.u-tokyo.ac.jp |
| `Zintgraf2018` | `W2952003143` | No candidate file URL; metadata 404 |
| `Vos2011` | `W1864630714` | Complete PDF, 6067347 bytes via repository.nwu.ac.za |
| `Hoeksema1984` | `W2120825857` | 403; HTTPS 403; no complete PDF observed; 403; no complete PDF observed |
| `Mukhoti2021` | `W3131119906` | Complete PDF, 3058316 bytes via export.arxiv.org |
| `Bader2008` | `W1581321939` | 200; linked files 403; no complete PDF observed; 200; no complete PDF observed |
| `Airapetian2019` | `W3103168219` | 200; linked files TimeoutError; no complete PDF observed; 502; no complete PDF observed |
| `Denehy1974` | `W784210694` | 403; HTTPS 403; no complete PDF observed; 403; no complete PDF observed |
| `Laufer1953` | `W1973799148` | 403; HTTPS 403; no complete PDF observed; 403; HTTPS 403; no complete PDF observed |
| `Sternal2010` | `W2614986028` | 200; no complete PDF observed; 200; no complete PDF observed |
| `Burlaga1983a` | `W1538429658` | 403; HTTPS 403; no complete PDF observed; 403; no complete PDF observed |
| `Kanevski2009` | `W1224841963` | RemoteDisconnected; HTTPS URLError; no complete PDF observed; TimeoutError; HTTPS URLError; no complete PDF observed |
| `Stone2003` | `W1632098127` | Complete PDF, 159972 bytes via authors.library.caltech.edu |
| `Kasim2021` | `W4200095806` | Complete PDF, 2432130 bytes via idus.us.es |
| `Axford1972` | `W1487982788` | 403; HTTPS 403; no complete PDF observed |
| `Plotnikov2025` | `W7118051217` | No candidate file URL; metadata 404 |
| `Anon1984` | `W2986810904` | 405; no complete PDF observed; 200; HTTPS 200; no complete PDF observed |
| `Calderhead2008` | `W2104911495` | URLError; HTTPS 429; no complete PDF observed; 404; HTTPS 404; no complete PDF observed |
| `Vanat2017` | `W2741448117` | 200; HTTPS 200; no complete PDF observed; 200; no complete PDF observed |
| `Chenette1992a` | `W243295392` | 200; HTTPS 200; no complete PDF observed |
| `Sandroos2010` | `W1638255502` | 200; no complete PDF observed |
| `Rogallo1981` | `W1528581602` | 403; HTTPS 403; no complete PDF observed; 429; HTTPS 429; no complete PDF observed |
| `Bowen2019` | `W3009676986` | 403; no complete PDF observed; 403; no complete PDF observed |
| `Ivascenko2016` | `W2745434303` | Complete PDF, 7561044 bytes via repository.nwu.ac.za |
| `Gieseler2018` | `W3195566814` | 200; no complete PDF observed; 403; no complete PDF observed |
| `Rouillard2025` | `W3098668023` | No candidate file URL; metadata 404 |
| `Prinsloo2016` | `W2527984346` | Complete PDF, 9162095 bytes via repository.nwu.ac.za |
| `Anon2016` | `W3103847480` | Complete PDF, 1429035 bytes via soar-ir.repo.nii.ac.jp |
| `McKee2020` | `W3034610922` | TimeoutError; HTTPS 500; no complete PDF observed; 200; no complete PDF observed |

Route and cost: [#169](https://github.com/LeonJoeeee/papervault/issues/169) should use the stored work ID in the existing OpenAlex member, inspect the best OA location and other location/identifier fields, then use normal verification/bookkeeping. This audit made metadata calls without an API key. Cost is one metadata lookup plus bounded file/landing requests; cached content is a separate authenticated/costed option and was **not probed** here. Do not infer absence from a closed OA flag or a stale work ID. See [OpenAlex locations](https://help.openalex.org/data/locations/) and [cached full text](https://help.openalex.org/access/fulltext/).

## arXiv: the nine IDs are real, and their latest versions are withdrawn

All nine HTTPS abstract pages returned 200 and explicitly identified withdrawal. All nine unversioned `https://arxiv.org/pdf/<id>` requests returned 404. For the seven stored URLs naming v2/v3, the named PDF also returned 404. In contrast, **9/9 (100%) historical v1 PDFs** streamed completely. The two records with unversioned URLs were probed explicitly at v1; that is historical retrieval, not retrieval of the latest version. Two records already have local PDFs.

| Key | ID present in stored URL | Current PDF | Historical v1 bytes, complete |
| --- | --- | --- | ---: |
| `Catt2026` | `2603.20546v2` | 404; withdrawn record | 203073 |
| `Ren2013` | `1311.2658v3` | 404; withdrawn record | 190781 |
| `Yun2023` | `2301.06732` | 404; withdrawn record | 12914497 |
| `Masso2012` | `1205.1956v2` | 404; withdrawn record | 385614 |
| `Ren2013a` | `1312.4250v2` | 404; withdrawn record | 152477 |
| `Yazdanpanah2016` | `1612.04490v2` | 404; withdrawn record | 918615 |
| `Plainaki2009` | `0911.5676v2` | 404; withdrawn record | 734728 |
| `Wu2022a` | `2208.04681v2` | 404; withdrawn record | 772153 |
| `Giannakisnd` | `1612.07272` | 404; withdrawn record | 535944 |

Source trace: `_try_arxiv` strips `vN`, requests the current PDF, and clears `paper.arxiv_id` on any PDF 404. Consequently, restoring these nine URL IDs alone would recreate the 404/clear loop. This explains why the proposed cheap backfill is insufficient; it does not prove which historical action originally blanked each field. arXiv's [withdrawal documentation](https://info.arxiv.org/help/withdraw.html) confirms that older versions remain accessible while the withdrawn version has no PDF download.

Route and cost: [#170](https://github.com/LeonJoeeee/papervault/issues/170) should preserve valid record identity and distinguish withdrawn-current/no-current-PDF from nonexistent IDs. **Recommend retiring current-version acquisition with a withdrawal reason**, while retaining valid metadata; an explicitly labeled historical-version route requires an owner decision before implementation. Do not silently substitute v1 or declare the record's full text nonexistent. Cost is a record/history lookup plus one selected-version PDF request; version and withdrawal provenance must survive citation/serving. No new source is needed.

## Empty and malformed URLs: source metadata still matters

The **214** genuinely URL-empty records include **130 CORE and 84 INSPIRE**. Together with Otsuka2020's malformed string, the parser's `(empty)` host bucket has **215** records, 177 `metadata_only`, 37 `ok`, and one `extract_failed`; 38 nonempty PDFs already exist. 213/215 have titles at least 20 characters long.

Five deterministic terminal INSPIRE titles were queried with `GET https://inspirehep.net/api/literature?q=title:"<title>"&size=3`. Each returned one exact title match, with no DOI/arXiv metadata; **4/5 (80%)** source document URLs served complete PDFs. This denominator is the INSPIRE sample, not all 214 empty URLs. The URL-empty CORE subset was not independently API-probed after the CORE 429; retained source-qualified output IDs and existing title routes need the CORE follow-up.

| Key | INSPIRE record | Title match | Document probe |
| --- | --- | --- | --- |
| `Ambrosi2013` | [1412451](https://inspirehep.net/literature/1412451) | Exact | Complete PDF, 303912 bytes |
| `Komorind` | [1371172](https://inspirehep.net/literature/1371172) | Exact | Complete PDF, 289714 bytes |
| `Fiorino2015` | [1830543](https://inspirehep.net/literature/1830543) | Exact | No document link; not proof of absence |
| `Wojdand` | [1826361](https://inspirehep.net/literature/1826361) | Exact | Complete PDF, 7006701 bytes |
| `Ferrandnd` | [1371238](https://inspirehep.net/literature/1371238) | Exact | Complete PDF, 163939 bytes |

Otsuka2020 stores `全文連結` followed by [VERSIM_Program_poster.pdf](http://pcwave.rish.kyoto-u.ac.jp/versim/data/VERSIM_Program_poster.pdf). Extracting that URL yielded a complete **291068-byte** PDF, and inspection of the 12-page document confirmed a table of poster titles, speakers and sessions rather than a paper body. Its relevance to a particular full paper remains unestablished. Normalize only with identity checks; a valid program file is not evidence of an available paper body.

Route and cost: [#171](https://github.com/LeonJoeeee/papervault/issues/171) should retain source record IDs/document links in the existing ingestion/enrichment path and make the existing INSPIRE cascade member accept that identity. The current search adapter omits `documents` and source record IDs; its URL is synthesized only from DOI/arXiv, so information loss is observable in source. Cost: an existing source lookup, or a bounded title match for historical records, then a file GET; no INSPIRE credential was needed. CORE-source empty records use [#167](https://github.com/LeonJoeeee/papervault/issues/167); malformed known URLs use [#166](https://github.com/LeonJoeeee/papervault/issues/166). Existing arXiv-title, CORE-title and web-title routes are already available where their prerequisites hold. **Do not retire the empty-URL class as having no full text.**

## Every miscellaneous host

Every one of the **12** records was probed. Four complete PDFs were observed: **4/12 (33.3%)** overall, or **4/9 (44.4%)** among the nine stored file endpoints. Two successes (Caltech and White Rose) already have PDFs; Lancaster and Warwick are `ok`/`firecrawl` records without canonical PDFs. Their reachable PDFs would complete the file acquisition rather than recover a metadata-only record. Three handles resolve to NASA NTRS landing pages, not PDF files.

| Host | Host count | Key / stored endpoint | Result | Verdict / route |
| --- | ---: | --- | --- | --- |
| `authors.library.caltech.edu` | 1 | `Murphy1995` / `/46479/1/1995-43.pdf` | PDF 200; 2453089 bytes, complete | Retrievable; [#166](https://github.com/LeonJoeeee/papervault/issues/166) |
| `hdl.handle.net` | 3 | `Jokipii1985` / `/2060/19850026519 → NTRS citation landing page` | 403; HTTPS 403; no complete PDF observed | Needs work; [#167](https://github.com/LeonJoeeee/papervault/issues/167) |
| `hdl.handle.net` | — | `Vainikka1985` / `/2060/19850026748 → NTRS citation landing page` | 403; HTTPS 403; no complete PDF observed | Needs work; [#167](https://github.com/LeonJoeeee/papervault/issues/167) |
| `hdl.handle.net` | — | `Roche2009` / `/2060/20090020463 → NTRS citation landing page` | 403; HTTPS 403; no complete PDF observed | Needs work; [#167](https://github.com/LeonJoeeee/papervault/issues/167) |
| `repository.fit.edu` | 1 | `Kosar2015` / `/cgi/viewcontent.cgi?article=1470&context=etd` | 403; no complete PDF observed | Needs work; [#167](https://github.com/LeonJoeeee/papervault/issues/167) |
| `umpir.ump.edu.my` | 1 | `Noordin2020` / `/id/eprint/36293/1/Characteristic%20of%20electrical%20signal%20of%20transient%20luminous%20events%20%28TLES%29%20in%20Pekan%20Pahang.wm.pdf` | TimeoutError; HTTPS URLError; no complete PDF observed | Needs work; [#167](https://github.com/LeonJoeeee/papervault/issues/167) |
| `ora.uniurb.it` | 1 | `Benella2020a` / `/bitstream/11576/2673737/1/phd_uniurb_280117.pdf` | 403; no complete PDF observed | Needs work; [#167](https://github.com/LeonJoeeee/papervault/issues/167) |
| `citeseerx.ist.psu.edu` | 1 | `Paris2020` / `/viewdoc/download?doi=10.1.1.1051.1229&rep=rep1&type=pdf` (stored URL also has a legacy session path parameter) | 404; HTTPS 404; no complete PDF observed | Needs work; [#167](https://github.com/LeonJoeeee/papervault/issues/167) |
| `eprints.lancs.ac.uk` | 1 | `Pulkkinen2006` / `/6681/1/art_834.pdf` | PDF 200; 352725 bytes, complete | Retrievable; [#166](https://github.com/LeonJoeeee/papervault/issues/166) |
| `eprints.whiterose.ac.uk` | 1 | `Utznd` / `/173753/1/2104.10126v1.pdf` | PDF 200; 3111298 bytes, complete | Retrievable; [#166](https://github.com/LeonJoeeee/papervault/issues/166) |
| `wrap.warwick.ac.uk` | 1 | `Parmarnd` / `/56873/7/WRAP_THESIS_Parmar_2012.pdf` | PDF 200; 10970705 bytes, complete | Retrievable; [#166](https://github.com/LeonJoeeee/papervault/issues/166) |
| `trace.tennessee.edu` | 1 | `Ademola2023` / `/cgi/viewcontent.cgi?article=9817&context=utk_graddiss` | 404; no complete PDF observed | Needs work; [#167](https://github.com/LeonJoeeee/papervault/issues/167) |

Cost: successful stored-file routes need one GET plus normal redirects and identity verification, with no new credential. A handle needs landing/API resolution before a file request. NTRS returned 403, Caltech redirected to a PDF-bearing object, UMP timed out, CiteSeer/TRACE returned 404, and Florida Tech/Urbino returned 403. These results justify **needs work**, not retirement. Recheck stale endpoints through the existing CORE source record/origin links before commissioning host-specific scrapers. None of these hosts has evidence here supporting a universal no-full-text verdict.

## Recommended cascade changes and accounting

Keep one pipeline. A constrained early known-file URL path in [#166](https://github.com/LeonJoeeee/papervault/issues/166) should sit beside the operator override/preprint paths, retaining downstream byte/identity verification. The existing override map is DOI/arXiv-keyed and manually maintained; it cannot represent this entire identifierless set as currently implemented. Machine-derived source URLs belong to the paper/source metadata and source resolver, not a second operational override registry.

Reuse the existing `oa_aggregators` members for CORE [#167](https://github.com/LeonJoeeee/papervault/issues/167) and OpenAlex [#169](https://github.com/LeonJoeeee/papervault/issues/169), and `domain_aggregators` members for ADS [#168](https://github.com/LeonJoeeee/papervault/issues/168) and INSPIRE [#171](https://github.com/LeonJoeeee/papervault/issues/171). Correct arXiv withdrawal/identity behavior in its existing tier under [#170](https://github.com/LeonJoeeee/papervault/issues/170). These are proposed follow-ups, not an accepted new architecture or changes implemented by this investigation. Serialize any implementation lanes that overlap `download.py`, as [issue 160](https://github.com/LeonJoeeee/papervault/issues/160) requires.

The base already supports CORE title fallback, `arxiv_by_title`, `web_search`, URL-aware Sci-Hub when enabled, and URL-aware firecrawl fallback. The issue body's universal identifier-guard description is inaccurate for current code. Targeted source-ID/known-URL gaps remain real. Classification in `services/classify.py` still makes `metadata_only` terminal; a new route alone will not automatically revisit old records.

After [PR 159](https://github.com/LeonJoeeee/papervault/pull/159), `download_miss` records an attempted tier failure and `download_skip` records an unmet no-network prerequisite. Any follow-up must keep the tier prerequisite checks aligned with its new URL/source-ID eligibility, including concurrent group eligibility. Evidence-based retirement needs a durable reason separate from a failed HTTP request; no new status schema is accepted here. A later bounded operator audit/retry must exclude existing assets and confirmed abstract-only/withdrawn-current cases, preserve unknown outcomes and rate-limit failures, and carry explicit live authorization. [Issue 157](https://github.com/LeonJoeeee/papervault/issues/157) covers the DOI-bearing retry only, so it is not authorization to reopen this group.

The investigation recommends retiring **confirmed published-abstract-only documents** from the separate-full-paper hunt and **withdrawn current versions** from current-version acquisition, with explicit reasons. It recommends **no blanket host retirement** and **no no-full-text label based solely on 403, 404, timeout, empty URL, absent esource, disabled CORE content, or closed OA metadata**. Resolving every remaining individual unknown requires the bounded per-source follow-ups; the sampling results cannot establish global absence.

## Reproduction and durable evidence

The census can be repeated without application imports:

```python
import json
from collections import Counter
from urllib.parse import urlsplit
with open('/data/paper-vault/index.json') as stream:
    papers = json.load(stream)['papers']
rows = [p for p in papers.values() if not p.get('doi') and not p.get('arxiv_id')]
print(len(papers), len(rows))
print(Counter(urlsplit(p.get('url') or '').hostname or '(empty)' for p in rows))
print(Counter(p.get('download_status') for p in rows))
```

The query shapes, sample keys/source IDs, failed observations, complete byte counts where captured, ADS doctype matrix, and fractions above are the durable evidence. The main source probes started at 09:42 UTC; the stalled ADS probe was canceled with its incomplete result retained. A later read-only inspection verified that the malformed URL points to a poster program. Temporary harnesses, request metadata and logs live in the single task scratch `/tmp/papervault-164-GXGWn5`; no unique deliverable depends on their retention. No source PDF, abstract body, credential, signed download URL, whole production index export, or runtime state is committed.
