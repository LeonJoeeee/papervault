# KS v3 下游 baseline(2026-06-03,**49 题加宽 gold**)

> ⚠️ **RETIRED FRAME(2026-06-08)** — 本文档的全部数字(headline 0.7403 / 0.864、噪声底
> 0.0127、10 陷阱)都属于 **89-篇 test100 / l0_probe 坐标系**,与全量 prod 图**不可比、
> 不再是"要打败的数"**。全量语料的现行 baseline + 协议 + drill 结论见
> **`BASELINE_FULLCORPUS.md`**(headline 0.4371,trap 门 = judge refusal,噪声底重测中)。
> 本文档保留作历史与方法论参照。

**配置**:l0_probe,**89/100 篇**(11 篇 480s worker-timeout 失败),抽取 = V2 领域示例 + V3 排除 + V7 切致谢(当前生产配置),查询 = aquery_data mix top_k=40 / chunk_top_k=12 / **reranker(bge-reranker-v2-m3)on**。
**测法**:**49 道 gold(39 可答 + 10 陷阱)**,**3 跑**(synth temp=0.2),判官 = Claude subagent(非 MiMo)逐句/逐 nugget 分解打分,**1 seed/run × 3 run**(变体按同协议测才可比)。绝对分、固定集、跑间配对 —— 非 A/B 胜率。

> **gold 从 25→49**(`a9e4815`):旧 25 题里 18/21 饱和在 headline 1.0(仅 ~3 mover),`verdict` 区分不出变体。加了 **18 道难多论文综合题**(gold 3-6 篇、横跨不同主题簇,grounded 全文阅读 + 逐 nugget 自引 rationale)+ **6 道 grep 验证的离题陷阱**。难题让头条从 0.864 掉到 **0.740**,腾出 headroom + ~14 个 mover;10 陷阱把 trap 门粒度细到 0.1。

## 🎯 HEADLINE = **0.7403**(要打败的数)
逐题 `harmonic(paper_recall@12-distinct, nugget_recall)` 对 **39 答题**平均。
- per-run = [0.745, 0.726, 0.750],sd=0.010;**噪声底 run_mean_sd=0.0127**(旧 25 题集 0.0075 → 难题加大了跑间合成抖动;`headline.BASELINE_NOISE_FLOOR_RUN_MEAN_SD` 已更新为 0.0127),range=0.0243,n_q=39。
- → **判据(§8 drill 加固,`headline.verdict`):变体赢 ⟺ (i) headline 在 **movable 子集**上 BCa `ci_low>0` 且 `n_movers≥2`(R-tail-1:饱和题钉死全样本检验,只在真动的题上判显著)**且** (ii) 全集 `mean_delta > k·0.0127`(k=1)**且** (iii) jackknife 扛留一-mover **且** (iv) 6 红线门全可判无退步。

## 向量(3 跑均值)
| 维 | 值 | 读法 |
|---|---|---|
| **paper_recall@12-distinct** | **0.816** | 难题横跨多簇 → 该捞的没全捞回(留 headroom)|
| hit@12 | 0.974 | 97% 的题至少捞到 1 篇 gold ✅ |
| paper_recall@5 | 0.696 | |
| **nugget_recall** | **0.764** | 答案覆盖 76% 的"必答要点"(难综合题不饱和)|
| relevance | 0.860 | 切题 ✅ |
| faithfulness | 0.891 | 89% 论断有检索证据撑 ✅ |
| citation_support_precision | 0.873 | 87% 的 `[key]` 真支撑其句 |
| citation_recall | 0.855 | |
| gold_citation_recall | 0.816 | cited_papers 含 82% 的 gold |
| **hallucinated_rate** | **0.006** | 几乎不瞎编引用键 ✅(红线干净)|
| over_confidence_rate | 0.000 | 无"strong 却召回 0" ✅(红线干净)|
| phantom_rate | 0.420 | cited_papers 比正文实际引的宽(纪律项,非毒)|
| **trap_correct_refusal_rate**(10 陷阱)| **0.7 / 1.0 / 0.8 ≈ 0.83** | 比旧 4-陷阱的 0.58 升高:纯离题(tokamak/exoplanet)正确拒答拉高了率;近邻陷阱(reconnection/DM-xenon)仍偶尔硬编 |

## 画像:检索强 + 答案稳 + 引用干净;短板仍 = 近邻越界校准
- ✅ **强**:检索(hit 0.97)、答案质量(nugget 0.76 在难题上、relevance 0.86)、忠实(0.89)、引用卫生(瞎编 0.6%、over-confidence 0)。**红线门全干净。**
- 🟡 **短板 = 近邻越界拒答**:纯离题陷阱已能正确拒答,但**主题相邻**的陷阱(磁重联率、暗物质直接探测)仍能从图里捞 ≥5 实体 → 判 thin 不判 empty → ~一两成会硬编貌似合理但无据的答案。**对 executor 最危险**(自信幻觉引进真论文)→ 仍是下一步优化的查询侧靶子(产品决策)。
- **难题真不饱和**:18 难题里 14 个是 mover(recall 0.2-0.85),给了变体真正的发挥空间。

## 注记
- **既有 `slice20-q2` 的 gold 含 `Engelbrecht2022`**(11 篇建图超时之一)→ 该题召回封顶;对所有变体同等常量偏移,不偏向变体对比。
- **`Zhao2013a` 磁盘抽取张冠李戴**(另一篇 Zhao)→ 已排除出所有 gold_keys,建议重新入库。
- 噪声底仅 run 级(3 跑 × 1 judge-seed);run 间散布已含 synth/keyword/judge-seed 三源(H5)。
- 判官两轮:首轮 72 中 63 成功(newq_r3 尾部 9 个掉了),补判 `w...` 9/9 补齐;一个判官曾吐非法 `partial` verdict → `full_baseline._recompute` 归一到 neutral(保守)。

---

## §8 尺子 drill 终结(2026-06-03 确认轮)
加固后跑了**确认 drill**(2 个独立对抗 agent 直攻 `verdict`):**逻辑/fail-open** agent 逐行 trace + 跑测试,0 缝 0 异常,且确认尺子赢得动;**Goodhart** agent 判 trustworthy,仅抓出 1 条**休眠边界**——synth 散文"综合深度"没设门。**裁定:上游优化(synth 钉死当尺子)阶段 = 钻不动;综合深度由固定 synth 提示词决定、上游变体不碰 → 归后面下游 / synth 阶段再设门。** 详见 `KS_SDD.md §6.10 G` + `tests/test_headline.py`。

**红线门(6 门,两类检验)**:
- **ABSOLUTE 率门(noise≈0,mean-vs-mean)**:`hallucinated_rate`↓ · `over_confidence_rate`↓ · `gold_citation_recall`↑。
- **PAIRED+地板门(judge 有抖动)**:`citation_support_precision`↑ · `faithfulness`↑。
- **trap 率门**:`trap_correct_refusal_rate`(10 陷阱,baseline ~0.83)不得跌破 baseline。
- **fail-CLOSED 收口(H8)**:门缺位/输入空/缺题/地板=0 一律当回归;`win` 要求每门可判 ∧ 无门回归 ∧ (i)-(iii)。

**这就是优化循环(#5)要超越的 baseline:HEADLINE 0.740,噪声底 0.0127,6 门全干净。**
