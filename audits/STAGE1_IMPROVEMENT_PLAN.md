# Stage 1 improvement roadmap

This plan follows the professor's [official brief](../CLT_AI_Trend_Project.pdf) and the evidence in [PROFESSOR_REQUIREMENTS.md](./PROFESSOR_REQUIREMENTS.md). Costs are estimates; one full notebook run took about 21 minutes on the available CPU.

**Nothing below requires, or should be done by, fabricating labels, scores, results or contributions.** Tasks needing the student are marked **Human input**.

## Completed in this iteration (evidence-based)

| Earlier item | Result | Evidence |
|---|---|---|
| P0.1 Stale "brief unavailable" statement | Done | Notebook introduction now cites the official PDF; a structural check fails if the stale text returns. |
| P0.2 Contribution disclosure | Done (individual) | Introduction: individual statement for Leon Bytyqi; no placeholder table. |
| (new) Mandatory AI disclosure | Done | "GitHub Copilot and AI language models drafted the code…" plus separate student responsibilities; mirrored in `stage1_metadata.json`. |
| P0.3 Sentiment/stance decision | Documented | Notebook §4.1; R8 = NOT APPLICABLE, **confirmation still needed** (P0.2 below). |
| P0.4 Topic-over-time interpretation and key insights | Done | §5.2–5.4, §11.1; claims asserted against executed tables. |
| P0.5 Acronym/entity consistency | Done, with stated limits | §5–6; `entity_normalization_report.csv`; lexicon split of distinct products/units; generic-term stop list; publisher-dominance flag. |
| P1.3 Topic stability | Done | 10 fixed-seed resamples × K∈{3,4,6}; K=3 most reproducible (mean ARI 0.83); K=6 contains an unrecoverable topic; lemma/acronym vs surface view ARI 0.85. |
| P1.4 Temporal denominators and uncertainty | Done | `topic_monthly_counts.csv`, `topic_temporal_trends.csv`, `topic_source_concentration.csv`, `bulk_ingest_publisher_months.csv`; two bulk publisher-months change effect sizes but not directions. |
| P1.5 Graph sensitivity | Done | `graph_layers_summary.csv`, `graph_threshold_sensitivity.csv`, `graph_node_metrics.csv`, plus emergence tables; 8,183 inventory nodes vs 7,291 connected vs 235 in candidate edges vs 0 validated; nested-mention pairs (e.g., Google inside "Google Cloud") no longer count as co-mentions; emerging-entity/pair rankings exclude publisher-dominated entities (borderline case: GitLab at 89.8% from one publisher, just under the predeclared 90% rule, remains in the entity ranking). |
| (new) Review workflow | Done | `outputs/evaluations/REVIEW_GUIDE.md`; labelled worksheets are preserved and scored (stratum-weighted selection estimator with Wilson intervals). |
| (new) Pinned environment | Done | `requirements-stage1.txt` (tested versions). |

## P0 — Blockers and decisions (student)

### P0.1 R11 (LangExtract or a comparable structured extraction approach) — DONE, pilot scope
- **Result:** LangExtract with OpenAI `gpt-4o-mini` was applied to a fixed-seed 50-article pilot (notebook §7.1; `llm_relation_pilot.csv`, `llm_entity_pilot.csv`, `llm_pilot_summary.csv`, `llm_vs_template_candidates.csv`). 41 of 65 relation extractions passed source-grounding validation; only 4 of 31 template candidates on those articles match an accepted LLM relation exactly. R11 = PASS.
- **Remaining limit:** pilot only; grounding is not correctness; no human labels, so no accuracy or improvement is claimed. Optional follow-up: label `llm_relation_review.csv` and `relation_review.csv` (P1.1) to compare precision, and extend the pilot only if useful.
- **Security:** key only in git-ignored `.env`; verified absent from all notebook text and exports.

### P0.2 Confirm the sentiment/stance decision (R8)
- **Purpose:** R8 is conditional ("where meaningful"). The notebook argues it is not meaningful without a defined target and validation labels.
- **Files:** none if confirmed; `notebooks/stage1_analysis.ipynb` §4.1 if you change your mind (then add a labelled-sample stance task, not a generic polarity score).
- **Acceptance:** written confirmation in §4.1 (or an agreed alternative).
- **Dependencies:** student judgement. **Cost:** none. **Human input:** **Required.**


### P0.3 Make the contribution statement true
- **Purpose:** The statement says Leon Bytyqi reviewed the outputs, interpreted the findings and prepared the submission. Do that before submitting: read the notebook end to end, check the results against the tables, and be ready to explain every method. If you change anything, the statement must still match what you did.
- **Files:** none unless the statement needs correction (`notebooks/stage1_analysis.ipynb`, introduction).
- **Acceptance:** you can explain each section's method, its rationale and its limitations without the notebook open.
- **Cost:** a few hours. **Human input:** **Required.**

## P1 — Scientific validity, evaluation, reproducibility

### P1.1 Human-label the review worksheets (optional, high value)
- **Purpose:** Replace unvalidated model output with measured quality: 40 article-selection rows, 56 entity rows, 51 relation rows (147 total).
- **Files:** `outputs/evaluations/stage1_review.csv`, `entity_review.csv`, `relation_review.csv` (follow `REVIEW_GUIDE.md`); then re-run notebook §9 and §11.
- **Expected outputs:** stratum-weighted selection precision/recall with Wilson intervals; entity and relation candidate precision (per method/predicate where n allows); updated compliance rows O1–O2.
- **Acceptance:** labels entered by a human without looking at other model outputs; counts and intervals reported; recall reported only for the selection task; a subset is acceptable if the count is stated. If a second annotator exists, report Cohen's κ; otherwise state none.
- **Dependencies:** REVIEW_GUIDE.md. **Cost:** negligible compute; several hours of review.
- **Human input:** **Required.**

### P1.2 Quantify NER truncation and type noise
- **Purpose:** NER sees only the title plus the first 2,500 characters, and spaCy `en_core_web_sm` mistypes many technologies (e.g., Docker, Kafka, Stripe as PERSON; Blackwell as PERSON). Measure how much this limits entity and claim coverage.
- **Files:** `notebooks/stage1_analysis.ipynb` (§6, new isolated cell); new `outputs/tables/ner_truncation_audit.csv`.
- **Expected outputs:** on a stratified sample of ~500 articles, entities/claims found in the full text versus the 2,500-character input; the share of typed-gate rejections attributable to wrong types; a comparison of `en_core_web_sm` with a larger model only if resources allow.
- **Acceptance:** fixed seed; sample and model versions documented; the main pipeline is unchanged unless the audit shows a clear, validated gain.
- **Dependencies:** none (human review of a few errors improves the audit). **Cost:** medium (full-text NER on a sample); high if a transformer model is added.

### P1.3 Detect templated/near-duplicate content
- **Purpose:** One publisher supplied 233 of 525 selected articles in 2025-01; many look templated. Near-duplicates inflate coverage and topic shares. Exact duplicates are already removed.
- **Files:** `notebooks/stage1_analysis.ipynb` (§2 or a new sensitivity cell); new `outputs/tables/near_duplicate_clusters.csv`.
- **Expected outputs:** MinHash/shingle clusters, a sensitivity analysis of §5.3 with templated clusters down-weighted, and a decision on whether to drop them.
- **Acceptance:** thresholds fixed before inspection; the primary analysis is unchanged unless the decision is documented and justified; before/after effect sizes reported.
- **Dependencies:** none. **Cost:** medium (shingling 47k articles). **Human input:** Recommended for borderline clusters.

### P1.4 Soft-assignment robustness of topic trends
- **Purpose:** Topic assignment is argmax over mixtures with low winning weights (about 0.08–0.10 on average). Check that the trends hold with weights instead of hard labels.
- **Files:** `notebooks/stage1_analysis.ipynb` (§5.3, small cell); new `outputs/tables/topic_trends_soft.csv`.
- **Acceptance:** mean normalized weights per month and window; the interpretation text is updated only if directions change.
- **Cost:** low. **Human input:** none.

### P1.5 Clean-environment reproduction
- **Purpose:** The notebook was run in one environment (with a harmless NumPy/Torch ABI warning). Reproduce it in a fresh virtual environment from `requirements-stage1.txt`.
- **Files:** `requirements-stage1.txt` (adjust pins if needed).
- **Acceptance:** a fresh environment runs all cells with the same exported counts (claims, entities, topic shares) or any differences are explained.
- **Cost:** ~25 minutes plus dataset download. **Human input:** none.

## P2 — Enhancements (use only if they improve the research questions)

- **P2.1 Selected term trajectories** (e.g., MCP, A2A, LangGraph) as monthly rates with the same denominators; files: notebook §4, `outputs/tables/agentic_term_monthly.csv`, `outputs/figures/agentic_term_trajectories.png`; acceptance: terms declared before plotting, aliases not double-counted, article IDs traceable; low cost; domain review of the term list.
- **P2.2 Community summaries** on the core co-occurrence graph, only if stable across thresholds (§8 sensitivity); files: notebook §8, `outputs/tables/graph_communities.csv`; low–medium cost; cautious interpretation.
- **P2.3 Relation-template coverage analysis:** which predicates/cues produce candidates and where gates reject; files: notebook §7, `outputs/tables/relation_gate_diagnostics.csv`; low cost.

## P3 — Optional experiments (not added merely to increase the model count)

- **P3.1 LLM structured extractor comparison** (extends P0.1): paired comparison against human-labelled relation candidates; high/variable cost; human labels and budget approval required.
- **P3.2 Dynamic or hierarchical topic model** (e.g., BERTopic) on the same corpus and time splits, retained only if it gives interpretable, stable, track-relevant gains over the NMF baseline; medium–high cost; depends on P1.3 and the stability baseline.
- **P3.3 Event-time extraction** for relation candidates (publication date is never substituted for event time); medium cost; human validation required.

## Highest-value next steps before Stage 2

1. P0.2–P0.4 — confirm the sentiment/stance decision, make the contribution statement true, and submit.
2. P1.1 — label whatever subset of the 147 worksheet rows you can; it is the cheapest way to turn candidates into measured quality.
3. P1.3 — near-duplicate sensitivity, because publisher bulk-ingest visibly changes effect sizes.
4. P1.5 — reproduce in a clean environment, so Stage 2/3 start from a verified baseline.

Do not begin Stage 2 or Stage 3 until you approve.
