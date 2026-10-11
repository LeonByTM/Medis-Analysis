# Stage 1 audit against the professor's brief

**Authority:** [CLT_AI_Trend_Project.pdf](../CLT_AI_Trend_Project.pdf), especially pp. 2–3 (Stage 1) and p. 5 (contribution and AI-disclosure rules). The project Markdown files are working specifications; where they differ from the PDF, the PDF governs.

**Object assessed:** the current [Stage 1 notebook](../notebooks/stage1_analysis.ipynb), executed top to bottom, and its exported artifacts. **Submission:** individual, by Leon Bytyqi (permitted; confirmed by the student). Status describes evidence in the workspace; it is not a statement that an NLP output is factually correct.

## Status key

- **PASS** — evidence is present and the relevant code/output was checked.
- **PARTIAL** — part of an explicit requirement is met; the gap is stated.
- **FAIL** — an explicit requirement is not met.
- **NOT VERIFIED** — depends on evidence not available in the workspace, or on human work not yet done.
- **NOT APPLICABLE** — a conditional requirement judged not to apply, with a written justification (and, where noted, awaiting the student's confirmation).

Classes follow the brief's own wording: **Mandatory** (stated without qualification), **Conditional** (qualified by "where appropriate / where meaningful / where useful"), **Optional** (not required by the brief; quality improvements I identified).

## Execution evidence (verified, not assumed)

- The notebook was executed top to bottom in a fresh kernel: **30 of 30 code cells executed, consecutive execution counts, 0 error outputs.** The only stderr text is the known NumPy 2.3.3 / PyTorch ABI import warning from the local environment, which does not affect results.
- **28 of 28 structural checks passed** ([stage1_structural_checks.csv](../outputs/evaluations/stage1_structural_checks.csv)), including exact entity-span and evidence-span equality with the source text, normalization invariants, and a check that the required contribution/AI-disclosure text is present and no placeholder/stale text remains. All artifact round-trip checks passed (tables, both GraphML files, schema JSON, figures present).
- Independent check outside the notebook: 496 claim evidence spans and 137,715 entity spans were compared with the NER source text (0 mismatches); both GraphML files reloaded (`Graph` 8,183 nodes / 31,817 edges; `MultiDiGraph` 8,183 nodes / 496 edges); the co-occurrence matrix is 35 × 35, symmetric, zero diagonal.
- **Not done:** no human labels exist for any worksheet; no precision, recall, F1 or agreement figure is reported or claimed. This is an **optional** improvement (see O1–O3).

## Requirement-by-requirement assessment

Notebook sections: 1 acquisition · 2 cleaning · 3 selection · 4 EDA (4.1 sentiment decision) · 5 topics (5.1–5.4) · 6 NER · 7 relations · 8 graph · 9 evaluation · 10 artifacts · 11 results (11.1 answers). Compliance rows are also exported to [stage1_compliance_audit.csv](../outputs/evaluations/stage1_compliance_audit.csv).

| ID | Professor's requirement (page) | Class | Status | Evidence and remaining gap |
|---|---|---|---|---|
| R1 | Use the AI Media Dataset (p. 2) | Mandatory | **PASS** | §1; [stage1_metadata.json](../data/processed/stage1_metadata.json): `jannalipenkova/ai-media-dataset` v2, 47,751 raw rows, 7 observed columns. |
| R2 | One track used consistently for the Stage 1 graph (p. 2) | Mandatory | **PASS** | Agentic Web; §3 selects 11,867 of 47,443 cleaned articles; the graph is built on that subset. |
| R3 | Graph evidence traceable to the dataset (p. 2) | Mandatory | **PASS** | Article/source IDs, URLs, publication dates and exact evidence offsets in [relation_claims.csv](../outputs/tables/relation_claims.csv), [entities.csv](../outputs/tables/entities.csv), [graph_edges.csv](../outputs/tables/graph_edges.csv) and GraphML. |
| R4 | Clean: duplicates, missing values, normalize text (p. 2) | Mandatory | **PASS** | §2: 65 exact duplicate texts removed, 0 invalid dates, 0 missing texts, 47,443 final records, raw text preserved. No near-duplicate removal (documented; see plan P1.4). |
| R5 | Tokenization, lemmatization, stop-word removal where appropriate (p. 2) | Conditional | **PASS** | §5: topic view uses tokenization, stop words and vocabulary-level lemmatization (3,672 of 12,000 terms mapped, 9,723 features after merging; [topic_term_lemma_map.csv](../outputs/tables/topic_term_lemma_map.csv)). Lemmatization deliberately not applied to NER/stored text. Approximation (one-word context) is disclosed. |
| R6 | Consistent representation incl. acronyms/abbreviations (p. 2) | Mandatory | **PASS** | §5 phrase/acronym canonicalization (`MCP`, `LLM`, `RAG`, `A2A`, `ADK`, `AGI`…); §6 lexicon and name normalization ([entity_normalization_report.csv](../outputs/tables/entity_normalization_report.csv); 0 IDs with leading *the*/possessive; one ID per normalized spaCy name). **Gaps:** no exhaustive acronym audit; bare `Copilot` stays ambiguous; spaCy entity *types* remain unreliable for technologies. |
| R7 | Temporal analysis Sep 2024 – Aug 2026 (p. 2) | Mandatory | **PASS** | §4, §5.3; [monthly_coverage.csv](../outputs/tables/monthly_coverage.csv), [topic_monthly_counts.csv](../outputs/tables/topic_monthly_counts.csv) with explicit denominators; first-vs-last-six-month windows; bulk-ingest robustness. |
| R8 | Sentiment or stance where meaningful (p. 3) | Conditional | **NOT APPLICABLE** | §4.1 documents why none is produced (sentiment is not stance; no validation labels; not needed for the research questions) and what a meaningful stance task would require. The student has decided not to pursue it. |
| R9 | Clear visualizations: bar charts, time series, term/topic trajectories (p. 3) | Mandatory | **PASS** | 10 figures in [outputs/figures/](../outputs/figures), e.g. monthly coverage, top sources, frequent terms, topic counts/shares by month, topic stability, entity co-occurrence heatmap, two graph views, threshold sensitivity. |
| R10 | Pretrained NER for organizations, technologies, models, standards, products, people (p. 3) | Mandatory | **PASS** | §6: spaCy `en_core_web_sm` + domain lexicon; 137,715 mentions, 8,183 entity IDs, article provenance. **No human evaluation**; spaCy types unreliable for technologies (examples in §6). |
| R11 | LangExtract or a comparable structured extraction approach to improve entity and relation extraction (p. 3) | Mandatory | **PASS** | §7 (schema-constrained lexical templates, 496 unreviewed candidates) and §7.1: **LangExtract with OpenAI `gpt-4o-mini`** (temperature 0, the notebook's entity types and nine predicates) applied to a fixed-seed 50-article pilot (25 with and 25 without template candidates). 423 entity and 65 relation extractions; 41 relations passed source-grounding validation (aligned span names both entities, predicate in schema), 24 were rejected with logged reasons ([llm_relation_pilot.csv](../outputs/tables/llm_relation_pilot.csv), [llm_pilot_summary.csv](../outputs/tables/llm_pilot_summary.csv), [llm_vs_template_candidates.csv](../outputs/tables/llm_vs_template_candidates.csv)). On the same articles only 4 of 31 template candidates match an accepted LLM relation exactly, and 36 of 41 accepted LLM relations are LLM-only: the methods largely find different relations. **Limits:** pilot only; grounding is not correctness; no human labels, so neither method's accuracy is measured and no improvement is claimed. The key is read from a git-ignored `.env` and a check confirms it appears in no notebook text or export. Previously PARTIAL. |
| R12 | Entity co-occurrence matrix (p. 3) | Mandatory | **PASS** | [entity_cooccurrence_matrix.csv](../outputs/tables/entity_cooccurrence_matrix.csv) (35 × 35, symmetric) and [heatmap](../outputs/figures/entity_cooccurrence_matrix.png). |
| R13 | Compact topic-specific schema before graph construction (p. 3) | Mandatory | **PASS** | [knowledge_graph_schema.json](../outputs/tables/knowledge_graph_schema.json), [relation_schema.csv](../outputs/tables/relation_schema.csv): entity types, predicate domain/range, graph types, provenance, temporal and confidence policies. |
| R14 | Graph: nodes = entities, edges = co-mentions or extracted relations (p. 3) | Mandatory | **PASS** | [agentic_web.graphml](../outputs/graph/agentic_web.graphml) (co-mentions) and [agentic_semantic_claims.graphml](../outputs/graph/agentic_semantic_claims.graphml) (unreviewed directed candidates), kept separate; layers table ([graph_layers_summary.csv](../outputs/tables/graph_layers_summary.csv)) separates inventory, connected, isolated, candidate and validated (none). |
| R15 | Network metrics (centrality, clustering) to reveal influential entities and emerging relationships (p. 3) | Mandatory | **PASS** | §8: degree, weighted degree, PageRank, clustering, approximate betweenness ([graph_node_metrics.csv](../outputs/tables/graph_node_metrics.csv), 680-node core graph); threshold sensitivity ([graph_threshold_sensitivity.csv](../outputs/tables/graph_threshold_sensitivity.csv)); normalized first-vs-last-six-month emergence of entities and pairs, excluding publisher-dominated entities ([graph_emerging_pairs.csv](../outputs/tables/graph_emerging_pairs.csv), [graph_emerging_entities.csv](../outputs/tables/graph_emerging_entities.csv); unfiltered versions exported too). Borderline: `GitLab` (89.8% from one publisher) is just under the predeclared 90% rule and stays in the entity ranking. My earlier audit marked this PASS although only degree centrality existed; that gap is now closed. No community detection (not required). |
| R16 | Visualize the graph (NetworkX/Graphviz) (p. 3) | Mandatory | **PASS** | [focused_knowledge_graph.png](../outputs/figures/focused_knowledge_graph.png), [semantic_claim_graph.png](../outputs/figures/semantic_claim_graph.png). |
| R17 | Store graph and schema for Stage 3 reuse (p. 3) | Mandatory | **PASS** | Two GraphML files, node/edge CSVs and schema JSON, reloaded and compared by the round-trip cell. |
| R18 | Topic model: LDA, NMF, Top2Vec or BERTopic (p. 3) | Mandatory | **PASS** | §5: TF–IDF/NMF K=3; terms, representative documents, prevalence ([topic_terms.csv](../outputs/tables/topic_terms.csv), [topic_assignments.csv](../outputs/tables/topic_assignments.csv)); stability over 10 resamples per K ([topic_stability_summary.csv](../outputs/tables/topic_stability_summary.csv)). |
| R19 | Hierarchical or dynamic topic analysis where useful (p. 3) | Conditional | **PARTIAL** | Topic evolution is analysed with fixed topics by month and window; subtopic splitting is inspected via K=4/6 (§5.1). No hierarchical or dynamic topic model was fitted; its usefulness was not tested. |
| R20 | Interpret topic distributions over time, related to the track (p. 3) | Mandatory | **PASS** | §5.2–5.4: topic labels, increases/decreases with two denominators, representative articles, publisher-concentration robustness, explicit "cannot establish" list; every number/direction is asserted against the executed tables. |
| R21 | Document key insights and notable developments (pp. 2–3) | Mandatory | **PASS** | §11.1 answers the four research questions with [Empirical]/[Interpretation]/[Not established] labels generated from executed results. |
| R22 | Notebook with cleaning/preprocessing, annotations, explanations and visualized results (pp. 3, 5) | Mandatory | **PASS** | Executed, documented notebook (see execution evidence). The earlier contradiction about the brief being unavailable is removed. |
| R23 | Each member indicates their contribution (p. 5) | Mandatory | **PASS** | Notebook introduction: individual contribution statement for Leon Bytyqi. The statement says he directed the work, reviewed outputs and interpreted findings; **the student must make that true before submission** (see plan P0.4). It claims no manual authorship of code and no human validation. |
| R24 | Disclose AI coding tools by name and use (p. 5) | Mandatory | **PASS** | Notebook introduction: "GitHub Copilot and AI language models drafted the code for Stage 1…" with separately stated student responsibilities; mirrored in the metadata JSON. |
| R25 | Submit each stage's notebook per ILIAS (p. 5) | Mandatory | **NOT VERIFIED** | Not verifiable from the workspace. |
| G1 | Stage 2, Stage 3 and Streamlit not started (project instruction) | — | **PASS** | No later-stage notebook or application exists. |

## Optional improvements (not required by the brief)

| ID | Improvement | Status | Note |
|---|---|---|---|
| O1 | Human-labelled article-selection precision/recall | **NOT VERIFIED** | 40-row stratified worksheet, unlabelled. Stratum-weighted estimator with Wilson intervals is implemented but receives no labels. |
| O2 | Human-labelled NER and relation precision | **NOT VERIFIED** | 56 entity and 51 relation rows, unlabelled. |
| O3 | Event-time extraction for relation candidates | **NOT VERIFIED** | Event time is null by design; modality/polarity are lexical heuristics. |
| O4 | Human review of the LangExtract pilot relations | **NOT VERIFIED** | 41 accepted LLM relations in `llm_relation_review.csv`, unlabelled. |

Instructions for O1–O4 are in [REVIEW_GUIDE.md](../outputs/evaluations/REVIEW_GUIDE.md). The notebook now preserves and uses a worksheet once any label is entered (it never overwrites labelled files).

## Decisions that still require the student

1. **R23** — confirm the contribution statement is accurate once you have actually reviewed the outputs and interpretations.
2. **R25** — submit via ILIAS.
3. **Optional** — whether to label any of the review worksheets (147 rows plus 41 LLM-pilot relations); nothing is required.

## Changes since the previous audit

R23 FAIL → PASS (individual statement); R22 PARTIAL → PASS (stale statement removed); R20/R21 PARTIAL → PASS; R5/R6 PARTIAL → PASS; R8 PARTIAL → NOT APPLICABLE (decision documented); R11 PASS → PARTIAL → **PASS** (an earlier PASS rested on templates alone and was not supported; a real LangExtract pilot has now been run and validated for grounding); R15 PASS with new evidence (the earlier mark rested on degree centrality only); R19 stays PARTIAL.

## Overall

All mandatory requirements that can be satisfied from the workspace now pass; **R25** (ILIAS submission) can only be completed by the student, and **R19** (conditional) is PARTIAL. The optional human review is unperformed and no quality metric exists for either extraction method, which limits how strongly any finding can be stated. Stage 1 should not be described as fully verified: its outputs are unvalidated model outputs, and the AI use (code drafting by GitHub Copilot/AI models, and OpenAI `gpt-4o-mini` via LangExtract for the pilot) is disclosed in the notebook introduction.
