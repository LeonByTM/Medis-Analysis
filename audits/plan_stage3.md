# Stage 3 Plan — Text and Knowledge-Graph RAG

## Status and sequencing

Stage 3 retrieval and candidate-generation implementation is in place. The notebook has run top-to-bottom with both API flags disabled, and all nine code cells executed. It saved a metadata-only 100-passage manifest, a deterministic grouped split for 200 question candidates, and 400 provisional retrieval-diagnostic rows (200 per variant). Eight focused unit tests pass.

The authorized question-authoring batch made 100 `gpt-4o-mini` requests at a recorded total of USD 0.0214509, well below the USD 10 cap. All 200 candidates remain `pending`. The automated whitespace-normalized evidence-quote check matched 115/200 candidates; the other 85 require human correction or rejection. No RAG answer calls, human answer ratings, or confirmatory metrics are available yet.

Stage 2 remains open: its annotated-notebook/output-package row is still PARTIAL, and the user plans to address it tomorrow. Stage 3 can proceed independently from Stage 1 inputs; keep the Stage 2 status separate and do not describe the whole project as complete.

**Stage 1 is frozen and out of scope. Never edit, repair, rerun, regenerate, or otherwise change any Stage 1 notebook, code, data, graph, or saved artifact. Do not start Stage 1 work.** Stage 3 may consume existing Stage 1 artifacts only as read-only inputs when required. If a required input is missing or unusable, stop and report the blocker; do not repair Stage 1.

At initial discovery there was no `stage3`-named notebook under `notebooks/` or `audits/`, and no obvious RAG evaluation outputs. This plan records the work now underway.

## Evidence-based implementation decisions

- The read-only Agentic Web subset has 11,867 unique articles with parsed dates from 2024-09-11 through 2026-08-31. The 100-passage sample is reproducibly stratified across publication quarters and the existing three `topic_id` groups; the manifest stores IDs and character offsets, not copied article text.
- The existing undirected graph has 8,183 nodes and 31,817 `MENTIONED_WITH` edges. Its 8,735 graph-linked article IDs all occur in the selected-track corpus. It is used only to retrieve candidate articles.
- The separate directed semantic-claim graph has 496 edges, all marked `unreviewed_candidate`; it is excluded from factual answering.
- The user authorized OpenAI calls and delegated the spend choice to project requirements. Use `gpt-4o-mini` for question drafts and `gpt-4o` for both RAG answer variants, with a combined hard code-level cap of USD 10. Send only the selected passage excerpt, title, and available date for question drafting. Record actual model response metadata and token usage; make no API calls during an ordinary notebook run unless its explicit flag is enabled.
- Generated Q&A candidates remain `pending` until human review. Do not run confirmatory answer evaluation or report final metrics before the reviewed question set is frozen.

## Objective and authority

Build and evaluate a text-only RAG baseline and a graph-enhanced RAG system over the assigned AI Media Dataset, focused on the Agentic Web track. Answers should be grounded in retrieved source passages and use publication dates when available and relevant.

The professor's formal assignment, `CLT_AI_Trend_Project.pdf` (pages 4–5), defines the mandatory scope. `EVALUATION_PROTOCOL.md` supplies controlled-comparison and error-analysis recommendations. Do not turn recommended metrics into formal grading requirements, and do not claim improvement without matched evaluation.

## Professor clarification and Stage 3 decision rules

In the professor's 24 October feedback, graph/data-representation checks may be integrated into Stage 3. The focus should be whether graph information improves RAG, not expanding the graph for its own sake. A frozen text-only baseline, a small set of manually checked questions, and comparisons of retrieval quality, answer correctness, and citation support are the practical evaluation priorities.

- Keep Stage 1 artifacts read-only. For a bounded, representative subset of entities and relations involved in the evaluation questions, independently inspect entity linking, relation direction, and source evidence. Record how many items were reviewed and the supported, incorrect, ambiguous, or uncheckable outcomes.
- Freeze the text-only baseline configuration and the manually checked evaluation questions before confirmatory comparison. Existing graph-retrieval code and diagnostics are exploratory/provisional; because they predate this clarification, document their status and do not tune against the final test questions.
- Use exactly the same corpus, reviewed questions, answer generator, prompt, context budget, decoding settings, and scoring procedure for both variants. Compare retrieval quality, answer correctness, and citation support with paired per-question results; report examples where graph use helps, hurts, or makes no difference.
- Only independently supported source passages can justify factual answer claims. A reviewed semantic relation may guide retrieval, but every answer claim must still be supported by cited article text. Unreviewed candidate edges and `MENTIONED_WITH` co-occurrence edges may serve only as clearly identified retrieval cues; never present them as factual relationships. Abstain when source evidence is insufficient.
- Keep the reviewed evaluation set small but sufficient for a transparent paired comparison. Report candidate and human-reviewed counts separately; pending/generated candidates are not gold labels. Do not claim improvement unless the matched results support it.

## Inputs to inspect before implementation

Validate the actual schema, provenance, and suitability of these candidate Stage 1 inputs; their existence alone does not establish that they are suitable:

- `data/processed/articles_clean.parquet`
- `data/processed/agentic_subset.parquet`
- `data/processed/stage1_metadata.json`
- `outputs/graph/agentic_semantic_claims.graphml`
- `outputs/graph/agentic_web.graphml`

Inspect node and edge types, article-ID links, evidence spans, relation direction, co-occurrence labels, publication dates, duplicate handling, and export metadata. Use only graph evidence that supports its represented relation; keep co-occurrence distinct from asserted relations. If a field or fact is unavailable, preserve that absence rather than infer or fabricate it.

## Planned workflow

### 1. Freeze the corpus passages

- Select approximately 100–150 representative passages from the Agentic Web track, covering varied subtopics, entities, and time periods.
- Preserve stable article IDs, passage IDs, source metadata, and offsets to the exact source text. Include publication dates only when supported by the source.
- Record the selection procedure, counts, exclusions, dataset/filter version, and random seed.
- Keep passages from a source article in the same evaluation partition to reduce leakage.

### 2. Construct and review the question set

- Generate approximately 200–300 Q&A pairs, typically 1–3 per selected passage, using a question-authoring LLM different from the model used to generate RAG answers.
- Cover factual, temporal, relational, analytical, and comparative questions.
- Store stable question IDs, question type, reference answer, supporting article/passage IDs, and source evidence.
- Manually review and refine every question/answer included in the frozen evaluation subset for clarity, correctness, diversity, and answerability from its evidence. Record review status; do not present generated labels as human-reviewed. The larger candidate pool may remain pending and must be reported separately.
- Freeze development and test partitions before system comparison. Use development data for tuning only; keep the test questions and evidence fixed for both systems.

### 3. Implement the text-only baseline

- Build a reproducible text retriever over the frozen corpus. Start with a transparent sparse baseline (for example, BM25); justify any alternative or additional retriever.
- Return source article/passage IDs with retrieved text. Include publication dates in answers when available and relevant.
- Fix retrieval depth and the context/token budget before comparing systems.

### 4. Integrate the Stage 1 knowledge graph

- Implement one documented graph-aware retrieval method that uses the inspected Stage 1 graph, such as evidence-backed entity matching followed by bounded article expansion or reranking.
- Preserve paths from graph evidence back to source article IDs and text passages.
- Do not treat co-mentions as factual relations or use unsupported edges to justify answers.
- The initial graph-enhanced variant uses bounded article expansion from `MENTIONED_WITH` edges. Select any additional enhancement only after reviewed error analysis identifies a concrete limitation; do not tune against the final test set.
- Keep the text-only and graph-enhanced variants matched in corpus, questions, answer generator, prompt, context/token budget, decoding settings, and scoring. Graph-based retrieval is the intended difference.

### 5. Evaluate, identify a limitation, and test one enhancement

- Compare the text-only and graph-enhanced systems on the same frozen questions and report paired per-question results, not only aggregate means.
- Evaluate faithfulness, relevance, context grounding, and temporal grounding; report retrieval measures such as Recall@5/MRR when the evidence labels support them. Describe the scoring rubric and any human review.
- Record citations, answer abstentions, latency, and failures where measurable. Use the same evaluator/scoring procedure for each variant.
- Use observed errors to identify at least one baseline or graph-system limitation. Select and implement one feasible enhancement (for example, reranking, query expansion, graph-based expansion, chunking, or retrieval filtering), then evaluate its effect and limitations. Keep its comparison controlled and describe exactly what changed.
- Include examples where graph use or the enhancement helps, hurts, or makes no difference. Do not claim improvement unless supported by the measured results.

### 6. Deliver and reproduce

Create a documented `notebooks/stage3_rag.ipynb` containing the objective, data and selection procedure, methods, executable pipeline, results, interpretation, limitations, saved-artifact locations, and AI-tool disclosure. Save the frozen passages/questions, split manifest, per-question results, aggregate metrics, error analysis, plots, and run metadata under documented `outputs/` paths.

Restart the notebook kernel and run feasible cells from top to bottom. Make any expensive or externally hosted model calls explicit and reproducible; do not disguise unavailable credentials, compute, or human review as successful execution.

## Proposed output paths

Confirm names and schemas before creating outputs:

- `outputs/evaluations/stage3_passages.csv`
- `outputs/evaluations/stage3_questions.csv`
- `outputs/evaluations/stage3_question_splits.csv`
- `outputs/evaluations/stage3_retrieval_diagnostics.csv`
- `outputs/evaluations/stage3_results.csv`
- `outputs/evaluations/stage3_metrics.csv`
- `outputs/evaluations/stage3_answer_quality_metrics.csv`
- `outputs/evaluations/stage3_error_analysis.csv`
- `outputs/evaluations/stage3_api_usage.csv`
- `outputs/tables/stage3_run_metadata.json`
- `outputs/figures/stage3_passage_sample_coverage.png`
- `outputs/figures/stage3_retrieval_comparison.png`
- `notebooks/stage3_rag.ipynb`

## Completion gate

After Stage 3 implementation and validation are done, **read `audits/Stage3_self_audit.md` from beginning to end**, fill it with exact notebook cell references, artifact paths, observed counts, checks, and unresolved blockers, and only then report Stage 3 complete. Do not mark a criterion PASS based only on successful cell execution. Keep the Stage 2 audit independent and unresolved until its remaining work is verified.
