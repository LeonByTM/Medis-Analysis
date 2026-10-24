# Stage 2 Self-Audit and Improvement Instructions

## Current-checkout re-audit (24 October 2026)

The final report notebook, `notebooks/stage2_models.ipynb`, was executed top-to-bottom after the notebook-package repair: its 11 original code cells completed without errors, the imported Colab evidence was validated, and three saved figures were rendered inline. On 24 October, the reviewer completed all 200 labels in the held-out test worksheet. The reviewer confirmed that the labels were assigned independently before consulting AI suggestions. All 200 IDs are unique test IDs and join one-to-one to saved predictions. The added final notebook cell computes human-reference metrics, excluding 11 `uncertain` labels from binary scores. The local CUDA-gated QLoRA cells correctly report that no local training occurred; they do not describe the separate Colab run.

The Colab Drive export has been selectively copied from the local `CTL/` staging folder into `outputs/`. The source/export notebook, `notebooks/Copy_of_stage2_models.ipynb`, remains as provenance and has mixed saved outputs: cell 22 (execution count 9) ends with `KeyboardInterrupt` during backpropagation; cell 23 copies adapter files from an active checkpoint into the final directory; and cell 29 labels the run complete based only on that directory's existence. Those cells are not presented as a clean successful run. Separately exported adapter files, metadata, training curve, evaluation manifest, and per-excerpt losses were checked against each other. Metadata reports 6,009 optimizer steps and 96,158 chunks. In the still-active Colab runtime, `refs/main` matched the sole cached snapshot, yielding revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, which is recorded in the metadata.

## Results after audit

- **Corpus and split:** 47,443 cleaned articles were screened; 47,375 entered the chronological split (38,002 train, 4,720 validation, 4,653 test) and 68 were quarantined. The split checks found no cross-partition overlap in IDs, content hashes, or normalized title groups.
- **Classical embeddings:** the Word2Vec model has 100 dimensions and 95,445 vocabulary items and was trained for three epochs on the 38,002 training articles. The held-out context-pair ROC-AUC is 0.677. On the test split, TF-IDF macro-F1 is 0.911 and mean-Word2Vec macro-F1 is 0.804, versus 0.385 for the majority baseline. The classification target is Stage 1's keyword-rule flag, so these scores indicate agreement with weak labels, not independently verified Agentic Web accuracy. Context pairs are generated from text, not human similarity judgments.
- **Agentic Web interpretation:** Word2Vec places `mcp` near `a2a`, `webmcp`, `api`, `sdk`, and `server`, and `langgraph` near `crewai`, `autogen`, `langchain`, and `llamaindex`. These neighbors are consistent with related technical contexts in the corpus, but do not establish factual relationships or interoperability. The PCA plot and neighbor examples are descriptive.
- **Human-reviewed extrinsic evaluation:** the 200-row sample was drawn from the existing test set, proportionally stratified by month and weak-rule group, then shuffled. The worksheet omits the group flag and model predictions. The reviewer labeled 139 `not_relevant`, 50 `relevant`, and 11 `uncertain`; the uncertain cases are excluded from binary metrics, leaving 189 scored articles (support: 139 not relevant, 50 relevant). Against those human labels, TF-IDF accuracy/macro-F1 are 0.862/0.840; mean-Word2Vec 0.783/0.757; the majority baseline 0.735/0.424; and the Stage 1 keyword-rule reference 0.873/0.851. These are sample-level results from one reviewer, not population estimates or general real-world accuracy. Full per-class metrics and confusion counts are in `outputs/evaluations/stage2_human_review_metrics.csv`; per-article comparisons are in `outputs/evaluations/stage2_human_review_predictions.csv`.
- **Colab QLoRA comparison:** on the same 256 held-out excerpts (130,816 tokens per model), pretrained/adapted mean token loss is 3.353336/2.715525 and perplexity is 28.597976/15.112536. The adapted-minus-pretrained paired mean excerpt-loss delta is -0.637811; the 95% paired-bootstrap interval is [-0.695067, -0.587951]. The per-excerpt file has 256 unique test IDs, no missing losses, and IDs matching the test manifest; its mean delta matches the saved aggregate. This supports improved next-token fit on the sampled corpus excerpts only, not factuality, semantic understanding, or trend-detection accuracy.
- **Training trade-off:** imported metadata reports Qwen3-1.7B QLoRA, one epoch, 38,002 training articles, 96,158 chunks, 6,009 optimizer steps, a Tesla T4, and 25,016.5 seconds (~6.95 hours). The adapter has 8,716,288 parameters, derived from 392 LoRA tensor shapes. The adapter is local and git-ignored; base-model weights are not bundled, and local inference was not run.
- **Visual results:** `stage2_agentic_embeddings_pca.png`, `stage2_qlora_training_loss.png`, and `stage2_qlora_test_loss_comparison.png` were loaded and rendered in the final notebook. The report notebook also validates the imported results; the combined metrics export contains 39 rows (32 embedding and seven language-model rows).

| Requirement | Mandatory / conditional / recommended | Evidence | Status | Gap |
|---|---|---|---|---|
| Prepare the assigned corpus and usable train/validation/test partitions | Mandatory workflow; full-corpus use is the project scope (the brief says Stage 2 may use the full corpus) | `data/processed/articles_clean.parquet`; `outputs/evaluations/stage2_split_ids.csv`; `outputs/evaluations/stage2_quarantined_articles.csv` | PASS | 47,443 source rows are accounted for by 47,375 split rows and 68 quarantined near-duplicate/title-boundary rows. Split counts are 38,002 train, 4,720 validation, and 4,653 test; IDs, content hashes, and normalized title groups are disjoint across partitions. |
| Train and visualize classical embeddings, including the selected track | Mandatory | `outputs/models/stage2_word2vec.model`; `outputs/tables/stage2_word2vec_metadata.json`; `outputs/tables/stage2_word2vec_neighbors.csv`; `outputs/figures/stage2_agentic_embeddings_pca.png` | PASS | Reloaded model has 100 dimensions and 95,445 vocabulary items. The recorded training-ID hash matches the 38,002 training IDs. PCA and neighbors are descriptive, not independent proof of semantic quality. |
| Evaluate representations with intrinsic and extrinsic tasks | Mandatory | `outputs/evaluations/stage2_metrics.csv`; `outputs/evaluations/stage2_rule_label_predictions.csv`; `outputs/evaluations/stage2_representation_review_200.csv`; `outputs/evaluations/stage2_representation_review_guide.md`; `outputs/evaluations/stage2_human_review_metrics.csv`; `outputs/evaluations/stage2_human_review_predictions.csv`; `notebooks/stage2_models.ipynb` final evaluation cell | PASS | Context-pair AUC is 0.677. The original TF-IDF and mean-Word2Vec macro-F1 scores (0.911 and 0.804) measure agreement with Stage 1 weak labels. The completed independent human-labeled sample has 189 determinate cases after excluding 11 `uncertain`; TF-IDF and mean-Word2Vec macro-F1 against the reviewer labels are 0.840 and 0.757, respectively. Results are limited to this one-reviewer sample and are not population estimates. |
| Fine-tune a compact open model with parameter-efficient adaptation | Mandatory | `notebooks/Copy_of_stage2_models.ipynb` cells 22–23 and 29; `outputs/models/qwen3_1.7b_agentic_media_qlora/adapter_config.json`; `adapter_model.safetensors`; `outputs/tables/stage2_qlora_metadata.json`; `outputs/evaluations/stage2_lm_training_curve.csv` | PASS | Exported artifacts report Qwen3-1.7B QLoRA on a Tesla T4, 38,002 training articles, 96,158 chunks, 6,009 optimizer steps, one epoch, and 25,016.5 seconds. The adapter parameter count (8,716,288) was derived from 392 LoRA tensor shapes, not recorded in Colab metadata. The recovered notebook includes an interrupted saved execution and a weak directory-based completion check; these are caveats about that notebook's mixed execution history, not evidence that the separate exported adapter and training artifacts are incomplete. Revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` was recovered from the active Colab cache and recorded with its source. |
| Compare the pretrained and adapted model on the same held-out examples and report trade-offs | Mandatory | `notebooks/Copy_of_stage2_models.ipynb` cell 34; `outputs/evaluations/stage2_metrics.csv`; `outputs/evaluations/stage2_lm_eval_examples.csv`; `outputs/evaluations/stage2_lm_per_excerpt_losses.csv`; `outputs/tables/stage2_qlora_metadata.json`; `outputs/figures/stage2_qlora_test_loss_comparison.png` | PASS | The loss file has 256 unique test IDs and no missing values; its example rows exactly match the saved test manifest in order. Recomputed paired-bootstrap 95% CI is [-0.6950674793, -0.5879510110], matching the reported [-0.695067, -0.587951] to rounding. Revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` was recovered from the sole matching Colab cache snapshot and recorded with its source. Peak GPU memory was not recorded; the metrics measure next-token fit, not factuality or semantic accuracy. |
| Include an annotated notebook, visualizations, focused-track interpretation, and model insights | Mandatory outputs; full-corpus use is permitted and should retain track focus | `notebooks/stage2_models.ipynb` sections 8–9 and final human-review evaluation cell; code cells 10–12; `outputs/figures/stage2_agentic_embeddings_pca.png`; `outputs/figures/stage2_qlora_training_loss.png`; `outputs/figures/stage2_qlora_test_loss_comparison.png`; imported metric and loss exports | PASS | The final report notebook's 11 original code cells ran without errors and validate imported Colab evidence and paired test losses. The new final cell was executed against the saved review, predictions, and split manifest; its metric outputs were reloaded and checked. The notebook renders the embedding and QLoRA figures and reports focused-track interpretation. The source Colab notebook's mixed history, missing base weights, and lack of local inference remain disclosed caveats; the Colab training was not rerun. |

The final adapter is local and ignored by Git; the base model is not bundled. Its Safetensors header and tensor offsets were structurally checked, but the model was not loaded for local inference. The two `CTL/stage2_models.ipynb` copies are byte-identical and retain the old local `BLOCKED` state. `notebooks/Copy_of_stage2_models.ipynb` remains the recovered source/provenance notebook with mixed saved outputs; its interrupted cell is not treated as proof that the separate export failed. The final report notebook, `notebooks/stage2_models.ipynb`, now clearly separates local blocked training from imported Colab evidence, validates the artifacts, checks all 256 paired loss rows against the test manifest and saved aggregate, and renders the three saved figures. The imported metrics export retains 39 rows (32 embedding and seven language-model rows). Remaining model limitations include absent base weights and unrecorded peak GPU memory.

### Audit–repair–verification loops completed

1. **Artifact reconciliation:** compared the existing tracked metrics with the Colab export, preserved all 32 existing rows, appended only the seven language-model rows, and copied only the final adapter, metadata, evaluation manifest, training curve, and QLoRA plots. Recovery checkpoints and duplicated Stage 1 files were left in the ignored `CTL/` folder. Counts, split membership, Word2Vec provenance, adapter checksums, and plot signatures passed.
2. **Notebook and documentation reconciliation:** added a separate imported-results section and a CPU-safe artifact validator; updated this audit, the README, and the Stage 2 plan so the old local `BLOCKED` outputs are not confused with the Colab run. The new validator source and final artifact check ran in local Python without loading the base model or starting training; notebook JSON validation passed.
3. **Colab execution-evidence triage:** recovered and inspected the downloaded Colab notebook. Cell 22's saved execution ends with `KeyboardInterrupt` in `.backward()`, and cell 29's directory-existence check alone cannot establish completion because cell 23 can create the final folder from checkpoint files. The separately exported adapter, training curve, and metadata report a completed run, so the interrupted cell is treated as an ambiguous saved attempt rather than proof the export failed. A follow-up cell in the still-live Colab runtime exported all 256 test-example losses; the rows exactly match the test manifest and an independent paired bootstrap reproduces the reported interval. A cache check found one Qwen3-1.7B snapshot whose hash matched `refs/main`; that revision is now recorded in metadata. The matched comparison is PASS, with the remaining limitations noted above.

## Mission

Continue auditing and verifying the existing Stage 2 work for **Agentic AI Radar — Agentic Web**. The local embedding workflow and a separate Colab QLoRA run now have saved artifacts; reconcile them honestly, close feasible gaps, and do not treat stale blocked outputs as evidence about the Colab run. Stage 1 is finished. Do not stop after writing a review or a plan: inspect the real files and saved results, make the smallest justified Stage 2 repairs, run every feasible check, inspect the resulting evidence, and repeat.

Work autonomously without asking the student for routine approval or clarification. Make safe, reversible decisions within this scope and document assumptions. Do not start Stage 3.

## Authority and scope

1. Read `CLT_AI_Trend_Project.pdf` and the project requirements before changing files. The professor's formal assignment is authoritative for grading. Use `PROJECT_SPEC.md`, `DATA_CONTRACT.md`, `EVALUATION_PROTOCOL.md`, `IMPLEMENTATION_PLAN.md`, `CODING_AGENT_INSTRUCTIONS.md`, and this repository's current documentation as supporting context.
2. Distinguish formal mandatory requirements from lecture recommendations and optional improvements. In particular, do not present the Course 3–6 lecture examples, tools, metrics, or suggested target counts as formal requirements unless the professor's assignment confirms them.
3. Work only on Stage 2 and directly related Stage 2 support files, configuration, tests, and documentation. **Never modify Stage 1 notebooks, code, exports, or artifacts.** Read Stage 1 outputs only as needed to verify Stage 2 input provenance and compatibility; preserve them.
4. Preserve existing user work, including uncommitted changes. Inspect `git status` before editing, do not overwrite unrelated work, and do not delete or regenerate valuable artifacts without a clear, necessary reason.
5. Do not begin Stage 3, build a UI, change the assigned dataset, or rerun paid API extraction.

## Inspect evidence before trusting claims

Inspect the complete Stage 2 notebook (`notebooks/stage2_models.ipynb`), its saved cell outputs, related code and dependency files, actual input data and Stage 1 exports, generated Stage 2 artifacts, and Git status. A successful cell run or a statement in a prior plan is not proof that a method is valid or that an artifact contains the claimed result.

The following statements were previously reported and must be traced to notebook outputs and reloadable artifacts; the current evidence snapshot above records what has since been checked:

- 47,443 source articles screened; 47,375 retained; 68 quarantined.
- 38,002 training, 4,720 validation, and 4,653 test articles.
- 26 algorithmically flagged near-duplicate candidate pairs.
- Word2Vec with 100 dimensions, 95,445 vocabulary items, and three epochs; held-out context-pair AUC 0.677.
- TF-IDF macro-F1 0.911 and mean-Word2Vec macro-F1 0.804.
- The earlier local notebook run marked Qwen3-1.7B QLoRA as not run because local CUDA was unavailable. The later Colab export contains a final adapter, metrics, and a downloaded notebook with training/evaluation source. Its saved execution history is mixed, so the completion status is based on the separate exported adapter, metadata, and training curve.

Check row counts, IDs, split disjointness, file contents, metadata, notebook execution state, and metric definitions directly. Explain any counts that differ, including the source of each count and whether it is before or after filtering.

## Stage 2 compliance audit

Create or update a concise compliance matrix in the Stage 2 notebook or its existing Stage 2 audit documentation. For every formal requirement, record:

`Requirement | Mandatory/conditional/recommended | Formal source | Evidence (cell, path, or output) | Status (PASS/PARTIAL/FAIL/NOT VERIFIED) | Gap and repair`

At minimum, check the following against the actual assignment and available evidence:

- Use the **full cleaned assigned corpus** for corpus-level Stage 2 work; document any exclusions and their evidence-based reasons.
- Record data provenance, stable article IDs, schema, filtering/quarantine counts, seed, and reproducibility metadata.
- Define train/validation/test partitions before evaluating generalization; verify ID-level disjointness and check duplicates or near-duplicates across partitions. Use the split method justified by the data and assignment.
- Train word embeddings only on permitted training text for held-out evaluation. Report configuration, objective, runtime, vocabulary handling, and saved-model provenance.
- Show interpretable nearest-neighbor examples and a reproducible quantitative evaluation with a suitable comparison or baseline; describe what the evaluation does and does not establish.
- Specify any language-model adaptation task, input/target format, training objective, label provenance, metrics, and model-selection rule before training or evaluation.
- Compare baseline and adapted models on the same held-out examples and under comparable settings if adaptation actually ran. Otherwise mark it **NOT RUN / NOT VERIFIED**, explain why, and provide a runnable, non-cost-incurring workflow where feasible.
- Report actual hyperparameters, seed, package/model versions, hardware, runtime, trained-parameter count, and limitations when available. Never invent missing measurements.
- Include results, negative findings, error analysis, limitations, reproducibility instructions, artifact paths, and AI-tool disclosure.
- Reload and validate saved Stage 2 artifacts; verify that notebook outputs correspond to the current inputs/configuration and are not stale.
- Confirm that no Stage 2 result is presented as independently verified real-world accuracy when its labels come from Stage 1 keyword/rule heuristics. Call such labels weak or silver labels and describe metrics as agreement with those heuristics, not ground truth.

Keep Course 3–6 lecture alignment scientifically useful but proportionate to Stage 2. For example, clarify representation evaluation, provenance, leakage, and evidence limitations where relevant; do not expand scope into Stage 1 repairs or Stage 3 retrieval work.

## Required improvement loops

Complete **two full audit–repair–verification loops**, and a **third loop if any feasible high-priority gap remains**. Each loop must:

1. Identify specific, evidence-backed gaps against the formal requirements.
2. Make the smallest complete Stage 2 changes that address those gaps, preserving working cells and existing results where valid.
3. Execute the affected notebook cells or targeted tests using the available environment.
4. Inspect actual outputs and reloadable artifacts, not just exit codes; update the compliance matrix and results text to match the evidence.
5. Record remaining blockers and use them to set the next loop's priorities.

Do not make cosmetic changes just to create activity. Stop after the second or third loop only when all requirements that can be completed and verified with currently available data, permissions, and hardware are PASS, or when remaining gaps are genuine external blockers. Never claim “100% complete” if any requirement is PARTIAL, FAIL, or NOT VERIFIED. Instead, state exactly what is blocked and what evidence or resource would resolve it.

## Safety, evidence, and resource limits

- Do not fabricate records, dates, labels, citations, scores, training runs, model outputs, or successful checks.
- Rule-derived Agentic Web labels are not independent gold labels. Do not train/test on them and describe the result as semantic understanding or real-world accuracy.
- Never download large model weights, start paid services, require a paid runtime, or launch expensive training. Prepare a reproducible workflow and mark GPU-dependent work not run unless suitable compute is already available and authorized.
- Do not silently replace unavailable fields or missing artifacts with guessed values. Fail clearly or mark the affected criterion blocked.
- Prefer deterministic, focused validation and report commands/cells, actual row/sample counts, and any failures.

## Final response

Finish with a concise, evidence-based report containing:

1. Stage 2 files changed and the purpose of each change.
2. The two or three audit–repair–verification loops completed.
3. Checks and notebook cells actually run, with pass/fail results and actual counts.
4. The requirement matrix with evidence references and PASS/PARTIAL/FAIL/NOT VERIFIED statuses.
5. What remains blocked by unavailable hardware, data, or permissions; distinguish unrun from failed.
6. The most important remaining risks and the precise next action needed to resolve each.
7. Confirmation that Stage 1 files/artifacts were not modified and Stage 3 was not started.

Do not promise a grade or claim full professor compliance without evidence.
