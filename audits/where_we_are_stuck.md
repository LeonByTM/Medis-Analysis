# Where We Are Stuck — Stage 2

## Stage 1 is frozen and out of scope

Never edit, repair, rerun, regenerate, or otherwise change Stage 1 notebooks, code, data, graphs, or saved artifacts. Do not start Stage 1 work. For Stage 3, use existing Stage 1 artifacts only as read-only inputs when required. If an input appears missing or unusable, stop and report the blocker; do not fix Stage 1.

## Current status

The Stage 2 self-audit row **“Provide an annotated notebook, visualizations, and a short trade-off/track-insight summary”** is now **PASS**. On 2026-10-23, the final report notebook was repaired, executed top-to-bottom, and checked against the imported Colab results. The QLoRA training and matched pretrained-versus-adapted comparison have separate PASS findings; this note records how the notebook/output-package gap was closed.

The formal assignment asks for a clearly annotated notebook with model training and fine-tuning steps, visualizations comparing embeddings and model outputs, and a short summary of model trade-offs and insights from the selected topic track. See pages 3–4 of `CLT_AI_Trend_Project.pdf`.

## Evidence already available

- The recovered Colab notebook is `notebooks/Copy_of_stage2_models.ipynb`.
- QLoRA training metadata and exported adapter artifacts are present under `outputs/`.
- Matched held-out model metrics and per-excerpt losses are exported and validated; the loss rows match the saved test manifest.
- Stage 2 embedding and QLoRA figures and metrics are available.
- The final local report notebook distinguishes imported Colab evidence from local `BLOCKED` outputs. Its final cells validate the imported evidence and display the saved embedding and QLoRA figures.
- The independent human-review results are in `outputs/evaluations/stage2_human_review_metrics.csv` and `outputs/evaluations/stage2_human_review_predictions.csv`; the final evaluation cell in `notebooks/stage2_models.ipynb` regenerates them.

The Colab notebook has mixed saved execution history, including an interrupted historical cell. Keep that caveat visible and preserve the notebook as provenance; do not represent the interrupted cell as a successful run.

## What is not, by itself, a blocker

The formal assignment does not require local QLoRA training or local inference. The QLoRA run was performed in Colab, and the existing exports provide evidence for its results. Do not rerun the multi-hour training merely to make the notebook-package row PASS.

## Completed repair and verification

- Repaired `notebooks/stage2_models.ipynb` so its classical-metric export preserves existing non-embedding rows instead of overwriting the imported Colab language-model results.
- Kept the local QLoRA cells gated and clearly distinguished their `BLOCKED` state from the separately exported Colab run.
- Added a focused Agentic Web interpretation with checked Word2Vec-neighbor examples, measured weak-label comparison, and explicit limits on what the metrics support.
- Validated the 256 per-excerpt loss rows against the saved test manifest and aggregate paired delta; rendered the three saved Stage 2 figures in the notebook.
- Restarted and executed all 11 notebook code cells without errors; verified that the combined metrics file retains 39 rows (32 embedding, seven language-model).
- Added and executed a twelfth notebook code cell that validates the 200 reviewed article IDs against the held-out test split, excludes 11 `uncertain` labels from binary scores, and saves human-reference metrics and per-article comparisons.
- Updated `audits/Stage2_self_audit.md` with the PASS finding and evidence. Colab training and local base-model inference were not rerun.

## Human-reviewed Stage 2 evaluation completed

The representation-evaluation row is now **PASS**. The reviewer completed all 200 labels and confirmed they were assigned independently before consulting AI suggestions. All sample IDs were unique, in the held-out test split, and matched to saved predictions. There are 139 `not_relevant`, 50 `relevant`, and 11 `uncertain` labels; binary metrics use the 189 determinate cases only. TF-IDF macro-F1 is 0.840 and mean-Word2Vec macro-F1 is 0.757 against the reviewer labels. These are single-reviewer results on a proportionally stratified sample, not a population estimate or general real-world accuracy claim. See `audits/Stage2_self_audit.md` for the full requirement matrix and limitations.

No Stage 1 files or artifacts need to be changed. Stage 3 is being pursued as a separate workstream and does not change the Stage 2 status.

## Stage 3 status — separate workstream

**Status as of 4 November 2026: the Stage 3 self-audit reports 20 PASS, 0 PARTIAL, 0 FAIL, and 0 NOT INSPECTED.** The notebook uses 100 passages from the assigned track, generated 200 candidate Q&A pairs with `gpt-4o-mini`, and compares text-only and bounded `MENTIONED_WITH` graph expansion with `gpt-4o`. All 11 notebook code cells have executed without errors with API calls disabled. The API ledger records $0.3123534 of the $10 cap. The student reports the professor approved Leon Bytyqi's solo participation despite the brief's general groups-of-three requirement.

The frozen review worksheet, `outputs/evaluations/stage3_question_review_subset.csv`, contains 16 attributable decisions: 8 approved and 8 rejected. The 8 approved questions span all five formal question types and have paired answers and reviews. A separate approved addendum has one multi-hop and one unanswerable question with paired answers and user-confirmed ratings. These do not replace or inflate the frozen baseline. The 10-edge graph sample is reviewed (6/10 pass the entity-link/co-mention rubric); this is not semantic-relation validation. Retrieval Recall@5 ties at 1.00 on the baseline sample, while MRR@5 is 0.906 text-only versus 0.806 graph-expanded. Baseline answer ratings tie at 5/8 correct, 2/8 partially correct, and 1/8 incorrect per variant. No graph improvement is claimed.

**Review-scope and artifact caveats:** All 200 candidate IDs have pool-level review labels (174 supported, 20 needs revision, 5 uncertain, 1 unsupported); all 26 flagged candidates have source-checked revisions approved by the user. The older 174/25/1 label snapshot is retained separately. The 16-decision worksheet remains the frozen benchmark: 8 approved, 8 rejected; the other 184 candidates are outside that benchmark, not unreviewed or approved. `outputs/evaluations/stage3_questions.csv` remains the authoring-pool snapshot and is not synchronized to the curated review worksheet (199 pending, 1 approved); use `stage3_question_review_subset.csv` as the source of truth for frozen question decisions and edited evidence. The pool has 116/200 automated quote-substring matches; the other 84 are not evidence of human review. Prompt-v3 ratings and supplemental answer ratings were user-confirmed after assistant-assisted analysis, not independent source verification.

The notebook and this status note should be read with `audits/Stage3_self_audit.md`. The notebook's final interpretation now includes the supplemental cases, prompt-v3 labels, current review scope, and known limitations. Stage 1 remains read-only; Stage 2 is a separate workstream.
