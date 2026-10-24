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

The Stage 3 notebook and retrieval pipeline are implemented and have been run locally. A deterministic 100-passage sample, a grouped split, and paired provisional retrieval diagnostics for 200 candidate questions have been saved. The question-authoring batch used 100 `gpt-4o-mini` calls and cost USD 0.0214509 under the authorized USD 10 cap.

**Stage 3 is not complete.** All 200 generated questions remain `pending`; only 115/200 evidence quotes passed the automated whitespace-normalized source-text check. The other 85 must be manually corrected or rejected. No RAG answers or confirmatory answer-quality metrics exist because the notebook intentionally gates those calls on human review.

Next: review `outputs/evaluations/stage3_questions.csv` against the source passages, record a reviewer ID and decision for every candidate, then run the paired answer variants, review the answers, complete the error analysis, and update `audits/Stage3_self_audit.md`. See `audits/plan_stage3.md` for the method and `audits/Stage3_self_audit.md` for the evidence-based current status. Stage 1 remains frozen; do not modify any of its files or artifacts.
