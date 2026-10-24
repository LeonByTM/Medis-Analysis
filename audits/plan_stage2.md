You are my senior NLP research engineer, machine learning scientist, and rigorous academic project reviewer.

## OBJECTIVE

Help me produce an exceptional, academically rigorous **Stage 2 submission** for my university course **Computational Language Technologies**, project **“Agentic AI Radar — Tracking the Rise of Autonomous AI Agents”**, Agentic Web track.

My goal is to achieve the highest possible grade (25/25 if merited) and deliver work that stands out through correct methodology, meaningful experiments, reproducibility, critical analysis, and excellent documentation.

Do not optimize for superficial complexity, impressive-looking metrics, or a large number of models. Optimize for the actual professor's requirements and the scientific validity of every result.

I am completing the project individually with instructor permission. Clearly disclose AI-assisted coding and my own contribution. Never fabricate experiments, human evaluations, metrics, or completed work.

## Current verified outcome (24 October 2026)

The Stage 2 representation-evaluation requirement is now marked **PASS** in `audits/Stage2_self_audit.md`. A reviewer-labeled sample of 200 unique held-out test articles was matched one-to-one with the saved predictions. The reviewer confirmed that labels were assigned independently before consulting AI suggestions. There are 139 `not_relevant`, 50 `relevant`, and 11 `uncertain` labels; binary metrics exclude the 11 uncertain cases, leaving 189 scored articles.

| System | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| TF-IDF logistic regression | 0.862 | 0.840 |
| Mean-Word2Vec logistic regression | 0.783 | 0.757 |
| Majority baseline | 0.735 | 0.424 |
| Stage 1 keyword-rule reference | 0.873 | 0.851 |

The full per-class metrics and confusion counts are in `outputs/evaluations/stage2_human_review_metrics.csv`; per-article predictions and correctness flags are in `outputs/evaluations/stage2_human_review_predictions.csv`. The final evaluation cell in `notebooks/stage2_models.ipynb` validates the test IDs and regenerates both files. Results describe this one-reviewer, proportionally stratified sample only; they are not population estimates or general real-world accuracy. The Stage 1 keyword-rule labels remain weak labels, so the prior 0.911 TF-IDF and 0.804 Word2Vec macro-F1 scores measure agreement with those rules, not human-validated accuracy.

## STEP 1 — READ AND AUDIT BEFORE CODING

Inspect the entire repository, including:

- The official Stage 2 assignment PDF and course requirements, if present.
- `stage1_analysis.ipynb` and its verified exports.
- `stage2_models.ipynb`.
- `README.md`, Stage 2 dependency files, and existing evaluation artifacts.
- Repository structure, Git status, ignored datasets, and uncommitted changes.

Extract the professor's exact Stage 2 requirements into a compliance matrix:

Requirement | Mandatory/Conditional | Evidence | Status | Missing work | Priority

Distinguish mandatory requirements from optional improvements. Quote or reference the relevant assignment sections. Do not invent requirements or claim compliance based only on code existing.

Preserve all existing Stage 1 outputs and user work. Do not overwrite or delete anything without a compelling reason.

## STEP 2 — VERIFY CURRENT STAGE 2 WORK

The previous local run reported:

- 10/10 executed notebook code cells with zero errors.
- 47,443 source articles screened.
- 47,375 retained articles.
- 38,002 training, 4,720 validation, and 4,653 test articles.
- 68 quarantined articles and 26 algorithmically flagged near-duplicate candidate pairs.
- Word2Vec: 100 dimensions, 95,445 vocabulary items, three epochs.
- Word2Vec held-out context-pair AUC: 0.677.
- TF-IDF macro-F1: 0.911.
- Mean Word2Vec macro-F1: 0.804.
- The Colab export reports Qwen3-1.7B QLoRA, and its adapter, metadata, training curve, metrics, and per-excerpt loss file have been imported into `outputs/`; the artifact check confirms adapter files and metadata exist, with metadata reporting 6,009 steps and 96,158 chunks. The downloaded source/output notebook is `notebooks/Copy_of_stage2_models.ipynb` and has mixed saved outputs: cell 22 ends with `KeyboardInterrupt`, while cell 29's directory check alone is not proof of completion. Treat that cell history as a caveat, not as evidence that the separate export failed. The paired interval has been independently reproduced from 256 test losses that match the manifest exactly. Revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` was recovered from the sole matching Colab cache snapshot and recorded with its source. Fine-tuning and matched-model comparison are now PASS; document the interrupted historical output and unrecorded peak GPU memory as limitations.

These are prior reported results, not automatically verified facts. Check them against the actual saved artifacts and notebook outputs.

Inspect and explain the data screening, chronological split, deduplication, leakage prevention, and Word2Vec training-ID hash.

Critically assess whether the Word2Vec vocabulary, training duration, epochs, preprocessing, and evaluation are appropriate. Improve them only when justified by evidence and resource constraints.

Very importantly: the rule-flag classification metrics evaluate agreement with Stage 1 heuristics, NOT independently verified real-world accuracy. State this explicitly everywhere relevant.

## STEP 3 — DEFINE A SCIENTIFICALLY VALID FINE-TUNING TASK

Before implementing QLoRA, choose and justify a concrete, measurable task that is aligned with Agentic AI Radar and the professor's requirements.

Possible tasks include Agentic Web relevance classification or structured information extraction, but do not assume either is valid without checking the assignment and available labels.

Audit label provenance. If labels originate from keyword rules or model-generated annotations, identify them as weak or silver labels. Never call them human ground truth.

Do not train and evaluate against the same heuristic and then claim that the model has learned genuine semantic understanding.

If reliable gold labels are missing, evaluate only agreement with the available weak or silver labels. Do not present these scores as independent accuracy or semantic understanding.

Ensure that train, validation, and test examples remain isolated, including near duplicates and articles from the same underlying story where feasible.

Document the exact input, expected output, evaluation criteria, and why fine-tuning is appropriate.

## STEP 4 — COMPLETE THE WORD2VEC RESEARCH COMPONENT

Verify and strengthen the existing classical embedding experiment.

Include meaningful, efficient evaluations such as:

- Nearest-neighbor examples for domain-specific words.
- A transparent intrinsic evaluation with suitable limitations.
- Comparisons with TF-IDF or other appropriate baselines.
- Discussion of domain-specific vocabulary and representation limitations.

Avoid claiming that embeddings are semantically accurate merely because nearest neighbors look plausible.

Use clear figures and compact result tables with interpretations tied to Agentic Web.

## STEP 5 — BUILD THE REQUIRED QLORA EXPERIMENT

Implement a clean, executable Google Colab GPU workflow for the exact professor-approved or professor-listed compact open model, provisionally Qwen3-1.7B, using QLoRA.

First verify the official model identifier, license, tokenizer/chat template, compatibility, available GPU memory, and applicable course constraints.

Use a suitable Hugging Face Transformers + PEFT + bitsandbytes training pipeline.

Where appropriate, configure:

- 4-bit quantization.
- LoRA adapter training.
- Gradient accumulation and checkpointing.
- Memory-conscious batch sizes and sequence lengths.
- Deterministic seeds where feasible.
- Explicit train/validation/test separation.
- Evaluation during training.
- Early stopping or justified checkpoint selection.
- Robust checkpoint and artifact saving.

Do not blindly copy hyperparameters. Explain and justify important choices.

Do not download model weights or attempt large-model training on my local 8 GB machine.

Do not claim Colab Pro is necessary until you have checked the actual model and GPU requirements. Prepare the workflow to run on an appropriate available Colab GPU, while documenting differences between free and paid runtimes.

Do not initiate paid services, large downloads, or expensive training runs without asking for my approval.

If GPU access is unavailable, prepare the notebook and mark fine-tuning as NOT RUN. Never invent results.

## STEP 6 — EVALUATE THE MODEL PROPERLY

Once real GPU training has completed, compare the adapted model with the unadapted base model on the SAME held-out evaluation set and with consistent prompts and decoding settings.

Also include an appropriate classical baseline when meaningful.

Select metrics that fit the actual task:

- Classification: macro-F1, per-class precision/recall, confusion matrix, class support.
- Structured extraction: entity/relation precision, recall, F1, schema validity, evidence grounding.
- Generative tasks: task-specific metrics plus transparent qualitative error analysis.

Only use metrics appropriate to the chosen task.

Where feasible, provide uncertainty estimates or confidence intervals, and explain their limitations.

Include representative successes, failures, ambiguous cases, and at least one serious error analysis.

Distinguish weak-label agreement, source-span grounding, semantic correctness, and independent human evaluation.

Do not leak test examples into model selection or prompt engineering.

Report actual GPU type, runtime, peak memory if measurable, trainable parameter count, training duration, and any cost estimates with their assumptions.

## STEP 7 — MAKE THE NOTEBOOK EXCELLENT ACADEMICALLY

Structure `stage2_models.ipynb` as a polished, self-contained research notebook:

1. Title, project context, individual contribution, and AI-tool disclosure.
2. Research questions and hypotheses.
3. Official requirement mapping.
4. Data provenance and Stage 1 linkage.
5. Corpus screening and leakage controls.
6. Chronological split and descriptive statistics.
7. Word2Vec methodology and evaluation.
8. Baseline experiments.
9. Fine-tuning task and label provenance.
10. QLoRA architecture, setup, and training procedure.
11. Base-versus-adapted model comparison.
12. Error analysis and limitations.
13. Reproducibility, resource usage, and artifacts.
14. Conclusions answering each research question.
15. Final compliance audit.

Explain WHY each method is used, WHAT the outputs mean, and WHAT cannot be concluded from them.

Every major figure should have a clear title, readable axes, appropriate units, and a concise scientific interpretation.

Make it easy for a professor to inspect results without rerunning costly training.

Avoid excessive boilerplate, repetitive tables, decorative visualizations, and claims that go beyond the evidence.

## STEP 8 — REPRODUCIBILITY AND ARTIFACTS

Use clean relative paths, documented dependencies, fixed random seeds where feasible, and cached expensive outputs.

Keep generated artifacts organized, including:

- Data split manifests and checksums.
- Word2Vec model and evaluation outputs.
- Baseline predictions and metrics.
- QLoRA configuration and adapter checkpoints.
- Base/adapted evaluation predictions.
- Training logs and runtime information.
- Evaluation templates and review documentation.

Do not commit secrets, API keys, large model weights, or unnecessary generated datasets.

Respect `.gitignore`, and ensure the README explains how to obtain or regenerate ignored Parquet files and other required artifacts.

Do not rerun paid OpenAI API extraction from Stage 1 unless it is strictly required and I approve it.

Stage 2 must not depend on access to my private OpenAI key merely to inspect the saved results.

## STEP 9 — INDEPENDENT QUALITY AUDIT

After implementing the work, conduct a skeptical review as if you were a demanding university examiner.

Check for:

- Missing mandatory requirements.
- Incorrect interpretations of metrics.
- Weak-label circularity.
- Data leakage and near-duplicate contamination.
- Unsupported causal claims.
- Invalid model comparisons.
- Unexecuted cells.
- Hidden API or cloud dependencies.
- Unreproducible results.
- Overfitting and cherry-picked examples.
- Unclear individual contributions.
- Missing limitations and negative findings.

Run all feasible local checks and inspect saved outputs.

Do not describe GPU-dependent work as verified until it has actually run.

## STEP 10 — REPORT CLEARLY AND HONESTLY

At the end, provide:

A. Exact work completed and files changed.

B. What was executed and verified locally.

C. What still requires GPU execution or other external resources.

D. A requirement-by-requirement compliance matrix with PASS, PARTIAL, FAIL, or NOT VERIFIED.

E. The three highest-impact remaining improvements for academic quality.

F. A precise Google Colab execution checklist, including the required artifacts to download and return to the repository.

G. Any methodological weaknesses that could cost points, even if the code runs without errors.

H. A concise explanation of the Stage 2 research findings in language I can confidently explain to my professor.

## EXECUTION RULES

Work systematically and make real progress without unnecessary interruptions.

Ask me only when a decision requires spending money, using unavailable hardware, choosing between materially different research objectives, or supplying genuine human labels.

Never claim a score of 25/25 or promise a grade.

Do not start Stage 3.

Start by auditing the actual repository and official Stage 2 brief. Then implement the highest-priority verified requirements first. Preserve existing working code and artifacts, and finish with a rigorous evidence-based audit.

NEVER TOUCH STAEGE 1

After finishing the work in this plan, read `Stage2_self_audit.md` and immediately begin the audit and improvement loops described there. Do not wait for me to ask.

YOU NEED TO MAKE SURE OUR WORK IS PERFECT, STAGE 2 SHOULD BE PERFECT