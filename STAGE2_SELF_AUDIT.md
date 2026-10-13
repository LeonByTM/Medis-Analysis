# Stage 2 Self-Audit and Improvement Instructions

## Mission

Immediately audit, improve, and verify the existing Stage 2 work for **Agentic AI Radar — Agentic Web**. Stage 1 is finished and Stage 2 is now in progress. Do not stop after writing a review or a plan: inspect the real files and saved results, make the smallest justified Stage 2 repairs, run every feasible check, inspect the resulting evidence, and repeat.

Work autonomously without asking the student for routine approval or clarification. Make safe, reversible decisions within this scope and document assumptions. Do not start Stage 3.

## Authority and scope

1. Read `CLT_AI_Trend_Project.pdf` and the project requirements before changing files. The professor's formal assignment is authoritative for grading. Use `PROJECT_SPEC.md`, `DATA_CONTRACT.md`, `EVALUATION_PROTOCOL.md`, `IMPLEMENTATION_PLAN.md`, `CODING_AGENT_INSTRUCTIONS.md`, and this repository's current documentation as supporting context.
2. Distinguish formal mandatory requirements from lecture recommendations and optional improvements. In particular, do not present the Course 3–6 lecture examples, tools, metrics, or suggested target counts as formal requirements unless the professor's assignment confirms them.
3. Work only on Stage 2 and directly related Stage 2 support files, configuration, tests, and documentation. **Never modify Stage 1 notebooks, code, exports, or artifacts.** Read Stage 1 outputs only as needed to verify Stage 2 input provenance and compatibility; preserve them.
4. Preserve existing user work, including uncommitted changes. Inspect `git status` before editing, do not overwrite unrelated work, and do not delete or regenerate valuable artifacts without a clear, necessary reason.
5. Do not begin Stage 3, build a UI, change the assigned dataset, or rerun paid API extraction.

## Inspect evidence before trusting claims

Inspect the complete Stage 2 notebook (`notebooks/stage2_models.ipynb`), its saved cell outputs, related code and dependency files, actual input data and Stage 1 exports, generated Stage 2 artifacts, and Git status. A successful cell run or a statement in a prior plan is not proof that a method is valid or that an artifact contains the claimed result.

The following are **unverified prior-run claims**, not accepted facts. Trace each applicable claim to notebook outputs and reloadable artifacts, correct it if needed, and report discrepancies:

- 47,443 source articles screened; 47,375 retained; 68 quarantined.
- 38,002 training, 4,720 validation, and 4,653 test articles.
- 26 algorithmically flagged near-duplicate candidate pairs.
- Word2Vec with 100 dimensions, 95,445 vocabulary items, and three epochs; held-out context-pair AUC 0.677.
- TF-IDF macro-F1 0.911 and mean-Word2Vec macro-F1 0.804.
- Qwen3-1.7B QLoRA was not run because CUDA GPU access was unavailable.

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
