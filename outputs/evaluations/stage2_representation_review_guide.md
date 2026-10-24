# Stage 2 Representation Review Guide

## Purpose

This worksheet supports an independent human check of the existing Stage 2 extrinsic classifiers. It contains a reproducible, blinded sample of 200 articles from the held-out test partition. The keyword-rule labels and all model predictions are intentionally omitted from the worksheet.

## Label each article

Use the supplied URL or consult the full local article by `article_id` in `data/processed/articles_clean.parquet`. The worksheet intentionally omits article text to avoid duplicating the source corpus. Do not guess; use `uncertain` if the article cannot be judged from the available evidence.

- `relevant`: the article substantively focuses on AI agents or agentic systems, including their development, capabilities, deployment, protocols, frameworks, or tools.
- `not_relevant`: AI agents/agentic systems are absent or only mentioned incidentally; they are not a substantive topic of the article.
- `uncertain`: the available evidence is insufficient or the article's focus is genuinely ambiguous.

Fill `manual_agentic_web_label` with exactly one allowed value and enter a stable `reviewer_id`. Add a short rationale for `uncertain` labels. Do not change article IDs or source excerpts, and do not consult model predictions or keyword-rule labels while assigning the human labels.

## Sampling and interpretation

The 200 rows were selected only from the existing chronological test partition, stratified proportionally by publication month and the existing Stage 1 keyword-rule flag, with seed `20261023`. The flag was used only to ensure both rule-label groups are represented; it is not shown in the worksheet. Rows were shuffled after selection. Preserve the labels as human review, not as model output.

After review, evaluate the already-saved TF-IDF, mean-Word2Vec, and majority-baseline predictions against the human labels by joining on `article_id`. Report the reviewed count, excluded `uncertain` count, class support, and metrics on this stratified sample. Do not describe the result as a population estimate unless using appropriate sampling weights. This sample does not replace or relabel the full test set.
