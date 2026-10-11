# Human review guide for the Stage 1 worksheets (optional)

**Status.** The three worksheets in this folder are unlabelled. Human review is an *optional* quality improvement: the professor's Stage 1 brief does not require human annotation. Until it is done, no precision, recall, F1 or agreement figure exists, and every article-selection decision, entity and relation candidate is an unvalidated model output. Do not enter labels you did not decide yourself, and do not report any metric that was not computed from labels you entered.

**Files.**

| Worksheet | Rows | What you judge |
|---|---|---|
| `stage1_review.csv` | 40 (8 per stratum) | Is the article about the Agentic Web? |
| `entity_review.csv` | 56 | Is the extracted span a correct, correctly typed and correctly normalized entity? |
| `relation_review.csv` | 51 | Does the evidence sentence support the directed candidate relation? |

147 judgements in total (40 + 56 + 51). A smaller subset reviewed honestly is better than all rows reviewed carelessly; report how many you reviewed.

## How the notebook uses your labels

Edit the CSV in place (spreadsheet or text editor; keep UTF-8 and the header row). When any `manual_label` is non-empty, the notebook treats that file as the source of truth, does **not** overwrite it on re-run, and computes the metrics below only from rows you labelled. Keep a backup copy before editing.

## 1. `stage1_review.csv` — article selection

Read `review_text_excerpt` (title + first 500 characters); open the URL via the dataset only if needed. Fill `manual_label` with one of:

- `relevant` — the article is substantially about autonomous AI agents, agentic workflows, multi-agent systems, agent tools/protocols/frameworks (the Agentic Web track).
- `not_relevant` — "agent" is used in another sense (insurance, real estate, customer-service human agents) or the article only mentions agents in passing.
- `uncertain` — you cannot decide from the excerpt.

Use `reviewer_notes` for borderline reasoning. The sample is **stratified**, not random: 8 articles from each of five strata (selected by title phrase, selected by 1–2 body phrases, selected by 3+ body phrases, not selected but agent-tagged, not selected with no signal). Do not read the pooled accuracy of 40 rows as the corpus precision.

*Estimating precision of the selection rule* (selected strata only): per stratum, `precision_s = relevant / (relevant + not_relevant)`, excluding `uncertain`; combine strata weighted by the stratum's population size `N_s` (counts of selected articles per stratum): `precision = Σ N_s · precision_s / Σ N_s`.
*Estimating recall* is only possible with the two non-selected strata, weighted by their population sizes in the cleaned corpus: estimated missed relevant articles `M = Σ N_s · (relevant_s / labelled_s)`, then `recall = TP / (TP + M)` where `TP` is the estimated number of relevant selected articles. With 8 rows per stratum these estimates are very uncertain: report the 95% Wilson interval for each stratum proportion and do not present a single number without it.

## 2. `entity_review.csv` — entity spans and normalization

Read `context_text`, `surface_form`, `canonical_name_predicted`, `entity_type_predicted`. Fill `manual_label`:

- `correct` — span, type and canonical name are all right.
- `wrong_span` — not an entity, or boundaries wrong.
- `wrong_type` — entity, but the type is wrong (fill `gold_entity_type`).
- `wrong_link` — span/type fine but the canonical name merges or splits entities wrongly (fill `gold_canonical_name`).
- `uncertain` — cannot decide.

*Metric:* mention-level precision = `correct / (correct + wrong_span + wrong_type + wrong_link)`, excluding `uncertain`; report separately for `extraction_method` values (`domain_alias_lexicon` vs `spacy_en_core_web_sm`), because their error profiles differ. **Recall cannot be estimated from this sample** (it contains only extracted mentions); estimating it needs a separate set of articles whose entities you annotate exhaustively.

## 3. `relation_review.csv` — directed relation candidates

Read `evidence_text` (the exact sentence), `subject_predicted`, `predicate_predicted`, `object_predicted`. Judge only what the sentence states, not world knowledge. Fill `manual_label`:

- `supported` — the sentence states that subject → predicate → object, in that direction.
- `unsupported` — the sentence does not state it (negated, hypothetical beyond the stated modality, or the entities are not the ones related).
- `wrong_direction` — related, but subject and object are reversed.
- `wrong_predicate` — related, but a different predicate fits (fill `gold_predicate`).
- `uncertain`.

Fill `gold_subject_id` / `gold_object_id` when the extracted entities are wrong. Event time is *not* extracted; if the sentence gives one you may note it in `reviewer_notes`, but never copy the publication date as event time.

*Metric:* candidate precision = `supported / (supported + unsupported + wrong_direction + wrong_predicate)`, excluding `uncertain`; also report a lenient version counting `wrong_direction` and `wrong_predicate` as partially correct, and per-predicate counts (many predicates have too few rows to interpret). Recall is not estimable from candidate-only samples.

## 3b. `llm_relation_review.csv` — LangExtract pilot relations (optional)

Same labels and rules as `relation_review.csv`, applied to the relations the LangExtract pilot extracted and that passed grounding validation (about 50 rows). Grounding only proves the evidence span names both entities; it does not prove the sentence states the relation, so judge the sentence exactly as in section 3. Useful if you want to compare the LLM with the template extractor: after labelling both, compare `supported / (supported + unsupported + wrong_direction + wrong_predicate)` for each worksheet, state each n, and do not claim a difference without intervals. The two worksheets sample different relations (the LLM sheet covers 50 pilot articles), so the comparison is indicative only.

## 4. Quality practice

- **Independence:** label without looking at other model outputs. If possible, have a second annotator label at least 20 rows of each worksheet and compute Cohen's κ = (p_o − p_e) / (1 − p_e) on the shared rows; report κ and disagreements, and do not report κ if you have no second annotator.
- **Uncertainty:** with n ≈ 30–50, intervals are wide (a 95% Wilson interval for 20/25 correct is roughly 61–91%). Always state n and the interval.
- **Provenance:** record who labelled, when, and the guideline version (this file) in the notes or the report. Mark any labels produced with AI assistance as such; they are not human gold labels.
- **After labelling:** re-run the notebook's evaluation cell (section 9) to compute the metrics from your labels; the compliance table then reports the labelled counts.
