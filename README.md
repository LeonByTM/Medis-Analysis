# Agentic AI Radar — Computational Language Technologies

**Research theme:** Tracking the Rise of Autonomous AI Agents  
**Chosen assignment track:** Agentic Web  
**Primary data source:** [AI Media Dataset (Kaggle)](https://www.kaggle.com/datasets/jannalipenkova/ai-media-dataset)

> From headlines to evidence: tracing how autonomous agents are discussed, connected, and queried.

## Objective
How are autonomous AI agents entering the news—and can structured knowledge help answer questions about that coverage? Agentic AI Radar investigates this through a reproducible, three-stage NLP study of the assigned AI Media Dataset: map the discourse, evaluate language representations, and test text-only against graph-enhanced retrieval. The project treats media attention as evidence about reporting, not proof of real-world adoption.

## Assignment alignment
The project is organized into **three documented Jupyter notebooks**. These are the main deliverables, not an external dashboard or report.

| Notebook | Core purpose |
| --- | --- |
| `notebooks/stage1_analysis.ipynb` | Data preparation, exploratory analysis, topic modeling, entity/relation extraction, knowledge graph |
| `notebooks/stage2_models.ipynb` | Word embeddings, compact language-model adaptation, evaluation |
| `notebooks/stage3_rag.ipynb` | Baseline RAG, graph-enhanced RAG, controlled comparison |

**Scope guardrails:** Use the assigned Kaggle dataset. Stage 1 and Stage 3 may emphasize Agentic Web articles, but Stage 2 should use the **full assigned dataset**, based on the previously discussed assignment instructions. Do not substitute a finance-news dataset or introduce live data as the primary source. There is **no Streamlit scope in this plan**.

> **Verification note:** The authoritative assignment is the professor's official brief, `CLT_AI_Trend_Project.pdf`. This pack is a working specification derived from earlier discussion; where it disagrees with the PDF, the PDF governs. Recheck deadlines and submission guidance (ILIAS) before submitting. The full three-stage project is an individual submission by Leon Bytyqi (confirmed by the student).

## Status
- **Stage 1** (`notebooks/stage1_analysis.ipynb`) is implemented and was executed top to bottom (30/30 code cells, 0 errors, 28/28 structural checks). Audit evidence: `audits/Stage1_Audit.md` (requirement-by-requirement status).
- **Stage 2** (`notebooks/stage2_models.ipynb`) has local split, Word2Vec, embedding, and weak-label results. The local environment did not have a CUDA GPU available for QLoRA, so the training and evaluation were run in **Google Colab** on a Tesla T4 ([Colab notebook](https://colab.research.google.com/drive/1qDJkr9GExcs_2rskXdb2aoeB3KvzywOS)). The resulting adapter and run artifacts were exported from Colab into this project; the downloaded source notebook is retained as `notebooks/Copy_of_stage2_models.ipynb`. The imported artifacts report pretrained/adapted test mean token loss of 3.3533/2.7155 and perplexity of 28.598/15.113 on the same 256 test excerpts; the paired loss delta is -0.6378 (95% CI [-0.6951, -0.5880], independently recomputed from the per-excerpt losses). These are next-token prediction metrics, not semantic or factual accuracy. The exported adapter is in the local, git-ignored `outputs/models/qwen3_1.7b_agentic_media_qlora/`; base-model weights are not included. The downloaded Colab notebook contains mixed saved outputs, including an interrupted training-cell execution. A later artifact check confirms the adapter files and training metadata exist; the metadata reports 6,009 steps and 96,158 chunks. The exact model revision, `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, was recovered from the active Colab cache, where `refs/main` matched the sole cached snapshot. The reported comparison is supported by the matching test manifest, per-excerpt losses, and revision-pinned model metadata; limitations are documented in the Stage 2 audit.
- **Stage 2 independent evaluation (24 October 2026):** a reviewer-labeled sample of 200 held-out test articles was completed (189 determinate labels; 11 `uncertain`). Against the reviewer labels, TF-IDF accuracy/macro-F1 are 0.862/0.840, mean-Word2Vec 0.783/0.757, the majority baseline 0.735/0.424, and the Stage 1 keyword-rule reference 0.873/0.851. The evaluation requirement is marked PASS in `audits/Stage2_self_audit.md`. These are single-reviewer sample results, not population estimates or general real-world accuracy.
- **Stage 3** asks whether knowledge-graph context improves RAG answers about Agentic Web coverage. The reproducible workflow uses 100 passages and 200 generated Q&A candidates; auditor `leon_review` checked all 200 (174 supported, 25 uncertain, 1 unsupported). This pool audit does not automatically approve questions for the frozen benchmark: that comparison remains based on 16 curated decisions (8 approved, 8 rejected) and 8 approved answerable questions. On this set, text-only and graph-expanded retrieval tie on Recall@5 (1.00), while graph expansion has lower MRR@5 (0.906 vs. 0.806); both variants receive identical answer ratings (5 fully correct, 2 partial, 1 incorrect). That negative result is reported rather than framed as a graph win. A separate multi-hop/unanswerable addendum finds partially supported multi-hop answers and correct abstention on both unanswerable questions; 6/10 reviewed graph edges are fully supported. A date-context bug that truncated `Dec. 15` was fixed and regression-tested. Separate prompt-v3 trials improve date citations but remain outside baseline metrics; 400 retrieval diagnostics remain provisional. The notebook was rerun with API flags disabled; spend remains $0.3123534 of the $10 cap. Results remain exploratory and Stage 3 remains PARTIAL. See `audits/Stage3_self_audit.md`.
- **AI use:** see the "AI tools and LLM use" section below.

**Consistent-prompt note:** The all-eight `source_precise_v3` comparison has zero deltas on applicable answer-quality measures; abstention is not applicable because all eight questions are answerable.

## AI tools and LLM use (disclosure)
- **Code drafting:** GitHub Copilot and AI language models assisted with code drafting across Stages 1–3, as well as debugging and documentation. Leon Bytyqi directed the project, specified the analytical objectives, oversaw implementation, and is responsible for reviewing the results, interpreting the findings and documenting limitations.
- **OpenAI LLM in the analysis:** the OpenAI model `gpt-4o-mini` (temperature 0), called through the LangExtract library, was used for a structured entity and relation extraction **pilot on 50 articles** (notebook section 7.1). Text from those 50 public-dataset articles was sent to OpenAI. The outputs are unreviewed candidates and are compared with the rule-based extractor.
- **API key:** stored only in a git-ignored `.env` file (`LANGEXTRACT_API_KEY`); it is never part of the notebook, outputs or repository. Raw LLM responses are cached locally in `outputs/llm_cache/` (also git-ignored).
- The Stage 1 notebook introduction records its analysis-specific disclosure; this README summarizes AI assistance across the project.

## Dependency files
The four files serve different purposes; they are not meant to be installed together. Use only the file for the environment and workflow you are running.

| File | Use |
| --- | --- |
| `requirements.txt` | Broad, mostly unpinned baseline; reference for the overall project, not the reproducible environment for a specific stage. |
| `requirements-stage1.txt` | Pinned local environment for Stage 1. |
| `requirements-stage2.txt` | Pinned local environment for Stage 2 embeddings and evaluation. |
| `requirements-stage2-colab.txt` | Colab-only additions for GPU QLoRA. Install in a CUDA-enabled Colab runtime and preserve its CUDA-matched PyTorch; do not combine with the local requirements files. |

The Stage 1 and local Stage 2 quick starts below show which pinned file to install. For Colab QLoRA, install only `requirements-stage2-colab.txt` after selecting the GPU runtime.

## Quick start (Stage 1)
1. Create a Python 3.11 environment and install `requirements-stage1.txt` (tested versions), then `python -m spacy download en_core_web_sm`. (`requirements.txt` is the broader baseline for later stages.)
2. The notebook downloads the dataset itself with `kagglehub` (`jannalipenkova/ai-media-dataset`); no files need to be placed in `data/raw/`, and original files are never modified.
3. Optional, for the LangExtract pilot only: create a git-ignored `.env` file in the project root containing `LANGEXTRACT_API_KEY=<your key>`. Without a key the pilot uses its local cache (`outputs/llm_cache/`) if present, otherwise it is skipped.
4. Run `notebooks/stage1_analysis.ipynb` top to bottom (about 25 minutes on CPU; entity extraction dominates).
5. Outputs are written to `data/processed/` and `outputs/` (tables, figures, graph, evaluations), with metadata in `data/processed/stage1_metadata.json`.
6. Stage 2's local embedding workflow and imported Colab QLoRA results are in `notebooks/stage2_models.ipynb` and `outputs/`. The local QLoRA training cells remain disabled by default and their saved `BLOCKED` outputs describe only the local run; the separate Colab results are summarized in the notebook's artifact-backed results section. The adapter is ignored by Git and must be downloaded separately if needed. Stage 3's text-only/graph-enhanced RAG comparison, enhancement, and limitations are in `notebooks/stage3_rag.ipynb`; its 200-candidate human-review summary is recorded in the notebook and in `outputs/evaluations/stage3_human_review_local_audit.csv`. The frozen baseline remains a separate, smaller curated evaluation set. OpenAI calls are disabled by default. See `audits/Stage3_self_audit.md`.

## Quick start (Stage 2)
1. Stage 2 expects Stage 1's full-corpus `data/processed/articles_clean.parquet` and `data/processed/stage1_metadata.json`. These are local/ignored artifacts; if missing, regenerate them using the Stage 1 quick start above. Stage 2 does not require the private OpenAI key.
2. Use Python 3.11 and install the pinned local dependencies: `python -m pip install -r requirements-stage2.txt`.
3. Open `notebooks/stage2_models.ipynb` from the repository root and run top to bottom. The local workflow verifies or loads the cached Word2Vec model, writes the chronological split and evaluation files under `outputs/evaluations/`, and saves tables/figures under `outputs/`.
4. The rule-label prediction file is diagnostic: its scores compare models with Stage 1's weak rule labels and are not independent accuracy measurements.
5. The exported Colab QLoRA metrics, evaluation manifest, per-excerpt loss file, training curve, plots, and metadata are under `outputs/`; the adapter is under `outputs/models/qwen3_1.7b_agentic_media_qlora/` and is git-ignored. The base model is not bundled. The downloaded Colab notebook/source is `notebooks/Copy_of_stage2_models.ipynb`; although one saved training-cell execution is interrupted, an artifact-check cell confirms the adapter files and metadata, which reports 6,009 steps and 96,158 chunks (see the Stage 2 audit). The paired confidence interval was independently recomputed from the 256 per-excerpt losses, and the model revision is pinned in metadata. `ENABLE_QLORA_RUN = False` remains the safe default: do not start another training run merely to view the saved results. Do not run this workload on the local 8-GiB machine.

## Stage 3 — From Agentic Web stories to grounded answers

Can a knowledge graph help a RAG system answer questions about agentic AI coverage? Stage 3 tests that idea against a text-only baseline instead of assuming graph context is better.

### The experiment
- The pipeline uses 100 representative passages from the assigned dataset and generated 200 Q&A candidates with a question model distinct from the answer model.
- Auditor `leon_review` checked all 200 candidates: **174 supported, 25 uncertain, and 1 unsupported**. Uncertain and unsupported items are not approvals; the review outcomes are recorded in [`stage3_human_review_local_audit.csv`](outputs/evaluations/stage3_human_review_local_audit.csv).
- The frozen baseline is deliberately smaller: a curated worksheet contains 16 decisions (8 approved and 8 rejected), and the 8 approved answerable questions are used for paired baseline metrics. The pool-level audit does not add questions to that frozen benchmark.
- The text-only system uses TF–IDF retrieval. The graph variant adds bounded article expansion through `MENTIONED_WITH` co-occurrence edges. Graph edges are retrieval cues only; generated answers must be grounded in and cite source article text.

### What the comparison found

| Metric | Text-only | Graph-expanded |
| --- | ---: | ---: |
| Recall@5 | 1.00 | 1.00 |
| MRR@5 | 0.906 | 0.806 |
| Fully correct answers | 5/8 | 5/8 |
| Partially correct answers | 2/8 | 2/8 |
| Incorrect answers | 1/8 | 1/8 |

On this small benchmark, graph expansion did not improve recall or answer ratings, and its MRR was lower. That negative result is part of the finding—not something to hide or overgeneralize.

### Enhancement and limitations

A date-context bug that split `Dec. 15` incorrectly was fixed and regression-tested. Separate prompt-v3 diagnostics improved the affected date answers, but they are reported separately and do not change the frozen baseline scores. A separate addendum tests one multi-hop and one unanswerable question; it also remains outside the baseline metrics. With only 8 approved answerable questions in the paired baseline, all results are exploratory. The notebook and the [Stage 3 self-audit](audits/Stage3_self_audit.md) document the methods, outputs, and caveats.

## Repository structure
Items marked *(planned)* do not exist yet.
```text
.
├── README.md
├── (local only) CLT_AI_Trend_Project.pdf   # professor's official brief (authoritative; not redistributed)
├── requirements.txt                # broad baseline
├── requirements-stage1.txt         # tested Stage 1 versions
├── requirements-stage2.txt         # pinned local Stage 2 environment
├── requirements-stage2-colab.txt   # Colab GPU QLoRA environment
├── .gitignore                      # excludes secrets, local working notes, and LLM caches
├── audits/
│   ├── Stage1_Audit.md             # Stage 1 assignment audit
│   ├── Stage2_self_audit.md        # Stage 2 self-audit
│   └── plan_stage2.md              # Stage 2 implementation plan
├── notebooks/
│   ├── stage1_analysis.ipynb
│   ├── Copy_of_stage2_models.ipynb # downloaded Colab source and saved execution outputs
│   ├── stage2_models.ipynb         # embeddings and imported Colab QLoRA results
│   └── stage3_rag.ipynb            # text vs. graph RAG, human-review summary, and evaluation
├── stage3_rag.py                   # reusable Stage 3 retrieval, answer, and evaluation helpers
├── data/processed/                 # cleaned corpus, Agentic Web subset, metadata
├── outputs/{figures,tables,graph,evaluations}/
├── outputs/models/                 # local ignored model weights; not included in Git
├── src/                            # (planned, optional support code)
└── tests/
    └── test_stage3_rag.py          # focused Stage 3 regression tests
```

The `src/` package is optional support code. **Notebook cells must clearly explain methodology, show results, and link to reproducible artifacts.**

## Research questions
Project-wide questions (RQ1–RQ3 are addressed by Stage 1; RQ4 by Stage 2; and RQ5 by Stage 3. See the notebooks for methods, results, and limitations):
- RQ1: How does Agentic Web coverage evolve over the dataset's observed time range?
- RQ2: Which organizations, tools, models, and concepts co-occur or participate in explicitly stated relationships?
- RQ3: Which latent themes characterize Agentic Web reporting?
- RQ4: Do domain-adapted representations improve the chosen intrinsic or downstream evaluation?
- RQ5: Does knowledge-graph augmentation improve retrieval-grounded answers compared with the same baseline RAG system?

**Important:** Media attention is not equivalent to real-world adoption or market impact. Report coverage trends as coverage trends.

## Definition of done
All three notebooks execute in order on documented hardware; each has markdown explanations, methodology, plots/tables, evaluation, limitations, and saved outputs. Every chart and quantitative claim is traceable to a reproducible cell. Dataset provenance and AI-assistance disclosure are included.
