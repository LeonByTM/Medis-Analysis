# Agentic AI Radar — Computational Language Technologies

**Research theme:** Tracking the Rise of Autonomous AI Agents  
**Chosen assignment track:** Agentic Web  
**Primary data source:** [AI Media Dataset (Kaggle)](https://www.kaggle.com/datasets/jannalipenkova/ai-media-dataset)

## Objective
Build a reproducible NLP research pipeline to investigate how AI agents are discussed in media coverage. Analyze topics, entities, and relationships; construct a knowledge graph; train and evaluate language representations; and test whether graph-enhanced retrieval improves evidence-grounded question answering.

## Assignment alignment
The project is organized into **three documented Jupyter notebooks**. These are the main deliverables, not an external dashboard or report.

| Notebook | Core purpose |
| --- | --- |
| `notebooks/stage1_analysis.ipynb` | Data preparation, exploratory analysis, topic modeling, entity/relation extraction, knowledge graph |
| `notebooks/stage2_models.ipynb` | Word embeddings, compact language-model adaptation, evaluation |
| `notebooks/stage3_rag.ipynb` | Baseline RAG, graph-enhanced RAG, controlled comparison |

**Scope guardrails:** Use the assigned Kaggle dataset. Stage 1 and Stage 3 may emphasize Agentic Web articles, but Stage 2 should use the **full assigned dataset**, based on the previously discussed assignment instructions. Do not substitute a finance-news dataset or introduce live data as the primary source. There is **no Streamlit scope in this plan**.

> **Verification note:** The authoritative assignment is the professor's official brief, `CLT_AI_Trend_Project.pdf`. This pack is a working specification derived from earlier discussion; where it disagrees with the PDF, the PDF governs. Recheck deadlines and submission guidance (ILIAS) before submitting. Stage 1 is an individual submission by Leon Bytyqi (confirmed by the student).

## Status
- **Stage 1** (`notebooks/stage1_analysis.ipynb`) is implemented and was executed top to bottom (30/30 code cells, 0 errors, 28/28 structural checks). It is an **individual submission by Leon Bytyqi**. Audit evidence: `audits/PROFESSOR_REQUIREMENTS.md` (requirement-by-requirement status) and `audits/STAGE1_IMPROVEMENT_PLAN.md`.
- **Stage 2** (`notebooks/stage2_models.ipynb`) has local results for its split, Word2Vec, embedding evaluation, and weak-label prediction audit. QLoRA training and evaluation completed in Colab; the adapter and results were verified in Drive on 2026-10-19 but have not yet been copied into this local project. Rule-flag classification scores measure agreement with Stage 1's heuristic, not independent Agentic Web accuracy. **Stage 3** has not been started.
- **AI use:** see the "AI tools and LLM use" section below.

## AI tools and LLM use (disclosure)
- **Code drafting:** GitHub Copilot and AI language models drafted the code for Stage 1. Leon Bytyqi directed the project, specified the analytical objectives, oversaw implementation, and is responsible for reviewing the results, interpreting the findings and documenting limitations. AI tools were also used for debugging and documentation.
- **OpenAI LLM in the analysis:** the OpenAI model `gpt-4o-mini` (temperature 0), called through the LangExtract library, was used for a structured entity and relation extraction **pilot on 50 articles** (notebook section 7.1). Text from those 50 public-dataset articles was sent to OpenAI. The outputs are unreviewed candidates and are compared with the rule-based extractor by overlap only; **no accuracy is claimed**, because no human labels exist.
- **API key:** stored only in a git-ignored `.env` file (`LANGEXTRACT_API_KEY`); it is never part of the notebook, outputs or repository. Raw LLM responses are cached locally in `outputs/llm_cache/` (also git-ignored).
- The full disclosure is in the introduction of `notebooks/stage1_analysis.ipynb`.

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
6. Stage 2's local embedding workflow and saved outputs are in `notebooks/stage2_models.ipynb` and `outputs/`. Its QLoRA section is blocked by default in the local notebook; the QLoRA run was completed in Colab, and its artifacts still need to be copied from Drive into this project. Stage 3 has not been started.

## Quick start (Stage 2)
1. Stage 2 expects Stage 1's full-corpus `data/processed/articles_clean.parquet` and `data/processed/stage1_metadata.json`. These are local/ignored artifacts; if missing, regenerate them using the Stage 1 quick start above. Stage 2 does not require the private OpenAI key.
2. Use Python 3.11 and install the pinned local dependencies: `python -m pip install -r requirements-stage2.txt`.
3. Open `notebooks/stage2_models.ipynb` from the repository root and run top to bottom. The local workflow verifies or loads the cached Word2Vec model, writes the chronological split and evaluation files under `outputs/evaluations/`, and saves tables/figures under `outputs/`.
4. The rule-label prediction file is diagnostic: its scores compare models with Stage 1's weak rule labels and are not independent accuracy measurements.
5. QLoRA is GPU-dependent and disabled by default (`ENABLE_QLORA_RUN = False`). For a deliberately authorized run, use a suitable CUDA-enabled Colab or equivalent runtime, install `requirements-stage2-colab.txt` without replacing its CUDA-matched PyTorch, set `AGENTIC_RADAR_ROOT`, inspect GPU memory/cost, and only then opt in. Colab Pro is not assumed to be necessary. Do not run this workload on the local 8-GiB machine.

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
├── audits/                         # requirement audit and improvement plan
├── notebooks/
│   ├── stage1_analysis.ipynb
│   ├── stage2_models.ipynb         # embeddings complete locally; QLoRA results from Colab pending sync
│   └── stage3_rag.ipynb            # (planned)
├── data/processed/                 # cleaned corpus, Agentic Web subset, metadata
├── outputs/{figures,tables,graph,evaluations}/
├── src/                            # (planned, optional support code)
└── tests/                          # (planned)
```

The `src/` package is optional support code. **Notebook cells must clearly explain methodology, show results, and link to reproducible artifacts.**

## Research questions
Project-wide questions (RQ1–RQ3 are answered by Stage 1, see section 11.1 of the Stage 1 notebook, which words them slightly more specifically; RQ4–RQ5 belong to Stages 2–3 and are not yet addressed):
- RQ1: How does Agentic Web coverage evolve over the dataset's observed time range?
- RQ2: Which organizations, tools, models, and concepts co-occur or participate in explicitly stated relationships?
- RQ3: Which latent themes characterize Agentic Web reporting?
- RQ4: Do domain-adapted representations improve the chosen intrinsic or downstream evaluation?
- RQ5: Does knowledge-graph augmentation improve retrieval-grounded answers compared with the same baseline RAG system?

**Important:** Media attention is not equivalent to real-world adoption or market impact. Report coverage trends as coverage trends.

## Definition of done
All three notebooks execute in order on documented hardware; each has markdown explanations, methodology, plots/tables, evaluation, limitations, and saved outputs. Every chart and quantitative claim is traceable to a reproducible cell. Dataset provenance and AI-assistance disclosure are included.
