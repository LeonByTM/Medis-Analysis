"""Utilities for the Agentic Web Stage 3 retrieval-augmented QA notebook."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer


AUTHORING_MODEL = "gpt-4o-mini"
ANSWER_MODEL = "gpt-4o"
API_BUDGET_USD = 10.0
QUESTIONS_PER_PASSAGE = 2
QUESTION_TYPES = ("factoid", "temporal", "relational", "analytical", "comparative")
QUESTION_MODEL_PRICES = {"input": 0.15, "output": 0.60}
ANSWER_MODEL_PRICES = {"input": 2.50, "output": 10.00}
OPENAI_PRICING_SOURCE = "https://platform.openai.com/docs/pricing"
OPENAI_PRICING_CHECKED = "2026-10-20"


@dataclass(frozen=True)
class RetrievedArticle:
    article_id: str
    title: str
    published_at: str
    source: str
    text_score: float
    retrieval_score: float
    graph_support: float
    context: str


def _atomic_csv_write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary_path, index=False)
    temporary_path.replace(path)


def _normalized_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _normalized_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def load_agentic_corpus(project_root: Path) -> pd.DataFrame:
    """Load and validate the existing Agentic Web subset without modifying it."""
    path = project_root / "data/processed/agentic_subset.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Required read-only Stage 1 input is missing: {path}")

    articles = pd.read_parquet(path)
    required = {"article_id", "title", "date", "source", "text_clean", "is_agentic_web"}
    missing = required - set(articles.columns)
    if missing:
        raise ValueError(f"Agentic subset is missing required columns: {sorted(missing)}")

    articles = articles.copy()
    articles["article_id"] = articles["article_id"].astype(str)
    articles["title"] = articles["title"].fillna("").astype(str)
    articles["source"] = articles["source"].fillna("").astype(str)
    articles["text_clean"] = articles["text_clean"].fillna("").astype(str)
    if articles["article_id"].duplicated().any():
        raise ValueError("Agentic subset contains duplicate article IDs.")
    if not articles["is_agentic_web"].astype(bool).all():
        raise ValueError("Agentic subset contains rows not marked as Agentic Web.")
    if articles["text_clean"].str.strip().eq("").any():
        raise ValueError("Agentic subset contains empty cleaned article text.")

    articles["published_at"] = pd.to_datetime(articles["date"], errors="coerce", utc=True)
    return articles.sort_values(["published_at", "article_id"], kind="stable").reset_index(drop=True)


def _proportional_quotas(
    counts: pd.Series,
    sample_size: int,
    min_per_group: int = 0,
) -> dict[str, int]:
    counts = counts[counts.gt(0)].astype(int)
    if sample_size > int(counts.sum()):
        raise ValueError("Sample size exceeds available records across strata.")
    if sample_size < len(counts) * min_per_group:
        raise ValueError("Sample size is too small to meet the requested stratum minimum.")

    quotas = pd.Series(min_per_group, index=counts.index, dtype=int)
    capacity = counts - quotas
    remaining_count = sample_size - int(quotas.sum())
    if remaining_count > int(capacity.sum()):
        raise ValueError("Requested sample cannot be allocated within stratum capacities.")
    proportions = capacity / capacity.sum() * remaining_count if capacity.sum() else capacity
    additional = proportions.apply(math.floor).astype(int)
    quotas += additional
    remaining = sample_size - int(quotas.sum())
    remainders = (proportions - additional).sort_values(ascending=False, kind="stable")
    for label in remainders.index[:remaining]:
        quotas.loc[label] += 1
    return {str(key): int(value) for key, value in quotas.items()}


def sample_passages(
    articles: pd.DataFrame,
    sample_size: int = 100,
    seed: int = 20261020,
    passage_chars: int = 2400,
) -> pd.DataFrame:
    """Select a deterministic, time-stratified sample and retain source offsets."""
    if sample_size < 1 or passage_chars < 200:
        raise ValueError("sample_size must be positive and passage_chars at least 200.")
    usable = articles.loc[
        articles["text_clean"].str.strip().ne("") & articles["published_at"].notna()
    ].copy()
    if sample_size > len(usable):
        raise ValueError(f"Requested {sample_size} passages from only {len(usable)} usable articles.")

    usable["quarter"] = (
        usable["published_at"].dt.tz_convert(None).dt.to_period("Q").astype(str)
    )
    quarter_counts = usable["quarter"].value_counts().sort_index()
    quarter_quotas = _proportional_quotas(
        quarter_counts,
        sample_size,
        min_per_group=1 if sample_size >= len(quarter_counts) else 0,
    )
    samples: list[pd.DataFrame] = []
    random_state = seed
    for quarter, count in sorted(quarter_quotas.items()):
        quarter_rows = usable.loc[usable["quarter"].eq(quarter)]
        if "topic_id" in quarter_rows and quarter_rows["topic_id"].notna().any():
            topic_values = quarter_rows["topic_id"].fillna("unknown").astype(str)
            topic_quotas = _proportional_quotas(
                topic_values.value_counts().sort_index(), count
            )
            for topic, topic_count in sorted(topic_quotas.items()):
                if topic_count:
                    samples.append(
                        quarter_rows.loc[topic_values.eq(topic)].sample(
                            n=topic_count, random_state=random_state
                        )
                    )
                    random_state += 1
        elif count:
            samples.append(
                quarter_rows.sample(n=count, random_state=random_state)
            )
            random_state += 1
    passages = pd.concat(samples, ignore_index=True)
    passages["passage_text"] = passages["text_clean"].str.slice(0, passage_chars)
    passages["excerpt_start_char"] = 0
    passages["excerpt_end_char"] = passages["passage_text"].str.len()
    passages["passage_id"] = passages["article_id"].map(
        lambda article_id: "p_" + hashlib.sha256(
            f"agentic-web-v1:{article_id}".encode("utf-8")
        ).hexdigest()[:16]
    )
    output_columns = [
        "passage_id",
        "article_id",
        "title",
        "published_at",
        "source",
        "quarter",
    ]
    if "topic_id" in passages.columns:
        output_columns.append("topic_id")
    output_columns.extend(["excerpt_start_char", "excerpt_end_char", "passage_text"])
    return passages[output_columns].sort_values(
        ["quarter", "article_id"], kind="stable"
    ).reset_index(drop=True)


def passage_manifest(passages: pd.DataFrame) -> pd.DataFrame:
    """Return passage metadata without copying article text into an export."""
    return passages.drop(columns=["passage_text"]).copy()


def load_cooccurrence_graph(project_root: Path) -> nx.Graph:
    """Load only the existing explicitly labeled co-occurrence graph, read-only."""
    path = project_root / "outputs/graph/agentic_web.graphml"
    if not path.is_file():
        raise FileNotFoundError(f"Required read-only Stage 1 graph is missing: {path}")

    graph = nx.read_graphml(path)
    if graph.is_directed():
        raise ValueError("Expected the undirected Stage 1 co-occurrence graph.")
    for _, _, attrs in graph.edges(data=True):
        if attrs.get("relation_kind") != "cooccurrence":
            raise ValueError("Graph contains an edge not labeled as co-occurrence.")
        if attrs.get("predicate") != "MENTIONED_WITH":
            raise ValueError("Co-occurrence graph contains an unexpected predicate.")
    return graph


def _article_ids_from_edge(attrs: dict[str, Any]) -> list[str]:
    value = attrs.get("article_ids", "[]")
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        raise ValueError("GraphML article_ids edge attribute must decode to a list.")
    return [str(article_id) for article_id in value]


def _select_context(text: str, title: str, query: str, max_chars: int = 1400) -> str:
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]
    if not sentences:
        excerpt = text[:max_chars]
    else:
        terms = set(re.findall(r"[a-z0-9]{3,}", query.casefold()))
        ranked = sorted(
            enumerate(sentences),
            key=lambda item: (
                len(terms & set(re.findall(r"[a-z0-9]{3,}", item[1].casefold()))),
                -item[0],
            ),
            reverse=True,
        )
        selected: list[tuple[int, str]] = []
        length = 0
        for index, sentence in ranked:
            separator_length = 1 if selected else 0
            if length + len(sentence) + separator_length > max_chars:
                continue
            selected.append((index, sentence))
            length += len(sentence) + separator_length
        if not selected:
            excerpt = text[:max_chars]
        else:
            excerpt = " ".join(sentence for _, sentence in sorted(selected))

    return f"Title: {title}\nEvidence excerpt: {excerpt}"


class TfidfGraphRetriever:
    """Lexical article retrieval with optional bounded co-mention expansion."""

    def __init__(
        self,
        articles: pd.DataFrame,
        graph: nx.Graph,
        max_features: int = 200_000,
    ) -> None:
        self.articles = articles.reset_index(drop=True).copy()
        self.graph = graph
        self.article_ids = self.articles["article_id"].astype(str).tolist()
        self.article_to_index = {
            article_id: index for index, article_id in enumerate(self.article_ids)
        }
        documents = (
            self.articles["title"].fillna("").astype(str)
            + " "
            + self.articles["text_clean"].fillna("").astype(str)
        )
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.98,
            max_features=max_features,
            stop_words="english",
            dtype=np.float32,
        )
        self.matrix = self.vectorizer.fit_transform(documents)
        self.entity_labels = [
            (node, _normalized_text(str(attrs.get("label", ""))))
            for node, attrs in graph.nodes(data=True)
            if len(_normalized_text(str(attrs.get("label", "")))) >= 3
        ]

    def _graph_article_support(
        self,
        query: str,
        max_entities: int,
        max_neighbors: int,
    ) -> dict[str, int]:
        normalized_query = f" {_normalized_text(query)} "
        matched = [
            (node, label)
            for node, label in self.entity_labels
            if f" {label} " in normalized_query
        ]
        matched.sort(key=lambda item: (-len(item[1]), item[1]))
        chosen_nodes: list[str] = []
        seen_labels: set[str] = set()
        for node, label in matched:
            if label not in seen_labels:
                chosen_nodes.append(node)
                seen_labels.add(label)
            if len(chosen_nodes) >= max_entities:
                break

        support: dict[str, int] = {}
        for node in chosen_nodes:
            neighbors = sorted(
                self.graph[node].items(),
                key=lambda item: (
                    int(item[1].get("support_count", 0)),
                    str(item[0]),
                ),
                reverse=True,
            )[:max_neighbors]
            for _, attrs in neighbors:
                count = int(attrs.get("support_count", 0))
                for article_id in _article_ids_from_edge(attrs):
                    if article_id in self.article_to_index:
                        support[article_id] = max(support.get(article_id, 0), count)
        return support

    def retrieve(
        self,
        query: str,
        k: int = 5,
        graph_enabled: bool = False,
        graph_weight: float = 0.05,
        max_entities: int = 3,
        max_neighbors: int = 5,
    ) -> list[RetrievedArticle]:
        if not query.strip():
            raise ValueError("Retrieval query must not be empty.")
        if k < 1:
            raise ValueError("k must be positive.")
        query_vector = self.vectorizer.transform([query])
        if query_vector.nnz == 0:
            raise ValueError("Query has no terms in the fitted TF-IDF vocabulary.")
        text_scores = np.asarray((self.matrix @ query_vector.T).toarray()).ravel()
        candidate_count = min(max(k * 10, 50), len(self.article_ids))
        lexical_candidates = np.argsort(-text_scores, kind="stable")[:candidate_count]
        support = (
            self._graph_article_support(query, max_entities, max_neighbors)
            if graph_enabled
            else {}
        )
        candidate_indices = set(int(index) for index in lexical_candidates)
        candidate_indices.update(
            self.article_to_index[article_id] for article_id in support
        )

        max_support = max(support.values(), default=0)
        ranked: list[tuple[float, float, int, float]] = []
        for index in candidate_indices:
            article_id = self.article_ids[index]
            graph_support = support.get(article_id, 0)
            normalized_support = (
                math.log1p(graph_support) / math.log1p(max_support)
                if max_support > 0 and graph_support > 0
                else 0.0
            )
            score = float(text_scores[index]) + graph_weight * normalized_support
            ranked.append((score, float(text_scores[index]), index, float(normalized_support)))
        ranked.sort(key=lambda item: (-item[0], -item[1], self.article_ids[item[2]]))

        results: list[RetrievedArticle] = []
        for retrieval_score, text_score, index, normalized_support in ranked[:k]:
            row = self.articles.iloc[index]
            results.append(
                RetrievedArticle(
                    article_id=self.article_ids[index],
                    title=str(row["title"]),
                    published_at=(
                        row["published_at"].isoformat()
                        if pd.notna(row["published_at"])
                        else ""
                    ),
                    source=str(row["source"]),
                    text_score=text_score,
                    retrieval_score=retrieval_score,
                    graph_support=normalized_support,
                    context=_select_context(
                        str(row["text_clean"]), str(row["title"]), query
                    ),
                )
            )
        return results


def load_openai_client(project_root: Path) -> OpenAI:
    """Load an existing API key without printing or writing it to disk."""
    load_dotenv(project_root / ".env")
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LANGEXTRACT_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OpenAI API key is not configured. Set OPENAI_API_KEY in the environment "
            "or the existing git-ignored .env file."
        )
    return OpenAI(api_key=api_key)


def _cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    if model == AUTHORING_MODEL:
        prices = QUESTION_MODEL_PRICES
    elif model == ANSWER_MODEL:
        prices = ANSWER_MODEL_PRICES
    else:
        raise ValueError(f"No approved cost schedule is configured for model {model!r}.")
    return (
        prompt_tokens * prices["input"] + completion_tokens * prices["output"]
    ) / 1_000_000


def _usage_spend(usage_path: Path) -> float:
    if not usage_path.is_file():
        return 0.0
    usage = pd.read_csv(usage_path)
    if "cost_usd" not in usage:
        raise ValueError(f"Usage ledger is missing cost_usd: {usage_path}")
    return float(pd.to_numeric(usage["cost_usd"], errors="raise").sum())


def _record_usage(
    usage_path: Path,
    task: str,
    model: str,
    response: Any,
    budget_usd: float,
    max_completion_tokens: int,
    prompt_chars: int,
) -> None:
    usage = response.usage
    if usage is None or usage.prompt_tokens is None or usage.completion_tokens is None:
        raise RuntimeError("OpenAI response did not include billable token usage.")
    prompt_tokens = int(usage.prompt_tokens)
    completion_tokens = int(usage.completion_tokens)
    cost = _cost_usd(model, prompt_tokens, completion_tokens)
    existing = pd.read_csv(usage_path) if usage_path.is_file() else pd.DataFrame()
    current_spend = float(existing["cost_usd"].sum()) if not existing.empty else 0.0
    row = pd.DataFrame(
        [
            {
                "task": task,
                "model": model,
                "response_model": getattr(response, "model", ""),
                "response_id": getattr(response, "id", ""),
                "system_fingerprint": getattr(response, "system_fingerprint", ""),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "prompt_chars": prompt_chars,
                "max_completion_tokens": max_completion_tokens,
                "cost_usd": cost,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "pricing_source": OPENAI_PRICING_SOURCE,
                "pricing_checked": OPENAI_PRICING_CHECKED,
            }
        ]
    )
    _atomic_csv_write(pd.concat([existing, row], ignore_index=True), usage_path)
    if current_spend + cost > budget_usd:
        raise RuntimeError(
            f"API budget cap exceeded after this response: "
            f"${current_spend + cost:.4f} > ${budget_usd:.2f}. "
            "The actual usage was recorded in the ledger."
        )


def _chat_completion(
    client: OpenAI,
    model: str,
    system_message: str,
    user_message: str,
    task: str,
    usage_path: Path,
    budget_usd: float = API_BUDGET_USD,
    max_completion_tokens: int = 500,
) -> str:
    prompt = f"{system_message}\n{user_message}"
    estimated_prompt_tokens = math.ceil(len(prompt) / 2) + 128
    estimated_cost = _cost_usd(model, estimated_prompt_tokens, max_completion_tokens)
    if _usage_spend(usage_path) + estimated_cost > budget_usd:
        raise RuntimeError(
            f"Estimated call cost would exceed the ${budget_usd:.2f} API budget cap."
        )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        temperature=0,
        max_tokens=max_completion_tokens,
        response_format={"type": "json_object"},
    )
    _record_usage(
        usage_path,
        task,
        model,
        response,
        budget_usd,
        max_completion_tokens,
        len(prompt),
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError(f"OpenAI returned an empty response for task {task!r}.")
    return content


def generate_question_candidates(
    passages: pd.DataFrame,
    client: OpenAI,
    question_path: Path,
    usage_path: Path,
    budget_usd: float = API_BUDGET_USD,
    questions_per_passage: int = QUESTIONS_PER_PASSAGE,
) -> pd.DataFrame:
    """Generate explicitly unreviewed question candidates and checkpoint each passage."""
    if questions_per_passage != QUESTIONS_PER_PASSAGE:
        raise ValueError("The current Stage 3 protocol freezes two questions per passage.")
    existing = pd.read_csv(question_path) if question_path.is_file() else pd.DataFrame()
    required_existing = {"passage_id", "question_id"}
    if not existing.empty and not required_existing.issubset(existing.columns):
        raise ValueError(f"Existing question file lacks columns: {sorted(required_existing - set(existing.columns))}")
    completed = set(existing["passage_id"].astype(str)) if not existing.empty else set()
    category_pairs = [
        (QUESTION_TYPES[index % len(QUESTION_TYPES)], QUESTION_TYPES[(index + 2) % len(QUESTION_TYPES)])
        for index in range(len(passages))
    ]

    for passage_index, passage in passages.reset_index(drop=True).iterrows():
        passage_id = str(passage["passage_id"])
        if passage_id in completed:
            continue
        first_type, second_type = category_pairs[passage_index]
        system_message = (
            "Draft source-grounded evaluation questions. Use only the supplied passage. "
            "Do not infer facts not stated in it. Return JSON with an 'items' array."
        )
        user_message = json.dumps(
            {
                "title": str(passage["title"]),
                "published_at": (
                    passage["published_at"].isoformat()
                    if pd.notna(passage["published_at"])
                    else None
                ),
                "passage": str(passage["passage_text"]),
                "requested_types": [first_type, second_type],
                "requirements": {
                    "number_of_items": QUESTIONS_PER_PASSAGE,
                    "each_item_fields": [
                        "question_type",
                        "question",
                        "reference_answer",
                        "evidence_quote",
                    ],
                    "question_type_options": list(QUESTION_TYPES),
                    "evidence_quote": "A short exact quotation copied from the passage.",
                    "unsupported_type": "Use a supported type instead; never invent a fact.",
                },
            },
            ensure_ascii=False,
        )
        content = _chat_completion(
            client,
            AUTHORING_MODEL,
            system_message,
            user_message,
            task="question_authoring",
            usage_path=usage_path,
            budget_usd=budget_usd,
            max_completion_tokens=700,
        )
        payload = json.loads(content)
        items = payload.get("items")
        if not isinstance(items, list) or len(items) != QUESTIONS_PER_PASSAGE:
            raise ValueError(
                f"Expected exactly {QUESTIONS_PER_PASSAGE} question items for {passage_id}."
            )

        rows: list[dict[str, Any]] = []
        passage_text = str(passage["passage_text"])
        for item_index, item in enumerate(items):
            question_type = str(item.get("question_type", ""))
            question = str(item.get("question", "")).strip()
            reference_answer = str(item.get("reference_answer", "")).strip()
            evidence_quote = str(item.get("evidence_quote", "")).strip()
            if question_type not in QUESTION_TYPES:
                raise ValueError(
                    f"Unsupported generated question type {question_type!r} for {passage_id}."
                )
            if not question or not reference_answer or not evidence_quote:
                raise ValueError(f"Generated question fields are incomplete for {passage_id}.")
            evidence_matches = (
                _normalized_whitespace(evidence_quote)
                in _normalized_whitespace(passage_text)
            )
            question_id = f"q_{passage_id}_{item_index + 1}"
            rows.append(
                {
                    "question_id": question_id,
                    "passage_id": passage_id,
                    "supporting_article_ids": json.dumps([str(passage["article_id"])]),
                    "question_type": question_type,
                    "question": question,
                    "reference_answer": reference_answer,
                    "evidence_quote": evidence_quote,
                    "evidence_quote_matches_passage": evidence_matches,
                    "authoring_model": AUTHORING_MODEL,
                    "review_status": "pending",
                    "reviewer_id": "",
                    "review_notes": "",
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
        existing = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
        _atomic_csv_write(existing, question_path)
        completed.add(passage_id)

    return existing


def freeze_question_splits(
    questions: pd.DataFrame,
    test_size: float = 0.30,
    seed: int = 20261020,
) -> pd.DataFrame:
    """Split by source passage so questions from one article cannot cross partitions."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be strictly between 0 and 1.")
    if questions["question_id"].duplicated().any():
        raise ValueError("Question IDs must be unique before splitting.")
    passages = np.array(sorted(questions["passage_id"].astype(str).unique()))
    if len(passages) < 5:
        raise ValueError("At least five source passages are needed for a grouped split.")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(passages)
    test_count = max(1, round(len(shuffled) * test_size))
    test_passages = set(shuffled[:test_count])
    result = questions.copy()
    result["split"] = result["passage_id"].astype(str).map(
        lambda passage_id: "test" if passage_id in test_passages else "development"
    )
    return result


def approved_questions(questions: pd.DataFrame, split: str | None = None) -> pd.DataFrame:
    """Return only explicitly human-approved records for confirmatory evaluation."""
    required = {
        "review_status",
        "reviewer_id",
        "review_notes",
        "question",
        "supporting_article_ids",
        "question_id",
    }
    missing = required - set(questions.columns)
    if missing:
        raise ValueError(f"Question file is missing review fields: {sorted(missing)}")
    allowed_statuses = {"approved", "rejected"}
    statuses = set(questions["review_status"].fillna("").astype(str).str.strip())
    if not statuses.issubset(allowed_statuses):
        raise RuntimeError(
            "Question review is incomplete. Mark every candidate approved or rejected "
            "and record a reviewer_id before confirmatory evaluation."
        )
    rejected_without_reason = questions.loc[
        questions["review_status"].eq("rejected")
        & questions["review_notes"].fillna("").astype(str).str.strip().eq("")
    ]
    if not rejected_without_reason.empty:
        raise RuntimeError("Rejected questions need a brief reviewer_notes explanation.")
    if questions["reviewer_id"].fillna("").astype(str).str.strip().eq("").any():
        raise RuntimeError("Every reviewed question needs a reviewer_id.")
    approved = questions.loc[questions["review_status"].eq("approved")].copy()
    if split is not None:
        if "split" not in approved:
            raise ValueError("Question file must be frozen into development/test splits first.")
        approved = approved.loc[approved["split"].eq(split)]
    if approved.empty:
        raise ValueError(
            "No human-approved questions are available. Review the candidates and set "
            "review_status to 'approved' before confirmatory evaluation."
        )
    return approved.reset_index(drop=True)


def generate_rag_answer(
    question: str,
    retrieved: Iterable[RetrievedArticle],
    client: OpenAI,
    usage_path: Path,
    budget_usd: float = API_BUDGET_USD,
) -> dict[str, Any]:
    """Generate a cited answer grounded only in retrieved article excerpts."""
    articles = list(retrieved)
    if not articles:
        raise ValueError("At least one retrieved article is required for answer generation.")
    contexts = [
        {
            "article_id": item.article_id,
            "title": item.title,
            "published_at": item.published_at or None,
            "source": item.source,
            "evidence_excerpt": item.context,
        }
        for item in articles
    ]
    system_message = (
        "Answer the question using only the supplied retrieved passages. If they do not "
        "support an answer, say 'Insufficient evidence in the retrieved passages.' "
        "Cite supporting article IDs in a JSON array. Do not treat co-mentions as "
        "factual relations. Use publication dates only when supplied."
    )
    user_message = json.dumps(
        {
            "question": question,
            "retrieved_passages": contexts,
            "output_schema": {"answer": "string", "citations": ["article_id"]},
        },
        ensure_ascii=False,
    )
    content = _chat_completion(
        client,
        ANSWER_MODEL,
        system_message,
        user_message,
        task="rag_answer",
        usage_path=usage_path,
        budget_usd=budget_usd,
        max_completion_tokens=400,
    )
    payload = json.loads(content)
    answer = payload.get("answer")
    citations = payload.get("citations")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("RAG answer response must contain a non-empty answer string.")
    if not isinstance(citations, list) or any(not isinstance(value, str) for value in citations):
        raise ValueError("RAG answer citations must be a list of article ID strings.")
    retrieved_ids = {item.article_id for item in articles}
    invalid = set(citations) - retrieved_ids
    if invalid:
        raise ValueError(f"RAG answer cited articles that were not retrieved: {sorted(invalid)}")
    return {"answer": answer.strip(), "citations": citations}


def run_paired_retrieval(
    questions: pd.DataFrame,
    retriever: TfidfGraphRetriever,
    top_k: int = 5,
) -> pd.DataFrame:
    """Run both retrieval variants without invoking an answer-generation model."""
    rows: list[dict[str, Any]] = []
    for question in questions.itertuples(index=False):
        for variant, graph_enabled in (("text_only", False), ("graph_expanded", True)):
            retrieved = retriever.retrieve(
                str(question.question), k=top_k, graph_enabled=graph_enabled
            )
            rows.append(
                {
                    "question_id": str(question.question_id),
                    "variant": variant,
                    "retrieved_article_ids": json.dumps(
                        [item.article_id for item in retrieved]
                    ),
                    "retrieval_scores": json.dumps(
                        [item.retrieval_score for item in retrieved]
                    ),
                    "graph_supported_in_top_k": sum(
                        item.graph_support > 0 for item in retrieved
                    ),
                    "answer": "",
                    "citations": "[]",
                    "answer_faithfulness_review": "",
                    "answer_relevance_review": "",
                    "temporal_grounding_review": "",
                    "reviewer_notes": "",
                }
            )
    return pd.DataFrame(rows)


def run_paired_answers(
    questions: pd.DataFrame,
    retriever: TfidfGraphRetriever,
    client: OpenAI,
    results_path: Path,
    usage_path: Path,
    budget_usd: float = API_BUDGET_USD,
    top_k: int = 5,
) -> pd.DataFrame:
    """Run the same answer model on approved questions for both retrieval variants."""
    approved = approved_questions(questions, split="test")
    existing = pd.read_csv(results_path) if results_path.is_file() else pd.DataFrame()
    required_existing = {"question_id", "variant", "question_hash"}
    if not existing.empty and not required_existing.issubset(existing.columns):
        raise ValueError("Existing Stage 3 results file has an incompatible schema.")
    if not existing.empty:
        existing["question_id"] = existing["question_id"].astype(str)
        existing["variant"] = existing["variant"].astype(str)
        existing["question_hash"] = existing["question_hash"].astype(str)

    for question in approved.itertuples(index=False):
        question_id = str(question.question_id)
        question_text = str(question.question)
        question_hash = hashlib.sha256(
            f"{question_id}:{question_text}".encode("utf-8")
        ).hexdigest()
        for variant, graph_enabled in (("text_only", False), ("graph_expanded", True)):
            cached = (
                not existing.empty
                and existing["question_id"].eq(question_id)
                & existing["variant"].eq(variant)
                & existing["question_hash"].eq(question_hash)
            )
            if not existing.empty and cached.any():
                continue

            retrieved = retriever.retrieve(
                question_text,
                k=top_k,
                graph_enabled=graph_enabled,
            )
            generated = generate_rag_answer(
                question_text,
                retrieved,
                client,
                usage_path,
                budget_usd=budget_usd,
            )
            row = {
                "question_id": question_id,
                "question_hash": question_hash,
                "split": str(question.split),
                "variant": variant,
                "answer_model": ANSWER_MODEL,
                "answer": generated["answer"],
                "citations": json.dumps(generated["citations"]),
                "retrieved_article_ids": json.dumps(
                    [item.article_id for item in retrieved]
                ),
                "retrieval_scores": json.dumps(
                    [item.retrieval_score for item in retrieved]
                ),
                "graph_supported_in_top_k": sum(
                    item.graph_support > 0 for item in retrieved
                ),
                "answer_faithfulness_review": "",
                "answer_relevance_review": "",
                "temporal_grounding_review": "",
                "reviewer_id": "",
                "reviewer_notes": "",
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            if not existing.empty:
                existing = existing.loc[
                    ~(
                        existing["question_id"].eq(question_id)
                        & existing["variant"].eq(variant)
                    )
                ]
            existing = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
            _atomic_csv_write(existing, results_path)
    return existing


def retrieval_metrics(
    questions: pd.DataFrame,
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Compute Recall@k and reciprocal rank from reviewed supporting article IDs."""
    reviewed = approved_questions(questions)
    merged = results.merge(
        reviewed[["question_id", "supporting_article_ids"]],
        on="question_id",
        how="inner",
        validate="many_to_one",
    )
    if merged.empty:
        raise ValueError("No reviewed questions match the supplied retrieval results.")
    rows: list[dict[str, Any]] = []
    for variant, group in merged.groupby("variant", sort=True):
        recalls: list[float] = []
        reciprocal_ranks: list[float] = []
        for row in group.itertuples(index=False):
            relevant = set(json.loads(row.supporting_article_ids))
            retrieved_ids = json.loads(row.retrieved_article_ids)
            hit_rank = next(
                (
                    rank
                    for rank, article_id in enumerate(retrieved_ids, start=1)
                    if article_id in relevant
                ),
                None,
            )
            recalls.append(float(any(article_id in relevant for article_id in retrieved_ids)))
            reciprocal_ranks.append(1.0 / hit_rank if hit_rank is not None else 0.0)
        rows.append(
            {
                "variant": variant,
                "n_questions": len(group),
                "recall_at_k": float(np.mean(recalls)),
                "mrr_at_k": float(np.mean(reciprocal_ranks)),
                "k": max(len(json.loads(value)) for value in group["retrieved_article_ids"]),
            }
        )
    return pd.DataFrame(rows)


def answer_quality_metrics(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate only completed human answer reviews; never infer review labels."""
    if results.empty:
        raise ValueError("No answer results are available for human review aggregation.")
    normalized_results = results.copy()
    review_columns = {
        "answer_faithfulness_review": {"supported", "partially_supported", "unsupported"},
        "answer_relevance_review": {"relevant", "partially_relevant", "irrelevant"},
        "temporal_grounding_review": {
            "supported",
            "partially_supported",
            "unsupported",
            "not_applicable",
        },
    }
    required = {"variant", "reviewer_id", *review_columns}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Answer results are missing review fields: {sorted(missing)}")
    for column, allowed in review_columns.items():
        normalized_results[column] = (
            normalized_results[column].fillna("").astype(str).str.strip()
        )
        values = set(normalized_results[column])
        if not values.issubset(allowed):
            raise RuntimeError(
                f"Human review is incomplete or invalid in {column}; "
                f"allowed values are {sorted(allowed)}."
            )
    normalized_results["reviewer_id"] = (
        normalized_results["reviewer_id"].fillna("").astype(str).str.strip()
    )
    if normalized_results["reviewer_id"].eq("").any():
        raise RuntimeError("Every reviewed answer needs a reviewer_id.")

    rows: list[dict[str, Any]] = []
    for variant, group in normalized_results.groupby("variant", sort=True):
        temporal = group.loc[group["temporal_grounding_review"].ne("not_applicable")]
        rows.append(
            {
                "variant": variant,
                "n_answers": len(group),
                "faithfulness_supported_rate": float(
                    group["answer_faithfulness_review"].eq("supported").mean()
                ),
                "relevance_rate": float(
                    group["answer_relevance_review"].eq("relevant").mean()
                ),
                "n_temporal_applicable": len(temporal),
                "temporal_grounding_supported_rate": (
                    float(temporal["temporal_grounding_review"].eq("supported").mean())
                    if not temporal.empty
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def answer_reviews_complete(results: pd.DataFrame) -> bool:
    """Return whether every answer has a valid, attributable human review."""
    review_columns = {
        "answer_faithfulness_review": {"supported", "partially_supported", "unsupported"},
        "answer_relevance_review": {"relevant", "partially_relevant", "irrelevant"},
        "temporal_grounding_review": {
            "supported",
            "partially_supported",
            "unsupported",
            "not_applicable",
        },
    }
    required = {"reviewer_id", *review_columns}
    if results.empty or not required.issubset(results.columns):
        return False
    for column, allowed in review_columns.items():
        values = results[column].fillna("").astype(str).str.strip()
        if values.eq("").any() or not set(values).issubset(allowed):
            return False
    return not results["reviewer_id"].fillna("").astype(str).str.strip().eq("").any()


def cooccurrence_reviews_complete(reviews: pd.DataFrame) -> bool:
    """Return whether every sampled undirected co-occurrence edge was reviewed."""
    review_columns = {
        "entity_a_review": {"correct", "incorrect", "uncertain"},
        "entity_b_review": {"correct", "incorrect", "uncertain"},
        "source_cooccurrence_review": {
            "both_mentioned",
            "one_or_both_missing",
            "unclear",
        },
    }
    required = {"reviewer_id", "review_notes", *review_columns}
    if reviews.empty or not required.issubset(reviews.columns):
        return False

    normalized = reviews.copy()
    for column, allowed in review_columns.items():
        normalized[column] = normalized[column].fillna("").astype(str).str.strip()
        values = set(normalized[column])
        if not values.issubset(allowed):
            return False
    normalized["reviewer_id"] = (
        normalized["reviewer_id"].fillna("").astype(str).str.strip()
    )
    normalized["review_notes"] = (
        normalized["review_notes"].fillna("").astype(str).str.strip()
    )
    if normalized["reviewer_id"].eq("").any():
        return False

    unsupported = (
        normalized["entity_a_review"].ne("correct")
        | normalized["entity_b_review"].ne("correct")
        | normalized["source_cooccurrence_review"].ne("both_mentioned")
    )
    return not (
        unsupported & normalized["review_notes"].eq("")
    ).any()
