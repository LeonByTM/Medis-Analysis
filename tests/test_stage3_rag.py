import json
import unittest

import networkx as nx
import pandas as pd

from stage3_rag import (
    _article_ids_from_edge,
    _cost_usd,
    _proportional_quotas,
    answer_reviews_complete,
    answer_quality_metrics,
    approved_questions,
    cooccurrence_reviews_complete,
    freeze_question_splits,
    sample_passages,
    TfidfGraphRetriever,
)


class Stage3RagTests(unittest.TestCase):
    def test_proportional_quotas_preserve_total_and_minimum(self) -> None:
        counts = pd.Series({"2024Q3": 10, "2024Q4": 40, "2025Q1": 50})
        quotas = _proportional_quotas(counts, sample_size=12, min_per_group=1)
        self.assertEqual(sum(quotas.values()), 12)
        self.assertTrue(all(value >= 1 for value in quotas.values()))

    def test_passage_sample_is_deterministic_and_stratified(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "article_id": f"a-{index}",
                    "title": f"Article {index}",
                    "date": date,
                    "source": f"source-{index % 3}",
                    "text_clean": f"Agentic AI passage {index}. " * 40,
                    "is_agentic_web": True,
                    "topic_id": index % 2,
                }
                for index, date in enumerate(
                    [
                        "2024-10-01",
                        "2025-01-01",
                        "2025-04-01",
                        "2025-07-01",
                        "2025-10-01",
                        "2026-01-01",
                        "2026-04-01",
                        "2026-07-01",
                    ]
                    * 5
                )
            ]
        )
        frame["published_at"] = pd.to_datetime(frame["date"], utc=True)
        first = sample_passages(frame, sample_size=8, seed=7)
        second = sample_passages(frame, sample_size=8, seed=7)
        self.assertEqual(first["passage_id"].tolist(), second["passage_id"].tolist())
        self.assertEqual(first["article_id"].nunique(), 8)
        self.assertEqual(set(first["quarter"]), set(frame["published_at"].dt.tz_convert(None).dt.to_period("Q").astype(str)))

    def test_graph_retriever_uses_only_cooccurrence_article_links(self) -> None:
        articles = pd.DataFrame(
            [
                {
                    "article_id": "a1",
                    "title": "Acme agent",
                    "source": "test",
                    "published_at": pd.Timestamp("2025-01-01", tz="UTC"),
                    "text_clean": "Acme built an agent that uses Protocol Zeta.",
                },
                {
                    "article_id": "a2",
                    "title": "Unrelated",
                    "source": "test",
                    "published_at": pd.Timestamp("2025-01-02", tz="UTC"),
                    "text_clean": "The unrelated article also mentions an agent and weather.",
                },
                {
                    "article_id": "a3",
                    "title": "Weather",
                    "source": "test",
                    "published_at": pd.Timestamp("2025-01-03", tz="UTC"),
                    "text_clean": "A separate weather report with no relevant entity.",
                },
            ]
        )
        graph = nx.Graph()
        graph.add_node("company:acme", label="Acme", entity_type="ORGANIZATION")
        graph.add_node("protocol:zeta", label="Protocol Zeta", entity_type="PROTOCOL")
        graph.add_edge(
            "company:acme",
            "protocol:zeta",
            predicate="MENTIONED_WITH",
            relation_kind="cooccurrence",
            support_count=1,
            article_ids=json.dumps(["a1"]),
        )
        retriever = TfidfGraphRetriever(articles, graph, max_features=100)
        result = retriever.retrieve("Acme agent Protocol Zeta", k=1, graph_enabled=True)
        self.assertEqual(result[0].article_id, "a1")
        self.assertGreater(result[0].graph_support, 0)

    def test_pending_questions_cannot_enter_confirmatory_evaluation(self) -> None:
        questions = pd.DataFrame(
            [
                {
                    "question_id": "q1",
                    "supporting_article_ids": '["a1"]',
                    "question": "What was announced?",
                    "review_status": "pending",
                    "reviewer_id": "",
                    "review_notes": "",
                }
            ]
        )
        with self.assertRaises(RuntimeError):
            approved_questions(questions)

    def test_question_split_keeps_passage_groups_together(self) -> None:
        questions = pd.DataFrame(
            [
                {"question_id": f"q{passage}-{item}", "passage_id": f"p{passage}"}
                for passage in range(10)
                for item in range(2)
            ]
        )
        result = freeze_question_splits(questions, test_size=0.3, seed=5)
        self.assertEqual(result.groupby("passage_id")["split"].nunique().max(), 1)
        self.assertIn("test", set(result["split"]))
        self.assertIn("development", set(result["split"]))

    def test_answer_quality_requires_complete_human_ratings(self) -> None:
        results = pd.DataFrame(
            [
                {
                    "variant": "text_only",
                    "reviewer_id": "reviewer",
                    "answer_faithfulness_review": "supported",
                    "answer_relevance_review": "relevant",
                    "temporal_grounding_review": "not_applicable",
                }
            ]
        )
        summary = answer_quality_metrics(results)
        self.assertTrue(answer_reviews_complete(results))
        self.assertEqual(summary.loc[0, "faithfulness_supported_rate"], 1.0)
        self.assertEqual(summary.loc[0, "relevance_rate"], 1.0)
        self.assertEqual(summary.loc[0, "n_temporal_applicable"], 0)
        whitespace_results = results.assign(
            answer_faithfulness_review=" supported ",
            answer_relevance_review=" relevant ",
            temporal_grounding_review=" not_applicable ",
        )
        whitespace_summary = answer_quality_metrics(whitespace_results)
        self.assertEqual(
            whitespace_summary.loc[0, "faithfulness_supported_rate"], 1.0
        )
        incomplete = results.copy()
        incomplete["answer_faithfulness_review"] = ""
        self.assertFalse(answer_reviews_complete(incomplete))
        self.assertFalse(answer_reviews_complete(pd.DataFrame()))

    def test_cooccurrence_review_gate_requires_attributable_decisions(self) -> None:
        reviews = pd.DataFrame(
            [
                {
                    "entity_a_review": "correct",
                    "entity_b_review": "correct",
                    "source_cooccurrence_review": "both_mentioned",
                    "reviewer_id": "reviewer",
                    "review_notes": "",
                }
            ]
        )
        self.assertTrue(cooccurrence_reviews_complete(reviews))
        self.assertFalse(
            cooccurrence_reviews_complete(
                reviews.assign(entity_b_review="pending")
            )
        )
        self.assertFalse(
            cooccurrence_reviews_complete(
                reviews.assign(
                    entity_b_review="incorrect",
                    review_notes="",
                )
            )
        )

    def test_api_cost_estimates_match_configured_standard_rates(self) -> None:
        self.assertAlmostEqual(_cost_usd("gpt-4o-mini", 1_000_000, 0), 0.15)
        self.assertAlmostEqual(_cost_usd("gpt-4o", 0, 1_000_000), 10.0)

    def test_graph_article_ids_are_decoded(self) -> None:
        self.assertEqual(
            _article_ids_from_edge({"article_ids": '["a1", "a2"]'}),
            ["a1", "a2"],
        )


if __name__ == "__main__":
    unittest.main()
