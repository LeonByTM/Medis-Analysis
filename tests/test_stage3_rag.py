import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import networkx as nx
import pandas as pd

from stage3_rag import (
    _article_ids_from_edge,
    _cost_usd,
    _select_context,
    _proportional_quotas,
    answer_reviews_complete,
    answer_quality_metrics,
    approved_questions,
    cooccurrence_review_metrics,
    cooccurrence_reviews_complete,
    freeze_question_splits,
    generate_rag_answer,
    paired_answer_quality_comparison,
    paired_retrieval_comparison,
    question_reviews_complete,
    retrieval_metrics,
    sample_passages,
    RetrievedArticle,
    TfidfGraphRetriever,
    _validate_paired_variants,
    validate_confirmatory_question_types,
    retrieval_latency_metrics,
    paired_bootstrap_intervals,
    run_paired_answers,
    run_paired_retrieval,
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

    def test_context_selection_preserves_abbreviated_month_dates(self) -> None:
        context = _select_context(
            "The integration was announced on Dec. 15, 2025. Another sentence follows.",
            "Integration update",
            "When was the integration announced?",
            published_at="2025-12-15T00:00:00+00:00",
        )
        self.assertIn("Dec. 15, 2025", context)
        self.assertIn("Article publication date: 2025-12-15", context)
        self.assertNotIn("announced on Dec. Another", context)

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
        diagnostics = run_paired_retrieval(
            pd.DataFrame(
                [{"question_id": "q1", "question": "Acme agent Protocol Zeta"}]
            ),
            retriever,
            top_k=1,
        )
        self.assertEqual(len(diagnostics), 2)
        self.assertTrue((diagnostics["retrieval_latency_ms"] >= 0).all())

    def test_retrieval_latency_reports_each_variant_and_question(self) -> None:
        class FakeRetriever:
            def retrieve(self, question: str, k: int, graph_enabled: bool) -> list:
                return []

        questions = pd.DataFrame(
            [
                {
                    "question_id": "q1",
                    "question": "What did Acme announce?",
                    "question_type": "factoid",
                }
            ]
        )
        metrics = retrieval_latency_metrics(
            questions, FakeRetriever(), top_k=3, repetitions=2
        )
        self.assertEqual(set(metrics["variant"]), {"text_only", "graph_expanded"})
        self.assertEqual(metrics["timed_runs"].tolist(), [2, 2])
        self.assertTrue((metrics["median_retrieval_latency_ms"] >= 0).all())

    def test_paired_bootstrap_intervals_are_reproducible_and_paired(self) -> None:
        paired = pd.DataFrame(
            {
                "question_id": ["q1", "q2", "q3", "q4"],
                "mrr_delta": [0.0, -0.5, 0.0, -0.25],
            }
        )
        first = paired_bootstrap_intervals(
            paired, {"mrr": "mrr_delta"}, sample_count=1000, seed=7
        )
        second = paired_bootstrap_intervals(
            paired, {"mrr": "mrr_delta"}, sample_count=1000, seed=7
        )
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first.loc[0, "n_pairs"], 4)
        self.assertLessEqual(first.loc[0, "ci_95_low"], first.loc[0, "mean_delta_graph_minus_text"])
        self.assertGreaterEqual(first.loc[0, "ci_95_high"], first.loc[0, "mean_delta_graph_minus_text"])
        with self.assertRaisesRegex(ValueError, "one paired row per question"):
            paired_bootstrap_intervals(
                pd.concat([paired, paired.iloc[[0]]], ignore_index=True),
                {"mrr": "mrr_delta"},
                sample_count=10,
            )

    def test_rag_answer_normalizes_numeric_article_id_citations(self) -> None:
        retrieved = [
            RetrievedArticle(
                article_id="153503",
                title="Oracle and NVIDIA",
                published_at="2025-03-19",
                source="test",
                text_score=1.0,
                retrieval_score=1.0,
                graph_support=0.0,
                context="Oracle and NVIDIA announced a partnership.",
            )
        ]
        response = json.dumps(
            {"answer": "They announced a partnership.", "citations": [153503]}
        )
        with patch("stage3_rag._chat_completion", return_value=response) as completion:
            result = generate_rag_answer(
                "What did they announce?",
                retrieved,
                client=object(),
                usage_path=Path("unused.csv"),
            )
        self.assertEqual(result["citations"], ["153503"])
        system_prompt = completion.call_args.args[2]
        self.assertIn("event date", system_prompt)
        self.assertIn("article publication date", system_prompt)
        self.assertIn("Answer concisely", system_prompt)
        request = json.loads(completion.call_args.args[3])
        self.assertEqual(request["retrieved_passages"][0]["retrieval_rank"], 1)
        self.assertEqual(
            request["retrieved_passages"][0]["published_at"],
            "2025-03-19",
        )

    def test_rag_answer_rejects_numeric_citations_not_in_retrieved_articles(self) -> None:
        retrieved = [
            RetrievedArticle(
                article_id="153503",
                title="Oracle and NVIDIA",
                published_at="2025-03-19",
                source="test",
                text_score=1.0,
                retrieval_score=1.0,
                graph_support=0.0,
                context="Oracle and NVIDIA announced a partnership.",
            )
        ]
        response = json.dumps(
            {"answer": "They announced a partnership.", "citations": [999999]}
        )
        with patch("stage3_rag._chat_completion", return_value=response):
            with self.assertRaisesRegex(ValueError, "cited articles that were not retrieved"):
                generate_rag_answer(
                    "What did they announce?",
                    retrieved,
                    client=object(),
                    usage_path=Path("unused.csv"),
                )

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

    def test_answer_rerun_preserves_human_reviewed_legacy_rows(self) -> None:
        questions = pd.DataFrame(
            [
                {
                    "question_id": "q1",
                    "question": "What did Acme announce?",
                    "split": "test",
                    "question_type": "factoid",
                    "reference_answer": "Acme announced an agent.",
                    "supporting_article_ids": '["a1"]',
                    "review_status": "approved",
                    "reviewer_id": "reviewer",
                    "review_notes": "",
                }
            ]
        )
        existing = pd.DataFrame(
            [
                {
                    "question_id": "q1",
                    "variant": "text_only",
                    "question_hash": "legacy-hash",
                    "answer_correctness_review": "correct",
                    "answer_faithfulness_review": "supported",
                    "answer_relevance_review": "relevant",
                    "temporal_grounding_review": "not_applicable",
                    "citation_support_review": "all_cited_claims_supported",
                    "abstention_review": "not_applicable",
                    "reviewer_id": "reviewer",
                }
            ]
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            results_path = Path(temporary_directory) / "results.csv"
            usage_path = Path(temporary_directory) / "usage.csv"
            existing.to_csv(results_path, index=False)
            with self.assertRaisesRegex(RuntimeError, "Refusing to overwrite human-reviewed"):
                run_paired_answers(
                    questions,
                    retriever=object(),
                    client=object(),
                    results_path=results_path,
                    usage_path=usage_path,
                )
            after = pd.read_csv(results_path, keep_default_na=False)
            self.assertEqual(after.loc[0, "answer_correctness_review"], "correct")

    def test_question_review_completion_requires_attribution_and_rejection_reason(self) -> None:
        questions = pd.DataFrame(
            [
                {
                    "review_status": "approved",
                    "reviewer_id": "reviewer",
                    "review_notes": "",
                },
                {
                    "review_status": "rejected",
                    "reviewer_id": "reviewer",
                    "review_notes": "Unsupported by source.",
                },
            ]
        )
        self.assertTrue(question_reviews_complete(questions))
        self.assertFalse(question_reviews_complete(questions.assign(reviewer_id="")))
        self.assertFalse(
            question_reviews_complete(
                questions.assign(review_notes=["", " "])
            )
        )
        self.assertFalse(
            question_reviews_complete(
                questions.assign(review_status=["approved", "pending"])
            )
        )

    def test_confirmatory_questions_validate_types_and_evidence(self) -> None:
        complete = pd.DataFrame(
            {
                "question_id": ["q1", "q2"],
                "question_type": ["factoid", "relational"],
                "reference_answer": ["A fact.", "A supported relation."],
                "supporting_article_ids": ['["a1"]', '["a2", "a3"]'],
            }
        )
        validate_confirmatory_question_types(complete)
        with self.assertRaisesRegex(RuntimeError, "unsupported question types"):
            validate_confirmatory_question_types(
                pd.DataFrame(
                    {
                        "question_id": ["q1"],
                        "question_type": ["unsupported_type"],
                        "reference_answer": ["A fact."],
                        "supporting_article_ids": ['["a1"]'],
                    }
                )
            )
        unanswerable = pd.DataFrame(
            {
                "question_id": ["q3"],
                "question_type": ["unanswerable"],
                "reference_answer": ["Insufficient evidence in the corpus."],
                "supporting_article_ids": ["[]"],
            }
        )
        validate_confirmatory_question_types(unanswerable)
        with self.assertRaisesRegex(ValueError, "no gold supporting"):
            validate_confirmatory_question_types(
                unanswerable.assign(supporting_article_ids=['["a3"]'])
            )

    def test_retrieval_metrics_exclude_unanswerable_questions(self) -> None:
        questions = pd.DataFrame(
            [
                {
                    "question_id": "fact",
                    "question": "What was announced?",
                    "reference_answer": "Acme announced its agent.",
                    "question_type": "factoid",
                    "supporting_article_ids": '["a1"]',
                    "review_status": "approved",
                    "reviewer_id": "reviewer",
                    "review_notes": "",
                },
                {
                    "question_id": "multi",
                    "question": "How did the events connect?",
                    "reference_answer": "The announcement linked the two events.",
                    "question_type": "multi_hop",
                    "supporting_article_ids": '["a2", "a3"]',
                    "review_status": "approved",
                    "reviewer_id": "reviewer",
                    "review_notes": "",
                },
                {
                    "question_id": "none",
                    "question": "Which unsupported claim is true?",
                    "reference_answer": "Insufficient evidence in the corpus.",
                    "question_type": "unanswerable",
                    "supporting_article_ids": "[]",
                    "review_status": "approved",
                    "reviewer_id": "reviewer",
                    "review_notes": "",
                },
            ]
        )
        results = pd.DataFrame(
            [
                {
                    "question_id": question_id,
                    "variant": variant,
                    "retrieved_article_ids": retrieved,
                }
                for variant in ["text_only", "graph_expanded"]
                for question_id, retrieved in [
                    ("fact", '["a1"]'),
                    ("multi", '["wrong", "a2"]'),
                    ("none", '["a3"]'),
                ]
            ]
        )
        summary = retrieval_metrics(questions, results)
        self.assertTrue(summary["n_questions"].eq(2).all())
        self.assertTrue(summary["recall_at_k"].eq(0.75).all())
        paired = paired_retrieval_comparison(questions, results)
        multi_hop = paired.set_index("question_id").loc["multi"]
        self.assertEqual(multi_hop["recall_at_k_text_only"], 0.5)
        self.assertEqual(multi_hop["recall_at_k_graph_expanded"], 0.5)
        self.assertEqual(multi_hop["recall_at_k_delta"], 0.0)

    def test_paired_retrieval_rejects_missing_variant_rows(self) -> None:
        results = pd.DataFrame(
            [
                {"question_id": "q1", "variant": "text_only"},
                {"question_id": "q1", "variant": "graph_expanded"},
                {"question_id": "q2", "variant": "text_only"},
            ]
        )
        with self.assertRaisesRegex(ValueError, "both retrieval variants"):
            _validate_paired_variants(results)

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
                    "question_id": "q1",
                    "variant": "text_only",
                    "question_type": "factoid",
                    "reviewer_id": "reviewer",
                    "answer_correctness_review": "correct",
                    "answer_faithfulness_review": "supported",
                    "answer_relevance_review": "relevant",
                    "temporal_grounding_review": "not_applicable",
                    "citation_support_review": "all_cited_claims_supported",
                    "abstention_review": "not_applicable",
                },
                {
                    "question_id": "q1",
                    "variant": "graph_expanded",
                    "question_type": "factoid",
                    "reviewer_id": "reviewer",
                    "answer_correctness_review": "partially_correct",
                    "answer_faithfulness_review": "partially_supported",
                    "answer_relevance_review": "relevant",
                    "temporal_grounding_review": "not_applicable",
                    "citation_support_review": "some_cited_claims_unsupported",
                    "abstention_review": "not_applicable",
                },
            ]
        )
        summary = answer_quality_metrics(results)
        self.assertTrue(answer_reviews_complete(results))
        summary_by_variant = summary.set_index("variant")
        self.assertEqual(
            summary_by_variant.loc["text_only", "faithfulness_supported_rate"], 1.0
        )
        self.assertEqual(
            summary_by_variant.loc[
                "text_only", "context_grounding_fully_supported_rate"
            ],
            1.0,
        )
        self.assertEqual(summary_by_variant.loc["text_only", "relevance_rate"], 1.0)
        self.assertEqual(summary_by_variant.loc["text_only", "correctness_rate"], 1.0)
        self.assertEqual(
            summary_by_variant.loc["graph_expanded", "mean_correctness_score"], 0.5
        )
        self.assertEqual(
            summary_by_variant.loc[
                "graph_expanded", "context_grounding_partially_supported_rate"
            ],
            1.0,
        )
        self.assertEqual(summary_by_variant.loc["text_only", "n_temporal_applicable"], 0)
        paired = paired_answer_quality_comparison(results).iloc[0]
        self.assertEqual(paired["correctness_delta"], -0.5)
        self.assertEqual(paired["citation_support_delta"], -0.5)
        self.assertEqual(paired["context_grounding_delta"], -0.5)
        whitespace_results = results.assign(
            answer_faithfulness_review=" supported ",
            answer_relevance_review=" relevant ",
            answer_correctness_review=" correct ",
            temporal_grounding_review=" not_applicable ",
            citation_support_review=" all_cited_claims_supported ",
            abstention_review=" not_applicable ",
        )
        whitespace_summary = answer_quality_metrics(whitespace_results)
        self.assertEqual(
            whitespace_summary.set_index("variant").loc[
                "text_only", "faithfulness_supported_rate"
            ],
            1.0,
        )
        self.assertEqual(
            whitespace_summary.set_index("variant").loc[
                "text_only", "all_citations_supported_rate"
            ],
            1.0,
        )
        missing_citations = results.assign(citation_support_review="no_citations")
        missing_citation_summary = answer_quality_metrics(missing_citations).set_index(
            "variant"
        )
        self.assertEqual(
            missing_citation_summary.loc[
                "text_only", "context_grounding_unsupported_rate"
            ],
            1.0,
        )
        self.assertEqual(
            paired_answer_quality_comparison(whitespace_results).iloc[0][
                "correctness_delta"
            ],
            0.0,
        )
        incomplete = results.copy()
        incomplete["answer_faithfulness_review"] = ""
        self.assertFalse(answer_reviews_complete(incomplete))
        self.assertFalse(answer_reviews_complete(pd.DataFrame()))

    def test_answer_quality_reports_abstention_only_for_unanswerable_questions(self) -> None:
        results = pd.DataFrame(
            [
                {
                    "question_id": "q-unanswerable",
                    "variant": "text_only",
                    "question_type": "unanswerable",
                    "reviewer_id": "reviewer",
                    "answer_correctness_review": "correct",
                    "answer_faithfulness_review": "supported",
                    "answer_relevance_review": "relevant",
                    "temporal_grounding_review": "not_applicable",
                    "citation_support_review": "no_citations",
                    "abstention_review": "correct_abstention",
                },
                {
                    "question_id": "q-unanswerable",
                    "variant": "graph_expanded",
                    "question_type": "unanswerable",
                    "reviewer_id": "reviewer",
                    "answer_correctness_review": "correct",
                    "answer_faithfulness_review": "supported",
                    "answer_relevance_review": "relevant",
                    "temporal_grounding_review": "not_applicable",
                    "citation_support_review": "no_citations",
                    "abstention_review": "correct_abstention",
                },
            ]
        )
        summary = answer_quality_metrics(results)
        self.assertTrue(answer_reviews_complete(results))
        self.assertEqual(summary.loc[0, "n_unanswerable"], 1)
        self.assertEqual(summary.loc[0, "correct_abstention_rate"], 1.0)
        self.assertEqual(summary.loc[0, "n_context_grounding_applicable"], 0)
        invalid = results.assign(abstention_review="not_applicable")
        self.assertFalse(answer_reviews_complete(invalid))
        with self.assertRaises(RuntimeError):
            answer_quality_metrics(invalid)

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

    def test_cooccurrence_review_metrics_report_determinate_denominators(self) -> None:
        reviews = pd.DataFrame(
            [
                {
                    "entity_a_review": "correct",
                    "entity_b_review": "correct",
                    "source_cooccurrence_review": "both_mentioned",
                    "reviewer_id": "reviewer",
                    "review_notes": "",
                },
                {
                    "entity_a_review": "uncertain",
                    "entity_b_review": "incorrect",
                    "source_cooccurrence_review": "unclear",
                    "reviewer_id": "reviewer",
                    "review_notes": "Entity A and source mention are unclear.",
                },
            ]
        )
        summary = cooccurrence_review_metrics(reviews).iloc[0]
        self.assertEqual(summary["n_edges_reviewed"], 2)
        self.assertEqual(summary["n_entity_a_determinate"], 1)
        self.assertEqual(summary["entity_a_link_correct_rate"], 1.0)
        self.assertEqual(summary["n_fully_supported_edges"], 1)
        self.assertEqual(summary["fully_supported_rate_all_reviewed"], 0.5)

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
