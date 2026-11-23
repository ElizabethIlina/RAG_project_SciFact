"""Проверки RRF и документных retrieval-метрик на ручных примерах."""

from rag_playground.retrieval.evaluation import evaluate_ranking
from rag_playground.retrieval.fusion import reciprocal_rank_fusion


def test_rrf_rewards_document_present_in_both_rankings():
    """Документ из двух списков должен обойти документы только из одного списка."""
    fused = reciprocal_rank_fusion(
        [["shared", "bm25-only"], ["dense-only", "shared"]],
        rrf_k=60,
        top_k=3,
    )
    assert fused[0][0] == "shared"
    assert {doc_id for doc_id, _score in fused} == {
        "shared",
        "bm25-only",
        "dense-only",
    }


def test_metrics_include_queries_missing_from_ranking():
    """Отсутствующий запрос должен давать нули и оставаться в macro average."""
    qrels = {
        "q1": {"relevant-1": 1},
        "q2": {"relevant-2": 1},
    }
    ranking = {"q1": ["relevant-1", "other"]}
    metrics = evaluate_ranking(ranking, qrels, cutoffs=(1, 2))

    assert metrics["queries"] == 2
    assert metrics["recall@1"] == 0.5
    assert metrics["precision@1"] == 0.5
    assert metrics["mrr@1"] == 0.5
    assert metrics["ndcg@1"] == 0.5


def test_metrics_respect_rank_and_multiple_relevant_documents():
    """Recall@2 должен учитывать два relevant docs, а MRR — позицию первого."""
    qrels = {"q": {"d1": 1, "d2": 1}}
    ranking = {"q": ["wrong", "d2", "d1"]}
    metrics = evaluate_ranking(ranking, qrels, cutoffs=(2, 3))

    assert metrics["recall@2"] == 0.5
    assert metrics["precision@2"] == 0.5
    assert metrics["mrr@2"] == 0.5
    assert metrics["recall@3"] == 1.0
