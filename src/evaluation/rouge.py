"""ROUGE evaluation metrics for summarization."""

from rouge_score import rouge_scorer


def compute_rouge(sys_summary, gold_summary, rouge1_results, rouge2_results,
                  rougeL_results, rougeLsum_results):
    """Compute ROUGE-1, ROUGE-2, ROUGE-L, and ROUGE-Lsum scores.

    Args:
        sys_summary: System-generated summary (string or list).
        gold_summary: Reference summary (string or list).
        rouge1_results: Dict accumulating ROUGE-1 scores.
        rouge2_results: Dict accumulating ROUGE-2 scores.
        rougeL_results: Dict accumulating ROUGE-L scores.
        rougeLsum_results: Dict accumulating ROUGE-Lsum scores.

    Returns:
        Tuple of updated (rouge1, rouge2, rougeL, rougeLsum) result dicts.
    """
    if isinstance(sys_summary, list):
        sys_summary = " ".join(sys_summary)
    if isinstance(gold_summary, list):
        gold_summary = " ".join(gold_summary)

    # ROUGE-1
    scorer_r1 = rouge_scorer.RougeScorer(["rouge1"])
    scores = scorer_r1.score(sys_summary, gold_summary)
    p, r, f = scores["rouge1"]
    rouge1_results["precision"].append(p)
    rouge1_results["recall"].append(r)
    rouge1_results["fmeasure"].append(f)

    # ROUGE-2
    scorer_r2 = rouge_scorer.RougeScorer(["rouge2"])
    scores = scorer_r2.score(sys_summary, gold_summary)
    p, r, f = scores["rouge2"]
    rouge2_results["precision"].append(p)
    rouge2_results["recall"].append(r)
    rouge2_results["fmeasure"].append(f)

    # ROUGE-L
    scorer_rl = rouge_scorer.RougeScorer(["rougeL"])
    scores = scorer_rl.score(sys_summary, gold_summary)
    p, r, f = scores["rougeL"]
    rougeL_results["precision"].append(p)
    rougeL_results["recall"].append(r)
    rougeL_results["fmeasure"].append(f)

    # ROUGE-Lsum
    scorer_rls = rouge_scorer.RougeScorer(["rougeLsum"])
    scores = scorer_rls.score(sys_summary, gold_summary)
    p, r, f = scores["rougeLsum"]
    rougeLsum_results["precision"].append(p)
    rougeLsum_results["recall"].append(r)
    rougeLsum_results["fmeasure"].append(f)

    return rouge1_results, rouge2_results, rougeL_results, rougeLsum_results


def init_rouge_results():  # Factory for metric accumulators
    """Initialize empty ROUGE result accumulators.

    Returns:
        Tuple of four dicts for ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum.
    """
    def _empty():
        return {"precision": [], "recall": [], "fmeasure": []}
    return _empty(), _empty(), _empty(), _empty()


def average_rouge(results: dict) -> float:
    """Compute average F-measure from accumulated results.

    Args:
        results: Dict with 'fmeasure' key containing list of scores.

    Returns:
        Average F-measure score.
    """
    if not results["fmeasure"]:
        return 0.0
    return sum(results["fmeasure"]) / len(results["fmeasure"])
