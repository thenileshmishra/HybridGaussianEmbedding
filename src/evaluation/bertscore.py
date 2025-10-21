"""BERTScore evaluation metric for summarization."""

from bert_score import score


def compute_bert_score(sys_summary, gold_summary, bert_results):
    """Compute BERTScore between system and reference summaries.

    Args:
        sys_summary: System-generated summary (string or list).
        gold_summary: Reference summary (string or list).
        bert_results: Dict accumulating BERTScore values.

    Returns:
        Updated bert_results dict.
    """
    if isinstance(sys_summary, list):
        sys_summary = " ".join(sys_summary)
    if isinstance(gold_summary, list):
        gold_summary = " ".join(gold_summary)

    P, R, F1 = score([sys_summary], [gold_summary], lang="en")

    bert_results["precision"].append(P.mean().item())
    bert_results["recall"].append(R.mean().item())
    bert_results["fmeasure"].append(F1.mean().item())

    return bert_results


def init_bert_results():
    """Initialize empty BERTScore result accumulator.

    Returns:
        Dict with empty precision, recall, and fmeasure lists.
    """
    return {"precision": [], "recall": [], "fmeasure": []}
