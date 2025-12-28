from typing import Iterable, List

try:
    import sacrebleu
except ImportError:  # pragma: no cover - optional dependency
    sacrebleu = None

try:
    from nltk.translate.bleu_score import corpus_bleu  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    corpus_bleu = None


def compute_bleu(references: Iterable[List[str]], hypotheses: Iterable[List[str]]) -> float:
    # Try sacrebleu first (stable MT metric); fall back to nltk if installed.
    refs = [" ".join(tokens) for tokens in references]
    hyps = [" ".join(tokens) for tokens in hypotheses]
    if sacrebleu is not None:
        bleu = sacrebleu.corpus_bleu(hyps, [refs])
        return float(bleu.score)
    if corpus_bleu is not None and refs:
        ref_tokens = [[r.split()] for r in refs]
        hyp_tokens = [h.split() for h in hyps]
        return float(corpus_bleu(ref_tokens, hyp_tokens) * 100)
    raise RuntimeError("sacrebleu or nltk must be installed for BLEU computation")
