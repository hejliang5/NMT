from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from .dataset import ParallelTextDataset
from .vocab import Vocabulary


def build_vocab_from_datasets(
    datasets: Sequence[ParallelTextDataset],
    min_freq: int = 2,
    max_size: Optional[int] = None,
    additional_tokens: Optional[Iterable[str]] = None,
) -> Vocabulary:
    # Build source-side vocab from aggregated token counts, honoring min_freq/max_size and any extra specials.
    counter = Counter()
    for dataset in datasets:
        for tokens, _ in dataset.samples:
            counter.update(tokens)
    vocab = Vocabulary(min_freq=min_freq, max_size=max_size)
    if additional_tokens:
        vocab.add_special_tokens(additional_tokens)
    vocab.add_tokens(counter.elements())
    return vocab


def build_target_vocab_from_datasets(
    datasets: Sequence[ParallelTextDataset],
    min_freq: int = 2,
    max_size: Optional[int] = None,
    additional_tokens: Optional[Iterable[str]] = None,
) -> Vocabulary:
    # Build target-side vocab separately to allow asymmetric source/target distributions.
    counter = Counter()
    for dataset in datasets:
        for _, tokens in dataset.samples:
            counter.update(tokens)
    vocab = Vocabulary(min_freq=min_freq, max_size=max_size)
    if additional_tokens:
        vocab.add_special_tokens(additional_tokens)
    vocab.add_tokens(counter.elements())
    return vocab
