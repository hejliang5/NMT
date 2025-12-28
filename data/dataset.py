import json
import re
import unicodedata
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from .vocab import Vocabulary, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN

TokenizeFn = Callable[[str], List[str]]


_WHITESPACE_RE = re.compile(r"\s+")


def _clean_text(text: str) -> str:
    """Basic data cleaning:
    - Remove illegal/control characters.
    - Normalize whitespace.

    This intentionally stays conservative (does not remove punctuation), since punctuation can help translation.
    """
    if not text:
        return ""
    # Remove Unicode control/format characters (Cc/Cf), including zero-width spaces.
    chars: list[str] = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat in {"Cc", "Cf"}:
            continue
        chars.append(ch)
    text = "".join(chars)
    # Normalize whitespace (spaces, newlines, tabs) to single spaces.
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def default_zh_tokenize(text: str) -> List[str]:
    try:
        import jieba as _jieba

        return list(_jieba.cut(text.strip()))
    except Exception:
        return list(text.strip())


def default_en_tokenize(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    return text.split()


class ParallelTextDataset(Dataset):
    def __init__(
        self,
        path: Path,
        src_tokenize: TokenizeFn = default_zh_tokenize,
        tgt_tokenize: TokenizeFn = default_en_tokenize,
        max_src_len: Optional[int] = None,
        max_tgt_len: Optional[int] = None,
        clean: bool = True,
        drop_long: bool = False,
    ) -> None:
        self.samples: List[Tuple[List[str], List[str]]] = []
        # Load jsonl lines, tokenize, and keep only non-empty pairs (optional length-trim applied).
        for line in path.read_text(encoding="utf-8").strip().splitlines():
            record: Dict[str, str] = json.loads(line)
            src_text = record["zh"] if "zh" in record else record["src"]
            tgt_text = record["en"] if "en" in record else record["tgt"]
            if clean:
                src_text = _clean_text(src_text)
                tgt_text = _clean_text(tgt_text)

            src_tokens = src_tokenize(src_text)
            tgt_tokens = tgt_tokenize(tgt_text)

            # Length filtering/truncation for excessively long sentences.
            if drop_long:
                if max_src_len is not None and len(src_tokens) > max_src_len:
                    continue
                if max_tgt_len is not None and len(tgt_tokens) > max_tgt_len:
                    continue
            else:
                if max_src_len is not None:
                    src_tokens = src_tokens[:max_src_len]
                if max_tgt_len is not None:
                    tgt_tokens = tgt_tokens[:max_tgt_len]
            if not src_tokens or not tgt_tokens:
                continue
            self.samples.append((src_tokens, tgt_tokens))

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[List[str], List[str]]:
        return self.samples[idx]


class NMTBatchCollator:
    def __init__(self, src_vocab: Vocabulary, tgt_vocab: Vocabulary) -> None:
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __call__(
        self, batch: List[Tuple[List[str], List[str]]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        src_encoded = [self.src_vocab.to_indices(src) for src, _ in batch]
        tgt_encoded = [self.tgt_vocab.to_indices(tgt) for _, tgt in batch]
        # Encoder inputs keep raw tokens; decoder inputs are <s> + tgt + </s>; decoder targets are tgt + </s>.
        src_tensor, src_lengths = self._pad_sequences(src_encoded, self.src_vocab.pad_idx)
        tgt_tensor, tgt_lengths = self._pad_sequences(
            [[self.tgt_vocab.sos_idx, *seq, self.tgt_vocab.eos_idx] for seq in tgt_encoded], self.tgt_vocab.pad_idx
        )
        decoder_targets, _ = self._pad_sequences(
            [[*seq, self.tgt_vocab.eos_idx] for seq in tgt_encoded], self.tgt_vocab.pad_idx
        )
        return src_tensor, src_lengths, tgt_tensor, decoder_targets

    def _pad_sequences(self, sequences: List[List[int]], pad_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
        max_len = lengths.max().item()
        tensor = torch.full((len(sequences), max_len), pad_idx, dtype=torch.long)
        for i, seq in enumerate(sequences):
            tensor[i, : lengths[i]] = torch.tensor(seq, dtype=torch.long)
        return tensor, lengths
