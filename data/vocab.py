import json
import logging
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Dict, Optional

logger = logging.getLogger(__name__)

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]


class Vocabulary:
    # Lightweight vocab wrapper handling token-to-index mapping with frequency filtering and persistence.
    def __init__(self, tokens: Optional[Iterable[str]] = None, min_freq: int = 1, max_size: Optional[int] = None) -> None:
        self.min_freq = min_freq
        self.max_size = max_size
        self.itos: List[str] = []
        self.stoi: Dict[str, int] = {}
        self.freqs: Counter = Counter()
        for tok in SPECIAL_TOKENS:
            self._add_token(tok, special=True)
        if tokens is not None:
            self.add_tokens(tokens)

    def _add_token(self, token: str, special: bool = False) -> int:
        if token in self.stoi:
            return self.stoi[token]
        idx = len(self.itos)
        self.itos.append(token)
        self.stoi[token] = idx
        if not special:
            self.freqs[token] += 1
        return idx

    def add_tokens(self, tokens: Iterable[str]) -> None:
        tokens = list(tokens)
        counter = Counter(tokens)
        self.freqs.update(counter)
        items = counter.most_common()
        if self.max_size is not None:
            items = items[: max(0, self.max_size - len(self.itos))]
        for token, freq in items:
            if freq < self.min_freq:
                continue
            if token in self.stoi:
                continue
            self._add_token(token)

    def add_special_tokens(self, tokens: Iterable[str]) -> None:
        for token in tokens:
            self._add_token(token, special=True)

    def to_indices(self, tokens: Iterable[str]) -> List[int]:
        unk = self.stoi[UNK_TOKEN]
        return [self.stoi.get(tok, unk) for tok in tokens]

    def to_tokens(self, indices: Iterable[int]) -> List[str]:
        return [self.itos[idx] if 0 <= idx < len(self.itos) else UNK_TOKEN for idx in indices]

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.itos)

    def save(self, path: Path) -> None:
        data = {
            "itos": self.itos,
            "freqs": {tok: int(freq) for tok, freq in self.freqs.items()},
            "min_freq": self.min_freq,
            "max_size": self.max_size,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        logger.info("Saved vocabulary to %s", path)

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        data = json.loads(path.read_text(encoding="utf-8"))
        vocab = cls(min_freq=data.get("min_freq", 1), max_size=data.get("max_size"))
        vocab.itos = list(data["itos"])
        vocab.stoi = {tok: idx for idx, tok in enumerate(vocab.itos)}
        vocab.freqs = Counter(data.get("freqs", {}))
        return vocab

    @property
    def pad_idx(self) -> int:
        return self.stoi[PAD_TOKEN]

    @property
    def sos_idx(self) -> int:
        return self.stoi[SOS_TOKEN]

    @property
    def eos_idx(self) -> int:
        return self.stoi[EOS_TOKEN]

    @property
    def unk_idx(self) -> int:
        return self.stoi[UNK_TOKEN]
