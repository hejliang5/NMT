from typing import List

from data.vocab import Vocabulary


def indices_to_sentence(indices: List[int], vocab: Vocabulary) -> List[str]:
    # Convert predicted token indices into tokens, stopping at EOS to drop padding tail.
    tokens = []
    for idx in indices:
        token = vocab.itos[idx] if 0 <= idx < len(vocab) else vocab.itos[vocab.unk_idx]
        if idx == vocab.eos_idx:
            break
        tokens.append(token)
    return tokens
