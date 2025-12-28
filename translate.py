import argparse
import sys
from pathlib import Path
from typing import List, Optional

import torch

from data.dataset import default_zh_tokenize, ParallelTextDataset
from data.builders import build_vocab_from_datasets, build_target_vocab_from_datasets
from data.vocab import Vocabulary
from models.rnn import EncoderRNN, DecoderRNN, Seq2SeqRNN
from models.transformer import Seq2SeqTransformer
from utils.decoding import indices_to_sentence


def _build_rnn_from_config(cfg: dict, src_vocab: Vocabulary, tgt_vocab: Vocabulary) -> Seq2SeqRNN:
    encoder = EncoderRNN(
        vocab_size=len(src_vocab),
        embed_dim=int(cfg.get("embed_dim", 256)),
        hidden_size=int(cfg.get("hidden_size", 512)),
        num_layers=int(cfg.get("num_layers", 2)),
        rnn_type=str(cfg.get("rnn_type", "gru")),
        dropout=float(cfg.get("dropout", 0.2)),
        pad_idx=src_vocab.pad_idx,
    )
    decoder = DecoderRNN(
        vocab_size=len(tgt_vocab),
        embed_dim=int(cfg.get("embed_dim", 256)),
        hidden_size=int(cfg.get("hidden_size", 512)),
        num_layers=int(cfg.get("num_layers", 2)),
        rnn_type=str(cfg.get("rnn_type", "gru")),
        dropout=float(cfg.get("dropout", 0.2)),
        pad_idx=tgt_vocab.pad_idx,
        alignment=str(cfg.get("alignment", "dot")),
    )
    return Seq2SeqRNN(
        encoder,
        decoder,
        pad_idx=tgt_vocab.pad_idx,
        sos_idx=tgt_vocab.sos_idx,
        eos_idx=tgt_vocab.eos_idx,
    )


def _build_transformer_from_config(cfg: dict, src_vocab: Vocabulary, tgt_vocab: Vocabulary) -> Seq2SeqTransformer:
    return Seq2SeqTransformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=int(cfg.get("d_model", 512)),
        nhead=int(cfg.get("nhead", 8)),
        num_encoder_layers=int(cfg.get("num_encoder_layers", 6)),
        num_decoder_layers=int(cfg.get("num_decoder_layers", 6)),
        dim_feedforward=int(cfg.get("dim_feedforward", 2048)),
        dropout=float(cfg.get("dropout", 0.1)),
        pad_idx=tgt_vocab.pad_idx,
        sos_idx=tgt_vocab.sos_idx,
        eos_idx=tgt_vocab.eos_idx,
        pos_encoding=str(cfg.get("pos_encoding", "sinusoidal")),
        norm_type=str(cfg.get("norm_type", "layernorm")),
    )


def _encode_src(src_text: str, src_vocab: Vocabulary, max_src_len: Optional[int]) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = default_zh_tokenize(src_text)
    if max_src_len is not None:
        tokens = tokens[:max_src_len]
    indices = src_vocab.to_indices(tokens)
    if not indices:
        raise ValueError("Empty source after tokenization")
    src = torch.tensor(indices, dtype=torch.long).unsqueeze(0)
    lengths = torch.tensor([src.size(1)], dtype=torch.long)
    return src, lengths


def _translate_one(
    model,
    arch: str,
    src: torch.Tensor,
    src_lengths: torch.Tensor,
    decode: str,
    beam_size: int,
    max_steps: int,
) -> torch.Tensor:
    arch = arch.lower()
    if decode == "beam":
        return model.beam_search_decode(src, src_lengths, max_steps=max_steps, beam_size=beam_size)
    if decode == "greedy":
        if arch == "transformer":
            return model.greedy_decode(src, src_lengths, max_steps=max_steps)
        # RNN greedy returns shape [B, T]
        return model.greedy_decode(src, src_lengths, max_steps=max_steps)
    raise ValueError(f"Unsupported decode: {decode}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Translate a single sentence using a saved NMT checkpoint")
    p.add_argument("--arch", choices=["rnn", "transformer"], required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--train", type=Path, default=None, help="Training jsonl used to rebuild vocabs (defaults to ckpt config train)")
    p.add_argument("--src_vocab", type=Path, default=None, help="Optional: path to saved source vocab json")
    p.add_argument("--tgt_vocab", type=Path, default=None, help="Optional: path to saved target vocab json")
    p.add_argument("--min_freq", type=int, default=None, help="Defaults to ckpt config min_freq")
    p.add_argument("--max_vocab", type=int, default=None, help="Defaults to ckpt config max_vocab")
    p.add_argument("--max_src_len", type=int, default=None, help="Optional truncation (defaults to ckpt config max_src_len)")
    p.add_argument("--text", type=str, default=None, help="Source sentence to translate")
    p.add_argument("--decode", choices=["greedy", "beam"], default="greedy")
    p.add_argument("--beam_size", type=int, default=4)
    p.add_argument("--max_decode_steps", type=int, default=100)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    state = torch.load(args.checkpoint, map_location=device)
    cfg = state.get("config", {})

    train_path = args.train or (Path(cfg["train"]) if "train" in cfg else None)

    # Prefer loading persisted vocabs (consistent ids, fewer <unk> surprises), else rebuild from training set.
    src_vocab_path = args.src_vocab
    tgt_vocab_path = args.tgt_vocab
    if src_vocab_path is None and cfg.get("src_vocab"):
        src_vocab_path = Path(cfg["src_vocab"])
    if tgt_vocab_path is None and cfg.get("tgt_vocab"):
        tgt_vocab_path = Path(cfg["tgt_vocab"])

    if src_vocab_path is not None and tgt_vocab_path is not None and src_vocab_path.exists() and tgt_vocab_path.exists():
        src_vocab = Vocabulary.load(src_vocab_path)
        tgt_vocab = Vocabulary.load(tgt_vocab_path)
        max_src_len = args.max_src_len if args.max_src_len is not None else cfg.get("max_src_len", None)
    else:
        if train_path is None:
            raise SystemExit("Need --train (or checkpoint must contain config.train) to rebuild vocabs")
        min_freq = args.min_freq if args.min_freq is not None else int(cfg.get("min_freq", 2))
        max_vocab = args.max_vocab if args.max_vocab is not None else cfg.get("max_vocab", None)
        max_src_len = args.max_src_len if args.max_src_len is not None else cfg.get("max_src_len", None)
        train_dataset = ParallelTextDataset(
            train_path,
            max_src_len=cfg.get("max_src_len", None),
            max_tgt_len=cfg.get("max_tgt_len", None),
        )
        src_vocab = build_vocab_from_datasets([train_dataset], min_freq=min_freq, max_size=max_vocab)
        tgt_vocab = build_target_vocab_from_datasets([train_dataset], min_freq=min_freq, max_size=max_vocab)

    if args.arch == "rnn":
        model = _build_rnn_from_config(cfg, src_vocab, tgt_vocab)
    else:
        model = _build_transformer_from_config(cfg, src_vocab, tgt_vocab)

    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()

    src_text = args.text
    if src_text is None:
        # Simple stdin mode: one line -> one translation.
        src_text = sys.stdin.read().strip()
    if not src_text:
        raise SystemExit("Provide --text or pipe a sentence on stdin")

    src, src_lengths = _encode_src(src_text, src_vocab, max_src_len)
    src = src.to(device)
    src_lengths = src_lengths.to(device)

    with torch.no_grad():
        out = _translate_one(
            model,
            arch=args.arch,
            src=src,
            src_lengths=src_lengths,
            decode=args.decode,
            beam_size=args.beam_size,
            max_steps=args.max_decode_steps,
        )

    tokens: List[str] = indices_to_sentence(out[0].cpu().tolist(), tgt_vocab)
    print(" ".join(tokens))


if __name__ == "__main__":
    main()
