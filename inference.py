import argparse
import sys
from pathlib import Path
from typing import Optional

import torch

from data.vocab import Vocabulary
from data.dataset import default_zh_tokenize
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


def _resolve_vocab_paths(checkpoint: Path, src_vocab: Optional[Path], tgt_vocab: Optional[Path]) -> tuple[Path, Path]:
    if src_vocab is not None and tgt_vocab is not None:
        return src_vocab, tgt_vocab

    ckpt_dir = checkpoint.parent
    candidate_src = ckpt_dir / "src_vocab.json"
    candidate_tgt = ckpt_dir / "tgt_vocab.json"
    if candidate_src.exists() and candidate_tgt.exists():
        return candidate_src, candidate_tgt

    raise SystemExit(
        "Could not find vocab files. Provide --src_vocab and --tgt_vocab, or place src_vocab.json/tgt_vocab.json next to the checkpoint."
    )


def _encode_src(text: str, src_vocab: Vocabulary, max_src_len: Optional[int]) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = default_zh_tokenize(text)
    if max_src_len is not None:
        tokens = tokens[:max_src_len]
    ids = src_vocab.to_indices(tokens)
    if not ids:
        raise ValueError("Empty source after tokenization")
    src = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    lengths = torch.tensor([src.size(1)], dtype=torch.long)
    return src, lengths


def translate_one(
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
        return model.greedy_decode(src, src_lengths, max_steps=max_steps)
    raise ValueError(f"Unsupported decode: {decode}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="One-click inference script for grading: load a checkpoint and translate zh->en."
    )
    p.add_argument(
        "--arch",
        choices=["transformer", "rnn"],
        default="transformer",
        help="Model family to use.",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/transformer_ablation/base/best_model.pt"),
        help="Path to model checkpoint.",
    )
    p.add_argument("--src_vocab", type=Path, default=None, help="Path to source vocab json")
    p.add_argument("--tgt_vocab", type=Path, default=None, help="Path to target vocab json")
    p.add_argument("--decode", choices=["greedy", "beam"], default="beam")
    p.add_argument("--beam_size", type=int, default=4)
    p.add_argument("--max_decode_steps", type=int, default=100)
    p.add_argument("--max_src_len", type=int, default=None)
    p.add_argument("--device", type=str, default="cuda")

    group = p.add_mutually_exclusive_group(required=False)
    group.add_argument("--text", type=str, default=None, help="Single source sentence")
    group.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Text file containing one source sentence per line (UTF-8)",
    )
    p.add_argument("--output", type=Path, default=None, help="Optional output file (one line per input)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    state = torch.load(args.checkpoint, map_location=device)
    cfg = state.get("config", {})

    src_vocab_path, tgt_vocab_path = _resolve_vocab_paths(args.checkpoint, args.src_vocab, args.tgt_vocab)
    src_vocab = Vocabulary.load(src_vocab_path)
    tgt_vocab = Vocabulary.load(tgt_vocab_path)

    if args.arch == "rnn":
        model = _build_rnn_from_config(cfg, src_vocab, tgt_vocab)
    else:
        model = _build_transformer_from_config(cfg, src_vocab, tgt_vocab)

    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()

    if args.text is not None:
        inputs = [args.text]
    elif args.input is not None:
        inputs = [line.strip() for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        stdin_text = sys.stdin.read().strip()
        inputs = [stdin_text] if stdin_text else []

    if not inputs:
        raise SystemExit("Provide --text, --input, or pipe text via stdin")

    max_src_len = args.max_src_len if args.max_src_len is not None else cfg.get("max_src_len", None)

    outputs: list[str] = []
    with torch.no_grad():
        for text in inputs:
            src, src_lengths = _encode_src(text, src_vocab, max_src_len)
            src = src.to(device)
            src_lengths = src_lengths.to(device)
            out = translate_one(
                model,
                arch=args.arch,
                src=src,
                src_lengths=src_lengths,
                decode=args.decode,
                beam_size=args.beam_size,
                max_steps=args.max_decode_steps,
            )
            tokens = indices_to_sentence(out[0].cpu().tolist(), tgt_vocab)
            outputs.append(" ".join(tokens))

    out_text = "\n".join(outputs)
    if args.output is not None:
        args.output.write_text(out_text + "\n", encoding="utf-8")
    else:
        print(out_text)


if __name__ == "__main__":
    main()
