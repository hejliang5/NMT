import argparse
import logging
import random
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import ParallelTextDataset, NMTBatchCollator
from data.builders import build_vocab_from_datasets, build_target_vocab_from_datasets
from data.vocab import Vocabulary
from models.transformer import Seq2SeqTransformer
from training.trainer import Seq2SeqTrainer, TrainerConfig
from utils.decoding import indices_to_sentence
from utils.metrics import compute_bleu

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Transformer-based NMT model")
    # Data / vocab paths
    parser.add_argument("--train", type=Path, required=True, help="Path to training jsonl file")
    parser.add_argument("--valid", type=Path, required=True, help="Path to validation jsonl file")
    parser.add_argument("--save_dir", type=Path, default=Path("checkpoints/transformer"), help="Directory to save artifacts")
    parser.add_argument("--src_vocab", type=Path, default=None, help="Path to save/load source vocab")
    parser.add_argument("--tgt_vocab", type=Path, default=None, help="Path to save/load target vocab")
    parser.add_argument("--min_freq", type=int, default=2, help="Minimum frequency for vocabulary")
    parser.add_argument("--max_vocab", type=int, default=None, help="Maximum vocabulary size")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_encoder_layers", type=int, default=6)
    parser.add_argument("--num_decoder_layers", type=int, default=6)
    parser.add_argument("--dim_feedforward", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pos_encoding", choices=["sinusoidal", "learned"], default="sinusoidal")
    parser.add_argument("--norm_type", choices=["layernorm", "rmsnorm"], default="layernorm")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_src_len", type=int, default=None)
    parser.add_argument("--max_tgt_len", type=int, default=None)
    parser.add_argument("--clean", action=argparse.BooleanOptionalAction, default=True, help="Apply basic text cleaning")
    parser.add_argument("--drop_long", action=argparse.BooleanOptionalAction, default=False, help="Drop pairs longer than max_*_len instead of truncating")
    parser.add_argument("--eval_decode", choices=["greedy", "beam"], default="greedy")
    parser.add_argument("--beam_size", type=int, default=4)
    parser.add_argument("--max_decode_steps", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision (AMP) on CUDA")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    # Ensure reproducibility for data order and weight init.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_or_build_vocab(
    path: Path | None,
    builder,
    datasets,
    min_freq: int,
    max_vocab: int | None,
    name: str,
) -> Vocabulary:
    # Load cached vocab if present; otherwise build from training data and save.
    if path and path.exists():
        logger.info("Loading %s vocab from %s", name, path)
        return Vocabulary.load(path)
    logger.info("Building %s vocab", name)
    vocab = builder(datasets, min_freq=min_freq, max_size=max_vocab)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        vocab.save(path)
    return vocab


def prepare_dataloaders(args: argparse.Namespace):
    # Build datasets, vocabularies, and dataloaders for training/eval.
    train_dataset = ParallelTextDataset(
        args.train,
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len,
        clean=args.clean,
        drop_long=args.drop_long,
    )
    valid_dataset = ParallelTextDataset(
        args.valid,
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len,
        clean=args.clean,
        drop_long=args.drop_long,
    )
    src_vocab = load_or_build_vocab(
        args.src_vocab,
        build_vocab_from_datasets,
        [train_dataset],
        args.min_freq,
        args.max_vocab,
        "source",
    )
    tgt_vocab = load_or_build_vocab(
        args.tgt_vocab,
        build_target_vocab_from_datasets,
        [train_dataset],
        args.min_freq,
        args.max_vocab,
        "target",
    )
    collator = NMTBatchCollator(src_vocab, tgt_vocab)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
    )
    return train_loader, valid_loader, src_vocab, tgt_vocab


def build_model(args: argparse.Namespace, src_vocab: Vocabulary, tgt_vocab: Vocabulary) -> Seq2SeqTransformer:
    # Configurable Transformer encoder-decoder with positional/norm choices for ablations.
    return Seq2SeqTransformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        pad_idx=tgt_vocab.pad_idx,
        sos_idx=tgt_vocab.sos_idx,
        eos_idx=tgt_vocab.eos_idx,
        pos_encoding=args.pos_encoding,
        norm_type=args.norm_type,
    )


def translate_samples(
    model: Seq2SeqTransformer,
    dataloader: DataLoader,
    tgt_vocab: Vocabulary,
    decode: str,
    max_steps: int,
    beam_size: int,
    device: torch.device,
) -> List[List[str]]:
    # Run greedy or beam search to produce detokenized predictions for BLEU.
    predictions: List[List[str]] = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            src, src_lengths, _, _ = batch
            src = src.to(device)
            src_lengths = src_lengths.to(device)
            if decode == "beam":
                outputs = model.beam_search_decode(src, src_lengths, max_steps=max_steps, beam_size=beam_size)
            else:
                outputs = model.greedy_decode(src, src_lengths, max_steps=max_steps)
            for seq in outputs.cpu().tolist():
                predictions.append(indices_to_sentence(seq, tgt_vocab))
    return predictions


def collect_references(dataloader: DataLoader, tgt_vocab: Vocabulary) -> List[List[str]]:
    # Convert gold target sequences back to tokens for BLEU reference construction.
    references: List[List[str]] = []
    for _, _, _, tgt_targets in dataloader:
        for seq in tgt_targets.tolist():
            references.append(indices_to_sentence(seq, tgt_vocab))
    return references


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # If vocab paths are not provided, persist them alongside the checkpoint.
    # This makes later evaluation/translation consistent (same token->id mapping).
    if args.src_vocab is None:
        args.src_vocab = args.save_dir / "src_vocab.json"
    if args.tgt_vocab is None:
        args.tgt_vocab = args.save_dir / "tgt_vocab.json"
    # Data + model setup
    train_loader, valid_loader, src_vocab, tgt_vocab = prepare_dataloaders(args)
    model = build_model(args, src_vocab, tgt_vocab)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_idx)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    trainer = Seq2SeqTrainer(
        model,
        optimizer,
        criterion,
        TrainerConfig(
            epochs=args.epochs,
            lr=args.lr,
            clip_norm=args.clip,
            teacher_forcing_ratio=1.0,
            device=str(device),
        ),
        scheduler=scheduler,
        scaler=(torch.cuda.amp.GradScaler() if args.amp and str(device).startswith("cuda") else None),
    )
    best_val = float("inf")
    args.save_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        train_stats = trainer.train_epoch(train_loader, epoch)
        val_stats = trainer.evaluate(valid_loader, teacher_forcing_ratio=1.0)
        logger.info(
            "Epoch %d: train_loss=%.4f train_ppl=%.2f val_loss=%.4f val_ppl=%.2f",
            epoch,
            train_stats["loss"],
            train_stats["ppl"],
            val_stats["loss"],
            val_stats["ppl"],
        )
        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            ckpt_path = args.save_dir / "best_model.pt"
            torch.save({"model": model.state_dict(), "config": vars(args)}, ckpt_path)
            logger.info("Saved checkpoint to %s", ckpt_path)
        if trainer.scheduler is not None:
            trainer.scheduler.step(val_stats["loss"])
    # Reload best checkpoint before final BLEU on validation set.
    best_ckpt = args.save_dir / "best_model.pt"
    if best_ckpt.exists():
        logger.info("Loading best checkpoint from %s", best_ckpt)
        state = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(state["model"])
        model.to(device)
    logger.info("Computing BLEU on validation set using %s decoding", args.eval_decode)
    predictions = translate_samples(model, valid_loader, tgt_vocab, args.eval_decode, args.max_decode_steps, args.beam_size, device)
    references = collect_references(valid_loader, tgt_vocab)
    bleu = compute_bleu(references, predictions)
    logger.info("Validation BLEU: %.2f", bleu)


if __name__ == "__main__":
    main()
