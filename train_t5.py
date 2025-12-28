import argparse
import json
import logging
import random
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW, get_linear_schedule_with_warmup

from utils.metrics import compute_bleu

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class T5Dataset(Dataset):
    def __init__(self, path: Path, tokenizer: T5Tokenizer, max_src_len: int = 128, max_tgt_len: int = 128):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                src = record.get("zh", record.get("src", "")).strip()
                tgt = record.get("en", record.get("tgt", "")).strip()
                if src and tgt:
                    self.samples.append((src, tgt))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        src, tgt = self.samples[idx]
        # T5 specific prefix for translation (though for fine-tuning on specific task it might not be strictly necessary if trained from scratch, but good practice)
        # For standard T5 pretraining tasks, prefix is used. Here we are fine-tuning.
        # We can add a prefix like "translate Chinese to English: "
        src_text = "translate Chinese to English: " + src
        
        model_inputs = self.tokenizer(
            src_text,
            max_length=self.max_src_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize targets (legacy API kept for compatibility with current transformers version).
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                tgt,
                max_length=self.max_tgt_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

        return {
            "input_ids": model_inputs.input_ids.squeeze(),
            "attention_mask": model_inputs.attention_mask.squeeze(),
            "labels": labels.input_ids.squeeze(),
        }

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, scaler=None):
    model.train()
    total_loss = 0
    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # Replace padding token id in labels with -100 so it's ignored by loss
        labels[labels == 0] = -100 

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        if step % 100 == 0:
            logger.info(f"Epoch {epoch} Step {step} Loss {loss.item():.4f}")
            
    return total_loss / len(dataloader)

def evaluate(model, dataloader, tokenizer, device, max_gen_len: int = 128, num_beams: int = 4):
    model.eval()
    predictions = []
    references = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Generate
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_gen_len,
                num_beams=num_beams,
                early_stopping=True
            )
            
            preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            # For references, we need to decode the labels (which are padded with -100 or 0)
            # But we can just use the raw text if we had it, but here we have tensors.
            # Let's decode labels.
            labels = batch["labels"].cpu().numpy()
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            refs = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            predictions.extend([p.split() for p in preds])
            references.extend([r.split() for r in refs])
            
    return compute_bleu(references, predictions)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--valid", type=Path, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--save_dir", type=Path, default=Path("checkpoints/t5"))
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_src_len", type=int, default=128)
    parser.add_argument("--max_tgt_len", type=int, default=128)
    parser.add_argument("--max_gen_len", type=int, default=128)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision (AMP) on CUDA")
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Loading T5 from {args.model_path}")
    tokenizer = T5Tokenizer.from_pretrained(args.model_path)
    model = T5ForConditionalGeneration.from_pretrained(args.model_path)
    model.to(device)
    
    train_dataset = T5Dataset(args.train, tokenizer, max_src_len=args.max_src_len, max_tgt_len=args.max_tgt_len)
    valid_dataset = T5Dataset(args.valid, tokenizer, max_src_len=args.max_src_len, max_tgt_len=args.max_tgt_len)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    args.save_dir.mkdir(parents=True, exist_ok=True)
    best_bleu = 0.0
    
    scaler = torch.cuda.amp.GradScaler() if args.amp and str(device).startswith("cuda") else None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch, scaler=scaler)
        logger.info(f"Epoch {epoch} Train Loss: {train_loss:.4f}")

        bleu = evaluate(model, valid_loader, tokenizer, device, max_gen_len=args.max_gen_len, num_beams=args.num_beams)
        logger.info(f"Epoch {epoch} Valid BLEU: {bleu:.2f}")
        
        if bleu > best_bleu:
            best_bleu = bleu
            model.save_pretrained(args.save_dir)
            tokenizer.save_pretrained(args.save_dir)
            logger.info(f"Saved best model with BLEU {best_bleu:.2f}")

if __name__ == "__main__":
    main()
