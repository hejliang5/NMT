from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

@dataclass
class TrainerConfig:
    epochs: int = 10
    lr: float = 1e-3
    clip_norm: float = 1.0
    teacher_forcing_ratio: float = 0.5
    device: str = "cuda"
    log_every: int = 100


class Seq2SeqTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        config: TrainerConfig,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.scaler = scaler
        self.model.to(self.device)

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_tokens = 0
        for step, batch in enumerate(dataloader, 1):
            # Expected batch tuple: (src, src_lengths, decoder_inputs, decoder_targets)
            src, src_lengths, tgt_inputs, tgt_targets = [tensor.to(self.device) for tensor in batch]
            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                # Teacher forcing ratio handled inside model (RNN) or ignored (Transformer uses full input).
                outputs, _ = self.model(src, src_lengths, tgt_inputs, teacher_forcing_ratio=self.config.teacher_forcing_ratio)
                logits = outputs.view(-1, outputs.size(-1))
                loss = self.criterion(logits, tgt_targets.view(-1))
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_norm)
                self.optimizer.step()
            total_loss += loss.item() * (tgt_targets.size(0) * tgt_targets.size(1))
            total_tokens += tgt_targets.numel()
            if step % self.config.log_every == 0:
                avg_loss = total_loss / total_tokens
                ppl = torch.exp(torch.tensor(avg_loss)).item()
                print(f"Epoch {epoch} Step {step}: loss={avg_loss:.4f} ppl={ppl:.2f}")
        avg_loss = total_loss / max(total_tokens, 1)
        return {"loss": avg_loss, "ppl": float(torch.exp(torch.tensor(avg_loss)).item())}

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, teacher_forcing_ratio: float = 0.0) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        for batch in dataloader:
            src, src_lengths, tgt_inputs, tgt_targets = [tensor.to(self.device) for tensor in batch]
            # Default uses free-running decoding for RNN; Transformer still uses full inputs.
            outputs, _ = self.model(src, src_lengths, tgt_inputs, teacher_forcing_ratio=teacher_forcing_ratio)
            logits = outputs.view(-1, outputs.size(-1))
            loss = self.criterion(logits, tgt_targets.view(-1))
            total_loss += loss.item() * (tgt_targets.size(0) * tgt_targets.size(1))
            total_tokens += tgt_targets.numel()
        avg_loss = total_loss / max(total_tokens, 1)
        return {"loss": avg_loss, "ppl": float(torch.exp(torch.tensor(avg_loss)).item())}