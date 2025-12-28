import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    # Minimal RMSNorm implementation to swap with LayerNorm for ablations.
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple math
        norm = x.norm(2, dim=-1, keepdim=True)
        rms = norm * (1.0 / math.sqrt(x.size(-1)))
        return self.scale * x / (rms + self.eps)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, learned: bool = False) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.learned = learned
        if learned:
            self.pe = nn.Embedding(max_len, d_model)
        else:
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        if self.learned:
            positions = torch.arange(start_pos, start_pos + x.size(1), device=x.device)
            pos_embed = self.pe(positions).unsqueeze(0)
        else:
            pos_embed = self.pe[:, start_pos : start_pos + x.size(1)]
        x = x + pos_embed
        return self.dropout(x)


def _make_norm(norm_type: str, dim: int) -> nn.Module:
    norm_type = norm_type.lower()
    if norm_type == "layernorm":
        return nn.LayerNorm(dim)
    if norm_type == "rmsnorm":
        return RMSNorm(dim)
    raise ValueError(f"Unsupported norm type: {norm_type}")


class Seq2SeqTransformer(nn.Module):
    # Thin wrapper around PyTorch Transformer to expose NMT-friendly interfaces and ablation knobs.
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        pad_idx: int = 0,
        sos_idx: int = 1,
        eos_idx: int = 2,
        pos_encoding: str = "sinusoidal",
        norm_type: str = "layernorm",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        learned_pe = pos_encoding.lower() == "learned"
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout, learned=learned_pe)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            norm_first=True,
            batch_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            norm_first=True,
            batch_first=True,
        )
        encoder_norm = _make_norm(norm_type, d_model)
        decoder_norm = _make_norm(norm_type, d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers, norm=encoder_norm)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers, norm=decoder_norm)
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)

    def forward(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
        tgt_inputs: Optional[torch.Tensor] = None,
        max_steps: Optional[int] = None,
        teacher_forcing_ratio: float = 1.0,
    ) -> Tuple[torch.Tensor, Optional[None]]:
        del teacher_forcing_ratio  # Transformer uses full teacher forcing with provided inputs.
        if tgt_inputs is None:
            raise ValueError("tgt_inputs is required for Transformer training")
        # Shift-right: decoder consumes <s> + tokens, target loss is on next token.
        decoder_inputs = tgt_inputs[:, :-1]
        memory, src_key_padding_mask = self._encode(src)
        logits = self._decode(decoder_inputs, memory, src_key_padding_mask)
        return logits, None

    def greedy_decode(self, src: torch.Tensor, src_lengths: torch.Tensor, max_steps: int = 100) -> torch.Tensor:
        memory, src_key_padding_mask = self._encode(src)
        batch_size = src.size(0)
        generated = torch.full((batch_size, 1), self.sos_idx, device=src.device, dtype=torch.long)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=src.device)
        for _ in range(max_steps):
            logits = self._decode(generated, memory, src_key_padding_mask)
            next_token = logits[:, -1].argmax(dim=-1)
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
            finished |= next_token.eq(self.eos_idx)
            if finished.all():
                break
        return generated[:, 1:]

    def beam_search_decode(self, src: torch.Tensor, src_lengths: torch.Tensor, max_steps: int, beam_size: int = 4) -> torch.Tensor:
        memory, src_key_padding_mask = self._encode(src)
        results: List[List[int]] = []
        for i in range(src.size(0)):
            mem_i = memory[i : i + 1]
            mask_i = src_key_padding_mask[i : i + 1]
            seq = self._beam_search_single(mem_i, mask_i, max_steps, beam_size)
            results.append(seq)
        max_len = max(len(r) for r in results)
        output = torch.full((len(results), max_len), self.pad_idx, device=src.device, dtype=torch.long)
        for idx, seq in enumerate(results):
            output[idx, : len(seq)] = torch.tensor(seq, device=src.device, dtype=torch.long)
        return output

    def _encode(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # src_key_padding_mask prevents attention from looking at padded positions.
        src_key_padding_mask = src.eq(self.pad_idx)
        src_emb = self.pos_encoding(self.src_embed(src))
        memory = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        return memory, src_key_padding_mask

    def _decode(self, tgt: torch.Tensor, memory: torch.Tensor, src_key_padding_mask: torch.Tensor) -> torch.Tensor:
        tgt_key_padding_mask = tgt.eq(self.pad_idx)
        # Causal mask blocks attention to future positions inside decoder.
        tgt_mask = self._generate_subsequent_mask(tgt.size(1), tgt.device)
        tgt_emb = self.pos_encoding(self.tgt_embed(tgt))
        decoded = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.output_proj(decoded)

    def _beam_search_single(
        self,
        memory: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
        max_steps: int,
        beam_size: int,
    ) -> List[int]:
        device = memory.device
        beams: List[Tuple[float, List[int]]] = [(0.0, [self.sos_idx])]
        finished: List[Tuple[float, List[int]]] = []
        for _ in range(max_steps):
            candidates: List[Tuple[float, List[int]]] = []
            for score, seq in beams:
                if seq[-1] == self.eos_idx:
                    finished.append((score, seq))
                    candidates.append((score, seq))
                    continue
                tgt = torch.tensor(seq, device=device, dtype=torch.long).unsqueeze(0)
                logits = self._decode(tgt, memory, src_key_padding_mask)
                log_probs = F.log_softmax(logits[:, -1], dim=-1)
                topk = torch.topk(log_probs, beam_size, dim=-1)
                for log_p, idx in zip(topk.values[0].tolist(), topk.indices[0].tolist()):
                    # Add next token hypothesis and accumulate log-prob for ranking.
                    candidates.append((score + log_p, seq + [idx]))
            beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_size]
            if all(seq[-1] == self.eos_idx for _, seq in beams):
                break
        if finished:
            best = max(finished, key=lambda x: x[0])
        else:
            best = max(beams, key=lambda x: x[0])
        result = best[1][1:]
        if result and result[-1] == self.eos_idx:
            result = result[:-1]
        return result

    @staticmethod
    def _generate_subsequent_mask(size: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(size, size, device=device, dtype=torch.bool), diagonal=1)