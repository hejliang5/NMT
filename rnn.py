from typing import Optional, Tuple, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

AlignmentType = str
HiddenState = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


class EncoderRNN(nn.Module):
    # Unidirectional multi-layer GRU/LSTM encoder over token embeddings.
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        num_layers: int,
        rnn_type: str = "gru",
        dropout: float = 0.1,
        pad_idx: int = 0,
        embeddings: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type.lower()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        if embeddings is not None:
            self.embed.weight.data.copy_(embeddings)
        rnn_cls = nn.GRU if self.rnn_type == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            embed_dim,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Pack variable-length batches for efficient RNN processing.
        embedded = self.dropout(self.embed(src))
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        outputs, hidden = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        return outputs, hidden


class Attention(nn.Module):
    # Supports dot, general, and additive alignment scoring for decoder attention ablations.
    def __init__(self, hidden_size: int, alignment: AlignmentType = "dot") -> None:
        super().__init__()
        alignment = alignment.lower()
        self.alignment = alignment
        if alignment not in {"dot", "general", "additive"}:
            raise ValueError(f"Unsupported alignment: {alignment}")
        if alignment == "general":
            self.linear_in = nn.Linear(hidden_size, hidden_size, bias=False)
        elif alignment == "additive":
            self.linear_query = nn.Linear(hidden_size, hidden_size, bias=False)
            self.linear_keys = nn.Linear(hidden_size, hidden_size, bias=False)
            self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, query: torch.Tensor, keys: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute alignment scores then mask padding positions before softmax.
        if self.alignment == "dot":
            scores = torch.bmm(keys, query.unsqueeze(2)).squeeze(2)
        elif self.alignment == "general":
            transformed = self.linear_in(query)
            scores = torch.bmm(keys, transformed.unsqueeze(2)).squeeze(2)
        else:
            q = self.linear_query(query).unsqueeze(1)
            k = self.linear_keys(keys)
            scores = self.v(torch.tanh(q + k)).squeeze(-1)
        scores.masked_fill_(~mask, float("-inf"))
        attn_weights = torch.softmax(scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), keys).squeeze(1)
        return context, attn_weights


class DecoderRNN(nn.Module):
    # Decoder consumes previous token + attention context to predict next token.
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        num_layers: int,
        rnn_type: str = "gru",
        dropout: float = 0.1,
        pad_idx: int = 0,
        alignment: AlignmentType = "dot",
        embeddings: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type.lower()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        if embeddings is not None:
            self.embed.weight.data.copy_(embeddings)
        rnn_cls = nn.GRU if self.rnn_type == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            embed_dim + hidden_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.attention = Attention(hidden_size, alignment=alignment)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_tokens: torch.Tensor,
        hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        encoder_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedded = self.dropout(self.embed(input_tokens))
        query = hidden[-1] if isinstance(hidden, torch.Tensor) else hidden[0][-1]
        # Attend over encoder outputs to build context for the current decoder step.
        context, attn_weights = self.attention(query, encoder_outputs, encoder_mask)
        rnn_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)
        output, hidden = self.rnn(rnn_input, hidden)
        output = output.squeeze(1)
        concat_output = torch.tanh(self.concat(torch.cat([output, context], dim=1)))
        logits = self.out(self.dropout(concat_output))
        return logits, hidden, attn_weights


class Seq2SeqRNN(nn.Module):
    # End-to-end encoder-decoder wrapper exposing training and decoding utilities.
    def __init__(
        self,
        encoder: EncoderRNN,
        decoder: DecoderRNN,
        pad_idx: int,
        sos_idx: int,
        eos_idx: int,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx

    def forward(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
        tgt: Optional[torch.Tensor] = None,
        max_steps: Optional[int] = None,
        teacher_forcing_ratio: float = 1.0,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        encoder_outputs, hidden = self.encoder(src, src_lengths)
        max_len = encoder_outputs.size(1)
        mask = torch.arange(max_len, device=src.device).unsqueeze(0) < src_lengths.unsqueeze(1)
        batch_size = src.size(0)
        if tgt is not None:
            steps = tgt.size(1) - 1
        else:
            steps = max_steps or 100
        logits = []
        attns: List[torch.Tensor] = []
        input_tokens = src.new_full((batch_size,), self.sos_idx)
        if tgt is not None:
            input_tokens = tgt[:, 0]
        for t in range(steps):
            decoder_input = input_tokens.unsqueeze(1)
            step_logits, hidden, attn = self.decoder(decoder_input, hidden, encoder_outputs, mask)
            logits.append(step_logits.unsqueeze(1))
            attns.append(attn)
            # Probabilistic teacher forcing: use gold token with given ratio, else use model prediction.
            use_teacher_forcing = tgt is not None and torch.rand(1).item() < teacher_forcing_ratio
            if use_teacher_forcing:
                input_tokens = tgt[:, t + 1] if t + 1 < tgt.size(1) else src.new_full((batch_size,), self.eos_idx)
            else:
                input_tokens = step_logits.argmax(dim=1)
        return torch.cat(logits, dim=1), attns

    def greedy_decode(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
        max_steps: int = 100,
    ) -> torch.Tensor:
        encoder_outputs, hidden = self.encoder(src, src_lengths)
        max_len = encoder_outputs.size(1)
        mask = torch.arange(max_len, device=src.device).unsqueeze(0) < src_lengths.unsqueeze(1)
        batch_size = src.size(0)
        input_tokens = src.new_full((batch_size,), self.sos_idx)
        outputs = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=src.device)
        for _ in range(max_steps):
            decoder_input = input_tokens.unsqueeze(1)
            step_logits, hidden, _ = self.decoder(decoder_input, hidden, encoder_outputs, mask)
            input_tokens = step_logits.argmax(dim=1)
            outputs.append(input_tokens.unsqueeze(1))
            finished |= input_tokens.eq(self.eos_idx)
            if finished.all():
                break
        return torch.cat(outputs, dim=1)

    def beam_search_decode(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
        max_steps: int,
        beam_size: int = 4,
    ) -> torch.Tensor:
        encoder_outputs, hidden = self.encoder(src, src_lengths)
        max_len = encoder_outputs.size(1)
        mask = torch.arange(max_len, device=src.device).unsqueeze(0) < src_lengths.unsqueeze(1)
        sequences: List[List[int]] = []
        for i in range(src.size(0)):
            enc_out = encoder_outputs[i : i + 1]
            enc_mask = mask[i : i + 1]
            hidden_i = self._select_hidden(hidden, i)
            tokens = self._beam_search_single(enc_out, enc_mask, hidden_i, max_steps, beam_size)
            sequences.append(tokens)
        max_seq_len = max(len(seq) for seq in sequences)
        device = src.device
        outputs = torch.full((len(sequences), max_seq_len), self.pad_idx, device=device, dtype=torch.long)
        for idx, seq in enumerate(sequences):
            seq_tensor = torch.tensor(seq, device=device, dtype=torch.long)
            outputs[idx, : seq_tensor.size(0)] = seq_tensor
        return outputs

    def _beam_search_single(
        self,
        encoder_outputs: torch.Tensor,
        encoder_mask: torch.Tensor,
        hidden: HiddenState,
        max_steps: int,
        beam_size: int,
    ) -> List[int]:
        device = encoder_outputs.device
        beams: List[Tuple[float, List[int], HiddenState, bool]] = [
            (0.0, [self.sos_idx], self._clone_hidden(hidden), False)
        ]
        completed: List[Tuple[float, List[int]]] = []
        for _ in range(max_steps):
            candidates: List[Tuple[float, List[int], HiddenState, bool]] = []
            for score, tokens, state, finished in beams:
                if finished:
                    candidates.append((score, tokens, state, True))
                    continue
                input_token = torch.tensor([tokens[-1]], device=device).unsqueeze(0)
                step_logits, new_state, _ = self.decoder(input_token, state, encoder_outputs, encoder_mask)
                log_probs = F.log_softmax(step_logits, dim=1)
                topk = torch.topk(log_probs, beam_size, dim=1)
                for log_p, idx in zip(topk.values[0].tolist(), topk.indices[0].tolist()):
                    next_tokens = tokens + [idx]
                    done = idx == self.eos_idx
                    # Accumulate log-prob scores for ranking candidate hypotheses.
                    candidates.append((score + log_p, next_tokens, self._clone_hidden(new_state), done))
            beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_size]
            completed.extend([(score, tokens) for score, tokens, _, finished in beams if finished])
            beams = [beam for beam in beams if not beam[3]]
            if not beams:
                break
        if completed:
            best = max(completed, key=lambda x: x[0])
        else:
            best = max([(score, tokens) for score, tokens, _, _ in beams], key=lambda x: x[0])
        result = best[1][1:]
        if result and result[-1] == self.eos_idx:
            result = result[:-1]
        return result

    @staticmethod
    def _select_hidden(hidden: HiddenState, index: int) -> HiddenState:
        if isinstance(hidden, tuple):
            return (
                hidden[0][:, index : index + 1, :].contiguous(),
                hidden[1][:, index : index + 1, :].contiguous(),
            )
        return hidden[:, index : index + 1, :].contiguous()

    @staticmethod
    def _clone_hidden(hidden: HiddenState) -> HiddenState:
        if isinstance(hidden, tuple):
            return (hidden[0].clone(), hidden[1].clone())
        return hidden.clone()
