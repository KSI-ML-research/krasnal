from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from tokenizer import PAD_ID, SPECIAL_TOKENS, THINK_END_ID, THINK_START_ID, Tokenizer


@dataclass
class Phase1RolloutBatch:
    token_ids: torch.Tensor
    completion_mask: torch.Tensor
    prompt_lengths: torch.Tensor
    think_lengths: torch.Tensor


class Phase1RolloutGenerator:
    """Generate phase-1 RLVR sequences with injected think delimiters."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Tokenizer,
        *,
        device: str | torch.device,
        temperature: float = 1.0,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.temperature = temperature
        self.move_token_ids = torch.tensor(
            [
                token_id
                for token_id in sorted(tokenizer.id_to_move)
                if token_id not in SPECIAL_TOKENS
            ],
            dtype=torch.long,
            device=self.device,
        )

    def _prepare_batch(self, contexts: list[list[int]]) -> torch.Tensor:
        rows = [
            torch.tensor(ctx[-self.model.config.block_size :], dtype=torch.long) for ctx in contexts
        ]
        max_len = max(row.numel() for row in rows)
        padded_rows = []
        for row in rows:
            pad_len = max_len - row.numel()
            if pad_len:
                row = F.pad(row, (pad_len, 0), value=PAD_ID)
            padded_rows.append(row)
        return torch.stack(padded_rows, dim=0).to(self.device)

    @torch.no_grad()
    def generate(
        self,
        prompts: torch.Tensor,
        prompt_lengths: torch.Tensor,
        *,
        group_size: int,
        think_min_tokens: int,
        think_max_tokens: int,
    ) -> Phase1RolloutBatch:
        if think_max_tokens < think_min_tokens:
            raise ValueError("think_max_tokens must be >= think_min_tokens")

        expanded_prompt_lengths = prompt_lengths.repeat_interleave(group_size)
        contexts: list[list[int]] = []
        for prompt, prompt_len in zip(prompts, prompt_lengths, strict=True):
            prefix = prompt[: int(prompt_len.item())].tolist()
            for _ in range(group_size):
                contexts.append(prefix + [THINK_START_ID])

        total_samples = len(contexts)
        think_lengths = torch.randint(
            low=think_min_tokens,
            high=think_max_tokens + 1,
            size=(total_samples,),
            device=self.device,
        )

        for think_step in range(int(think_lengths.max().item())):
            active = [
                idx for idx, target in enumerate(think_lengths.tolist()) if think_step < target
            ]
            if not active:
                break
            batch = self._prepare_batch([contexts[idx] for idx in active])
            logits, _ = self.model(batch)
            next_token = _sample_move_tokens(
                logits[:, -1, :],
                move_token_ids=self.move_token_ids,
                temperature=self.temperature,
            )
            for local_idx, sample_idx in enumerate(active):
                contexts[sample_idx].append(int(next_token[local_idx].item()))

        for context in contexts:
            context.append(THINK_END_ID)

        batch = self._prepare_batch(contexts)
        logits, _ = self.model(batch)
        played_moves = _sample_move_tokens(
            logits[:, -1, :],
            move_token_ids=self.move_token_ids,
            temperature=self.temperature,
        )
        for sample_idx, next_token in enumerate(played_moves.tolist()):
            contexts[sample_idx].append(int(next_token))

        rows = [torch.tensor(context, dtype=torch.long) for context in contexts]
        token_ids = pad_sequence(rows, batch_first=True, padding_value=PAD_ID).to(self.device)
        completion_mask = torch.zeros_like(token_ids, dtype=torch.float)

        for row_idx, (prompt_len, think_len) in enumerate(
            zip(expanded_prompt_lengths.tolist(), think_lengths.tolist(), strict=True)
        ):
            think_start_offset = prompt_len
            thought_start = think_start_offset + 1
            thought_end = thought_start + think_len
            final_move_idx = thought_end + 1
            completion_mask[row_idx, thought_start:thought_end] = 1.0
            completion_mask[row_idx, final_move_idx] = 1.0

        return Phase1RolloutBatch(
            token_ids=token_ids,
            completion_mask=completion_mask,
            prompt_lengths=expanded_prompt_lengths,
            think_lengths=think_lengths,
        )


def _sample_move_tokens(
    logits: torch.Tensor,
    *,
    move_token_ids: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    move_logits = logits.index_select(dim=-1, index=move_token_ids)
    if temperature <= 0:
        selected = torch.argmax(move_logits, dim=-1)
    else:
        probs = F.softmax(move_logits / temperature, dim=-1)
        selected = torch.multinomial(probs, num_samples=1).squeeze(1)
    return move_token_ids.index_select(dim=0, index=selected)
