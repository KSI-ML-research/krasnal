from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_masked_log_probs(
    model: torch.nn.Module,
    token_ids: torch.Tensor,
    completion_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = token_ids[:, :-1]
    targets = token_ids[:, 1:]
    target_mask = completion_mask[:, 1:]

    logits, _ = model(inputs, targets)
    log_probs = F.log_softmax(logits, dim=-1)
    gathered = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    gathered = gathered * target_mask

    token_counts = target_mask.sum(dim=-1).clamp_min(1.0)
    seq_log_prob = gathered.sum(dim=-1)
    return seq_log_prob, token_counts


def compute_grpo_loss(
    sequence_log_probs: torch.Tensor,
    token_counts: torch.Tensor,
    rewards: torch.Tensor,
    *,
    group_size: int,
) -> torch.Tensor:
    if sequence_log_probs.numel() % group_size != 0:
        raise ValueError("sequence_log_probs size must be divisible by group_size")

    norm_log_probs = sequence_log_probs / token_counts
    grouped_rewards = rewards.view(-1, group_size)
    reward_mean = grouped_rewards.mean(dim=-1, keepdim=True)
    reward_std = grouped_rewards.std(dim=-1, unbiased=False, keepdim=True).clamp_min(1e-6)
    advantages = ((grouped_rewards - reward_mean) / reward_std).view(-1)
    return -(advantages.detach() * norm_log_probs).mean()


def compute_reference_kl(
    policy_model: torch.nn.Module,
    reference_model: torch.nn.Module,
    token_ids: torch.Tensor,
    completion_mask: torch.Tensor,
) -> torch.Tensor:
    inputs = token_ids[:, :-1]
    targets = token_ids[:, 1:]
    target_mask = completion_mask[:, 1:]

    policy_logits, _ = policy_model(inputs, targets)
    reference_logits, _ = reference_model(inputs, targets)

    policy_log_probs = F.log_softmax(policy_logits, dim=-1)
    reference_log_probs = F.log_softmax(reference_logits, dim=-1)

    chosen_policy = policy_log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    chosen_reference = reference_log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    kl = (chosen_policy - chosen_reference) * target_mask
    return kl.sum() / target_mask.sum().clamp_min(1.0)
