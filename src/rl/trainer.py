import torch

from src.config import GRPOConfig


def compute_advantages(rewards, group_size):
    """
    Standardizes rewards within each group to compute advantages.
    Input: (num_samples * group_size)
    Output: (num_samples * group_size)
    """
    rewards = rewards.view(-1, group_size)
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True)
    std = torch.clamp(std, min=0.2)
    advantages = (rewards - mean) / std
    return advantages.view(-1)


class GRPOTrainer:
    def __init__(self, model, config: GRPOConfig):
        self.model = model
        self.config = config

    def compute_loss(self, log_probs, old_log_probs, advantages, kl_penalty):
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = (
            torch.clamp(ratio, 1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps) * advantages
        )
        policy_loss = -torch.min(surr1, surr2).mean()
        return policy_loss + self.config.kl_coeff * kl_penalty
