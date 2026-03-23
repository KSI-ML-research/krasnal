from dataclasses import dataclass


@dataclass
class RLPhase1Config:
    batch_size: int = 8
    sft_batch_size: int = 32
    group_size: int = 8
    think_min_tokens: int = 2
    think_max_tokens: int = 8
    learning_rate: float = 1e-5
    min_lr: float = 1e-6
    warmup_iters: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    kl_coef: float = 0.1
    sft_coef: float = 1.0
    sft_mix_ratio: float = 0.3
    temperature: float = 1.0
    max_prompt_tokens: int = 256
    log_every: int = 10
    save_minutes: float = 30.0
    compile: bool = False
    compile_mode: str = "default"
    compile_dynamic: bool = False

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.sft_batch_size <= 0:
            raise ValueError("sft_batch_size must be > 0")
        if self.group_size <= 0:
            raise ValueError("group_size must be > 0")
        if self.think_min_tokens <= 0:
            raise ValueError("think_min_tokens must be > 0")
        if self.think_max_tokens < self.think_min_tokens:
            raise ValueError("think_max_tokens must be >= think_min_tokens")
        if self.warmup_iters <= 0:
            raise ValueError("warmup_iters must be > 0")
        if self.save_minutes <= 0:
            raise ValueError("save_minutes must be > 0")
        if not 0.0 <= self.sft_mix_ratio <= 1.0:
            raise ValueError("sft_mix_ratio must be between 0.0 and 1.0")
