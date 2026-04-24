import torch


class KVCache:
    """
    KV Cache designed for Flash Attention 2 style.
        - Stores key and value tensors for each layer and each head.
        - Supports efficient appending of new key and value tensors during autoregressive decoding.
        - Provides methods to retrieve the cached key and value tensors for a given layer and head.

    Highly inspired by Karpathy's implementation in nanochat.
    """

    def __init__(self, batch_size, num_layers, num_heads, head_dim, max_seq_len, device, dtype):
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        # Initialize the cache for keys and values
        self.key_cache = torch.zeros(
            num_layers, batch_size, num_heads, max_seq_len, head_dim, device=device, dtype=dtype
        )
        self.value_cache = torch.zeros(
            num_layers, batch_size, num_heads, max_seq_len, head_dim, device=device, dtype=dtype
        )
        self.cache_seqlen = torch.zeros(batch_size, dtype=torch.long, device=device)

    def reset(self):
        """Reset the KV cache to its initial state."""
        self.cache_seqlen.zero_()

    def get_seq_len(self) -> int:
        """Get the cached sequence length (assumes uniform length across batch)."""
        return self.cache_seqlen[0].item()

    def append_layer(self, layer_idx, new_keys, new_values):
        """
        Append new keys/values for a single layer at the current cached position.

        Args:
            layer_idx (int): The index of the layer to update
            new_keys (torch.Tensor): Shape [batch_size, num_heads, t_new, head_dim]
            new_values (torch.Tensor): Shape [batch_size, num_heads, t_new, head_dim]
        """
        if new_keys.ndim != 4 or new_values.ndim != 4:
            raise ValueError("new_keys/new_values must have shape [B, H, T, D]")

        b, h, t_new, d = new_keys.shape
        if (b, h, d) != (self.batch_size, self.num_heads, self.head_dim):
            raise ValueError(
                "new_keys shape mismatch: "
                f"expected [B={self.batch_size}, H={self.num_heads}, T, D={self.head_dim}], "
                f"got {tuple(new_keys.shape)}"
            )
        if new_values.shape != new_keys.shape:
            raise ValueError("new_values shape must match new_keys shape")

        start = self.get_seq_len()
        end = start + t_new
        if end > self.max_seq_len:
            raise ValueError(f"Sequence length exceeded max_seq_len: {end} > {self.max_seq_len}")

        self.key_cache[layer_idx, :, :, start:end, :] = new_keys
        self.value_cache[layer_idx, :, :, start:end, :] = new_values

    def advance(self, t_new: int) -> None:
        """Advance the global cached sequence length after all layers are appended."""
        if t_new < 0:
            raise ValueError("t_new must be non-negative")
        new_len = self.get_seq_len() + t_new
        if new_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length exceeded max_seq_len: {new_len} > {self.max_seq_len}"
            )
        self.cache_seqlen += t_new

    def get_layer_cache(self, layer_idx):
        """
        Retrieve cached key/value tensors for a layer up to current sequence length.
        """
        seq_len = self.get_seq_len()
        return (
            self.key_cache[layer_idx, :, :, :seq_len, :],
            self.value_cache[layer_idx, :, :, :seq_len, :],
        )
