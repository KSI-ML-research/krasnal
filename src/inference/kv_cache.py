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
        self.key_cache = torch.zeros(num_layers, batch_size, num_heads, max_seq_len,
                                     head_dim, device=device, dtype=dtype)
        self.value_cache = torch.zeros(num_layers, batch_size, num_heads, max_seq_len,
                                       head_dim, device=device, dtype=dtype)
        # Current sequence length per batch (initially 0, will be updated during decoding)
        self.cache_seqlen = torch.zeros(batch_size, dtype=torch.long, device=device)

    def reset(self):
        """Reset the KV cache to its initial state."""
        self.cache_seqlen.zero_()

    def get_pos(self, batch_idx):
        """Get the current position in the sequence for a given batch index."""
        return self.cache_seqlen[batch_idx]

    def update_cache(self, layer_idx, batch_idx, new_keys, new_values):
        """
        Update the KV cache with new key and value tensors for a specific layer and batch index.

        Args:
            layer_idx (int): The index of the layer to update
            batch_idx (int): The index of the batch to update
            new_keys (torch.Tensor): The new key tensor to append (shape: [num_heads, head_dim])
            new_values (torch.Tensor): The new value tensor to append (shape: [num_heads, head_dim])
        """
        # Get the current position in the sequence for the given batch index
        pos = self.get_pos(batch_idx)

        # Ensure that we do not exceed the maximum sequence length
        if pos >= self.max_seq_len:
            raise ValueError(f"Sequence length exceeded max_seq_len: {pos} >= {self.max_seq_len}")

        # Update the key and value caches at the appropriate position
        self.key_cache[layer_idx, batch_idx, :, pos, :] = new_keys
        self.value_cache[layer_idx, batch_idx, :, pos, :] = new_values

        # Increment the sequence length for this batch index
        self.cache_seqlen[batch_idx] += 1

    def get_cache(self, layer_idx, batch_idx):
        """
        Retrieve the cached key and value tensors for a specific layer and batch index.

        Args:
            layer_idx (int): The index of the layer to retrieve
            batch_idx (int): The index of the batch to retrieve
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The cached key and value tensors
            (shapes: [num_heads, current_seq_len, head_dim])
        """
        # Get the current position in the sequence for the given batch index
        pos = self.get_pos(batch_idx)

        # Retrieve the cached key and value tensors up to the current position
        cached_keys = self.key_cache[layer_idx, batch_idx, :, :pos, :]
        cached_values = self.value_cache[layer_idx, batch_idx, :, :pos, :]

        return cached_keys, cached_values
