import torch

from krasnal.config import GPTConfig
from krasnal.model import GPT


def _tiny_config() -> GPTConfig:
    return GPTConfig(
        block_size=16,
        n_layer=1,
        n_head=2,
        n_embd=32,
        use_time_conditioning=False,
        time_conditioning_hidden=8,
        vocab_size=64,
        dropout=0.0,
    )


def test_segment_mask_blocks_cross_game_attention():
    model = GPT(_tiny_config())
    model.eval()

    B, T = 1, 4
    idx = torch.tensor([[10, 11, 20, 21]], dtype=torch.long)
    segment_ids = torch.tensor([[0, 0, 1, 1]], dtype=torch.long)
    position_ids = torch.tensor([[0, 1, 0, 1]], dtype=torch.long)

    attn = model.transformer.h[0].attn
    x = model.transformer.wte(idx)

    q, k, _v = attn.c_attn(x).split(attn.n_embd, dim=2)
    nh = attn.n_head
    hs = attn.n_embd // nh
    q = q.view(B, T, nh, hs).transpose(1, 2)
    k = k.view(B, T, nh, hs).transpose(1, 2)
    q, k = attn.rope(q, k, position_ids=position_ids)

    seg_q = segment_ids.unsqueeze(-1)
    seg_k = segment_ids.unsqueeze(-2)
    pos_q = position_ids.unsqueeze(-1)
    pos_k = position_ids.unsqueeze(-2)
    mask = ((seg_q == seg_k) & (pos_k <= pos_q)).unsqueeze(1)

    assert mask[0, 0, 2, 0].item() is False
    assert mask[0, 0, 2, 1].item() is False
    assert mask[0, 0, 3, 0].item() is False
    assert mask[0, 0, 1, 0].item() is True
    assert mask[0, 0, 3, 3].item() is True


def test_packed_forward_runs_with_segment_position_ids():
    model = GPT(_tiny_config())
    idx = torch.tensor([[10, 11, 20, 21]], dtype=torch.long)
    targets = torch.tensor([[11, 12, 21, 22]], dtype=torch.long)
    segment_ids = torch.tensor([[0, 0, 1, 1]], dtype=torch.long)
    position_ids = torch.tensor([[0, 1, 0, 1]], dtype=torch.long)

    _, loss = model(
        idx,
        targets,
        segment_ids=segment_ids,
        position_ids=position_ids,
    )
    assert loss is not None
    assert torch.isfinite(loss)
