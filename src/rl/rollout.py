import chess
import torch
import torch.nn.functional as F

from src.tokenizer import SPECIAL_TOKENS, STEP_BACK_ID, THINK_END_ID, THINK_START_ID


@torch.inference_mode()
def build_prompt_boards(prompts, tokenizer):
    boards = []
    special_ids = set(SPECIAL_TOKENS)
    for row in prompts:
        board = chess.Board()
        for t_id in row.tolist():
            if t_id in special_ids:
                continue
            uci = tokenizer.id_to_move.get(int(t_id), "")
            if not uci:
                break
            try:
                board.push_uci(uci)
            except Exception:
                break
        boards.append(board)
    return boards


def generate_rollouts(
    model,
    prompts,
    group_size,
    tokenizer,
    max_new_tokens=128,
    think_min_tokens=12,
    think_max_tokens=28,
):
    B, T_p = prompts.shape
    device = prompts.device
    idx = prompts.repeat_interleave(group_size, dim=0)
    total_samples = B * group_size

    think_start = torch.full((total_samples, 1), THINK_START_ID, dtype=torch.long, device=device)
    idx = torch.cat([idx, think_start], dim=1)

    if think_max_tokens < think_min_tokens:
        think_max_tokens = think_min_tokens

    think_targets = torch.randint(
        low=think_min_tokens,
        high=think_max_tokens + 1,
        size=(total_samples,),
        device=device,
    )
    think_lengths = torch.zeros(total_samples, dtype=torch.long, device=device)
    in_think = torch.ones(total_samples, dtype=torch.bool, device=device)
    needs_post_think_move = [False for _ in range(total_samples)]

    prompt_boards = build_prompt_boards(prompts, tokenizer)
    base_boards = [board.copy() for board in prompt_boards for _ in range(group_size)]
    think_boards = [board.copy() for board in base_boards]
    special_ids = set(SPECIAL_TOKENS)
    logits_vocab_size = None

    for _ in range(max_new_tokens):
        logits, _ = model(idx)
        logits = logits[:, -1, :]
        if logits_vocab_size is None:
            logits_vocab_size = logits.size(-1)

        for i in range(total_samples):
            allowed_ids = None
            if in_think[i].item():
                legal_ids = [tokenizer.move_to_id[m.uci()] for m in think_boards[i].legal_moves]
                allowed_ids = legal_ids + [STEP_BACK_ID, THINK_END_ID]
            elif needs_post_think_move[i]:
                legal_ids = [tokenizer.move_to_id[m.uci()] for m in base_boards[i].legal_moves]
                if legal_ids:
                    allowed_ids = legal_ids

            if allowed_ids:
                mask = torch.ones(logits_vocab_size, dtype=torch.bool, device=device)
                mask[allowed_ids] = False
                logits[i, mask] = -float("inf")

        probs = F.softmax(logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1)

        force_end = in_think & (think_lengths >= think_targets)
        if force_end.any():
            next_tokens = next_tokens.clone()
            next_tokens[force_end, 0] = THINK_END_ID

        was_in_think = in_think.clone()
        in_think = in_think & (next_tokens.squeeze(1) != THINK_END_ID)
        think_lengths = (
            think_lengths + (was_in_think & (next_tokens.squeeze(1) != THINK_END_ID)).long()
        )

        for i in range(total_samples):
            next_id = int(next_tokens[i, 0].item())
            if was_in_think[i].item():
                if next_id == STEP_BACK_ID:
                    if think_boards[i].move_stack:
                        think_boards[i].pop()
                elif next_id == THINK_END_ID:
                    needs_post_think_move[i] = True
                elif next_id not in special_ids:
                    uci = tokenizer.id_to_move.get(next_id, "")
                    try:
                        move = chess.Move.from_uci(uci)
                    except ValueError:
                        move = None
                    if move and move in think_boards[i].legal_moves:
                        think_boards[i].push(move)

            if needs_post_think_move[i] and next_id not in special_ids:
                needs_post_think_move[i] = False

        idx = torch.cat([idx, next_tokens], dim=1)

    completions_mask = torch.zeros_like(idx, dtype=torch.float)
    completions_mask[:, T_p:] = 1.0
    return idx, completions_mask
