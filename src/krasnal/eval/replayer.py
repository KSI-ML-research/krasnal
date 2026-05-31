import bulletchess

from krasnal.config import CLOCK_IGNORE_ID
from krasnal.eval.metrics.context import EvalContext
from krasnal.eval.parsers import GameTokens
from krasnal.tokens import is_move_token_id, to_uci

PIECE_TYPE_TO_INT = {pt: i + 1 for i, pt in enumerate(bulletchess.PIECE_TYPES)}


def _clock_for_body_token(values: list[int] | None, body_idx: int) -> int:
    if values is None or body_idx >= len(values):
        return CLOCK_IGNORE_ID
    value = int(values[body_idx])
    return CLOCK_IGNORE_ID if value >= CLOCK_IGNORE_ID else value


def replay_game_tokens(game_tokens: GameTokens) -> list[EvalContext]:
    if not game_tokens.move_tokens:
        return []

    what_is_on_game_key = " ".join(to_uci(t) for t in game_tokens.move_tokens)

    contexts: list[EvalContext] = []
    board = bulletchess.Board()
    context = game_tokens.initial_context.copy()
    if (
        game_tokens.prefix_active_seconds is not None
        and game_tokens.prefix_opponent_seconds is not None
    ):
        pa, po = game_tokens.prefix_active_seconds, game_tokens.prefix_opponent_seconds
    else:
        pa, po = CLOCK_IGNORE_ID, CLOCK_IGNORE_ID
    clock_active = [pa] * len(context)
    clock_opponent = [po] * len(context)

    move_idx = 0
    for body_idx, token in enumerate(game_tokens.body_tokens):
        body_active = _clock_for_body_token(game_tokens.body_active_seconds, body_idx)
        body_opponent = _clock_for_body_token(game_tokens.body_opponent_seconds, body_idx)
        if not is_move_token_id(token):
            context.append(token)
            clock_active.append(body_active)
            clock_opponent.append(body_opponent)
            continue

        move_token = token
        in_check = board in bulletchess.CHECK
        phase = get_game_phase(move_idx)
        player_elo_token = (
            game_tokens.white_elo_token if move_idx % 2 == 0 else game_tokens.black_elo_token
        )

        uci_move = to_uci(move_token)
        if not uci_move:
            break

        try:
            move = bulletchess.Move.from_uci(uci_move)
        except Exception:
            break

        piece = board[move.origin]
        piece_type = PIECE_TYPE_TO_INT.get(piece.piece_type, 0) if piece else 0
        fen = board.fen()

        board.apply(move)
        gives_check = board in bulletchess.CHECK
        post_move_fen = board.fen()

        contexts.append(
            EvalContext(
                sequence=context.copy(),
                piece_type=piece_type,
                actual_token=move_token,
                in_check=in_check,
                phase=phase,
                player_elo_token=player_elo_token,
                gives_check=gives_check,
                fen=fen,
                post_move_fen=post_move_fen,
                what_is_on_game_key=what_is_on_game_key,
                what_is_on_ply=move_idx,
                active_clock_seconds=None if body_active >= CLOCK_IGNORE_ID else body_active,
                opponent_clock_seconds=None if body_opponent >= CLOCK_IGNORE_ID else body_opponent,
                active_clock_sequence=clock_active.copy(),
                opponent_clock_sequence=clock_opponent.copy(),
            )
        )

        context.append(move_token)
        clock_active.append(body_active)
        clock_opponent.append(body_opponent)
        move_idx += 1

    return contexts


def get_game_phase(move_idx: int) -> str:
    if move_idx < 20:
        return "opening"
    elif move_idx < 80:
        return "middlegame"
    else:
        return "endgame"


def replay_games(game_tokens_list: list[GameTokens], block_size: int) -> list[EvalContext]:
    contexts: list[EvalContext] = []

    for game_tokens in game_tokens_list:
        moves = game_tokens.move_tokens
        initial_context = game_tokens.initial_context

        if len(moves) < 1:
            continue

        if len(game_tokens.body_tokens) + len(initial_context) > block_size:
            continue

        game_contexts = replay_game_tokens(game_tokens)
        contexts.extend(game_contexts)

    return contexts
