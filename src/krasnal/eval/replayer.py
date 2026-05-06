import bulletchess

from krasnal.eval.metrics.context import EvalContext
from krasnal.eval.parsers import GameTokens
from krasnal.tokens import legal_token_ids, to_uci

PIECE_TYPE_TO_INT = {pt: i + 1 for i, pt in enumerate(bulletchess.PIECE_TYPES)}


def replay_game_tokens(game_tokens: GameTokens) -> list[EvalContext]:
    if not game_tokens.move_tokens:
        return []

    what_is_on_game_key = " ".join(to_uci(t) for t in game_tokens.move_tokens)

    contexts: list[EvalContext] = []
    board = bulletchess.Board()
    context = game_tokens.initial_context.copy()

    for move_idx, move_token in enumerate(game_tokens.move_tokens):
        legal_ids = legal_token_ids(board)
        if not legal_ids:
            break

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
                probs=None,
                legal_ids=legal_ids,
                sequence=context.copy(),
                piece_type=piece_type,
                actual_token=move_token,
                in_check=in_check,
                phase=phase,
                player_elo_token=player_elo_token,
                gives_check=gives_check,
                fen=fen,
                post_move_fen=post_move_fen,
                top1_fen=None,
                what_is_on_game_key=what_is_on_game_key,
                what_is_on_ply=move_idx,
            )
        )

        context.append(move_token)

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

        if len(moves) + len(initial_context) > block_size:
            continue

        game_contexts = replay_game_tokens(game_tokens)
        contexts.extend(game_contexts)

    return contexts
