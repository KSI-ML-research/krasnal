import bulletchess

from krasnal.eval.metrics.context import EvalContext
from krasnal.eval.parsers import GameTokens
from krasnal.inference.utils import get_legal_token_ids
from krasnal.tokens import ID_TO_MOVE

PIECE_TYPE_TO_INT = {pt: i + 1 for i, pt in enumerate(bulletchess.PIECE_TYPES)}


def replay_game_tokens(game_tokens: GameTokens) -> list[EvalContext]:
    if not game_tokens.move_tokens:
        return []

    contexts: list[EvalContext] = []
    board = bulletchess.Board()
    context = game_tokens.initial_context.copy()

    for move_idx, move_token in enumerate(game_tokens.move_tokens):
        legal_ids = get_legal_token_ids(board)
        if not legal_ids:
            break

        in_check = board in bulletchess.CHECK
        phase = get_game_phase(move_idx)

        uci_move = ID_TO_MOVE.get(move_token)
        if not uci_move:
            break
        uci_move = uci_move[1:]  # strip prefix

        try:
            move = bulletchess.Move.from_uci(uci_move)
        except Exception:
            break

        piece = board[move.origin]
        piece_type = PIECE_TYPE_TO_INT.get(piece.piece_type, 0) if piece else 0
        fen = board.fen()

        board.apply(move)
        gives_check = board in bulletchess.CHECK

        contexts.append(
            EvalContext(
                probs=None,
                legal_ids=legal_ids,
                sequence=context.copy(),
                piece_type=piece_type,
                actual_token=move_token,
                in_check=in_check,
                phase=phase,
                gives_check=gives_check,
                fen=fen,
                top1_fen=None,
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
