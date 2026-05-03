# Vocabulary

## Example

`<game_start> <white_won> <elo_2000_2499> <elo_1500_1999> <think_start> w:e2e4 b:e7e5 w:g1f3 b:b8c6 <think_end> w:d2d4 <game_end>`

## Result Tokens

Result tokens control model's goal (which side to play for).

`<white_won>`, `<black_won>`, `<draw>`, `<unknown>` — prepend to sequence

## ELO Tokens

ELO tokens control model difficulty level.

`<elo_below_1000>`, `<elo_1000_1499>`, `<elo_1500_1999>`, `<elo_2000_2499>`, `<elo_2500_2999>`, `<elo_above_3000>`, `<elo_unknown>` — prepend to sequence

## UCI Move Tokens

By default, UCI moves use format `w:e2e4` for white and `b:e7e5` for black.

White: `w:e2e4`, `w:d2d4`, ...
Black: `b:e7e5`, `b:d7d5`, ...

For ablations, `side_prefixed_moves: false` switches to a shared move vocabulary without side prefix:

`e2e4`, `e7e5`, ...

## Annotation Tokens

Annotation tokens help the model learn chess rules and board representation during the training.

`<is_check>` asks if the just-played move gives check.

`<yes_check>`, `<no_check>` are answer tokens.

`<piece_type_moved>` asks what piece type just moved.

`<pawn>`, `<knight>`, `<bishop>`, `<rook>`, `<queen>`, `<king>` are answer tokens.

Training format around a move can be:

`... w:h5f7 <is_check> <yes_check> ...`

or

`... b:a7a6 <is_check> <no_check> ...`

`... w:e2e4 <piece_type_moved> <pawn> ...`

Question tokens are loss-masked, so model learns answers and chess continuation, not question timing.

## CoT Tokens (future)

CoT tokens structure internal reasoning: `<think_start>` moves... `<think_end>`.

`<think_start>`, `<think_end>` — delimit reasoning block

## Special Tokens

`<game_start>`, `<game_end>`, `<pad>`
