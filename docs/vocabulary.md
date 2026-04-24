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

UCI moves use format `w:e2e4` for white and `b:e7e5` for black.

White: `w:e2e4`, `w:d2d4`, ...
Black: `b:e7e5`, `b:d7d5`, ...

## Annotation Tokens

Annotation tokens help the model learn chess rules and board representation during the training.

`<check>` — mark check moves (future: `<capture>`, `<promotion>`, `<en-passant>`)

## CoT Tokens (future)

CoT tokens structure internal reasoning: `<think_start>` moves... `<think_end>`.

`<think_start>`, `<think_end>` — delimit reasoning block

## Special Tokens

`<game_start>`, `<game_end>`, `<pad>`
