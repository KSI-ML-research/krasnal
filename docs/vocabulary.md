# Vocabulary

## Example

`<game_start> <tc_rapid_inc> <white_won> <elo_2000_2099> <elo_1500_1599> w:e2e4 b:e7e5 w:g1f3 b:b8c6 w:d2d4 <game_end>`

## Result Tokens

Result tokens control model's goal (which side to play for).

`<white_won>`, `<black_won>`, `<draw>`, `<unknown>` — prepend to sequence

## ELO Tokens

ELO tokens control model difficulty level.

`<elo_below_1000>`, 100-point buckets from `<elo_1000_1099>` through `<elo_2100_2199>`, `<elo_above_2200>`, `<elo_unknown>` — prepend to sequence

## Time Control Tokens

Time-control tokens condition the model on game pace.

Preprocessing estimates game duration as `time_initial + 40 * time_increment` seconds, then prepends one of:
- `<tc_blitz_no_inc>`: estimated duration below 480 seconds and no increment
- `<tc_blitz_inc>`: estimated duration below 480 seconds with increment
- `<tc_rapid_no_inc>`: estimated duration from 480 to 1499 seconds and no increment
- `<tc_rapid_inc>`: estimated duration from 480 to 1499 seconds with increment
- `<tc_classical>`: estimated duration at least 1500 seconds
- `<tc_unknown>`: missing time control metadata

## UCI Move Tokens

Default (side-prefixed & piece-aware): `w:pawn:e2e4`, `b:knight:g8f6`
Ablations:
- `side_prefixed_moves: false` → `pawn:e2e4`, `knight:g8f6` (no side prefix)
- `piece_aware_moves: false` → `w:e2e4`, `b:g8f6` (no piece type)
Promotion suffixes remain in UCI string (e.g., `e7e8q` ≠ `e7e8r`)

## Generated Move Vocabulary

Preprocessing builds `data/2_tokenized/move_vocab.json` from the full `data/1_filtered/` corpus before the train/eval split. Move token strings are sorted before IDs are assigned, so IDs are deterministic for a fixed corpus and config.

The file is the source of truth for move IDs during preprocessing, pretraining, and inference. It has:
- `manifest`: `piece_aware_moves`, `side_prefixed_moves`, `generation_timestamp`, `vocab_size`
- `vocab`: token string to integer ID mapping, including special tokens and generated move tokens

`just preprocess` always overwrites this file. `just pretrain` and model inference fail at startup if the runtime config does not match the manifest.

History move strings used by the UCI bridge are normalized through `normalize_history_uci_moves`, which strips training-style prefixes before replay.

## Q&A Tokens

Q&A tokens help the model learn chess rules and board representation during training.

Question tokens (loss-masked):
- `<is_check>` → answers: `<yes_check>`, `<no_check>`
- `<piece_type_moved>` → answers: `<pawn>`, `<knight>`, `<bishop>`, `<rook>`, `<queen>`, `<king>`

Training format examples:
- `... w:h5f7 <is_check> <yes_check> ...`
- `... b:a7a6 <is_check> <no_check> ...`
- `... w:e2e4 <piece_type_moved> <pawn> ...`

Square queries:
- `<whats_on_a1>` ... `<whats_on_h8>` — answers: `<empty>` or `<w:pawn>` ... `<b:king>`
- During preprocessing/eval, square is drawn from post-move FEN, per-game key (space-separated UCI string), ply, and run seed

## Special Tokens

`<game_start>`, `<game_end>`, `<pad>`
