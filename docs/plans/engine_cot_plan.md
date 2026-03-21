# Engine CoT Plan

## Summary
Add a model-backed `ChessModelProvider` that samples moves with optional
`<think>...</think>` blocks, ignores the content of the thought block during
inference, and always returns a legal UCI move to the engine. Keep the UCI loop
unchanged and select providers via env config.

## Implementation Changes
- Add a new provider `ModelProvider` in `src/engine/model_provider.py` that:
  - Loads the model + tokenizer via `load_model`.
  - Builds a `chess.Board` from `uci_moves`.
  - Converts move history to token IDs and feeds an `InferenceSession`.
  - Calls `trace_think_block(..., force_thinking=False)` so the model decides
    whether to emit `<think>`.
  - Ignores all tokens inside `<think>` for gameplay; uses only the returned move token.
  - If the chosen move token is illegal or missing, fallback to the highest-probability
    legal move from the returned `move_probs`.
  - If there are no legal moves, return `0000`.

- Update `src/engine/run.py` to select provider by `ENGINE_ENV`:
  - `ENGINE_ENV=mock` -> `RandomMockProvider` (default for tests).
  - `ENGINE_ENV=model` -> `ModelProvider`.

- Add optional inference config via environment variables in `ModelProvider`:
  - `ENGINE_MODEL_PATH` (default `models/chess_model.pt`)
  - `ENGINE_DEVICE` (default auto: `cuda` if available else `cpu`)
  - `ENGINE_TEMPERATURE`, `ENGINE_TOP_P`, `ENGINE_MAX_TOKENS`, `ENGINE_MAX_THINK_TOKENS`

## Public API / Interface Changes
- `ENGINE_ENV` is now respected by `src/engine/run.py` to switch providers.
- New env vars (all optional) to tune inference settings.

## Test Plan
- Run existing UCI integration test with default mock:
  - `pytest tests/test_uci_engine.py`
- Add a focused unit test for `ModelProvider` legality fallback if a lightweight mock
  model is available; otherwise document a manual check:
  - Start engine with `ENGINE_ENV=model` and ensure `bestmove` is legal for a few positions.

## Assumptions
- We do not force `<think>`; the model decides when to emit it.
- The content inside `<think>` does not affect inference behavior beyond conditioning the model.
- If the model outputs an illegal final move, we select the best legal move by probability.
