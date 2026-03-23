# Chain of Thought (CoT) for Chess Engine

## Token Types

The model uses only chess-specific tokens:

- **UCI moves**: `<e2e4>`, `<d1d4>`, `<g1f3>`, etc.
- **Thinking tokens**: `<think>`, `</think>`, `<back>`
- **Special tokens**: `<SOS>`, `<EOS>`, `<PAD>`

## Thinking Block

- `<think>` starts a thinking block
- `</think>` ends a thinking block
- The move **right after** `</think>` is the played move
- All moves **inside** the thinking block are NOT played — they represent the search tree

## Search Tree Structure

Tokens inside `<think>...</think>` form a search tree:

- Each move must be legal from the current position
- `<back>` reverts the last move in the thinking block

### Example

Position after White plays `e2e4`:

```
Input: <e2e4>
Output: <think> <c7c5> <g1f3> <back> <d2d4> <d7d4>
```

Interpretation:
1. Engine (Black) considers `...c5`
2. Then considers `Sf3`
3. Reverts with `<back>`
4. Considers `...d4`
5. Plays `...d4` (the move after `</think>`)

## Validation Rules

1. **Thinking block**: must form valid legal move sequence from the current position
2. **Played move**: move after `</think>` must be legal from the original position (ignoring thinking block)
