# Chain of Thought (CoT) for Chess Engine

## Token Types

The model uses only chess-specific tokens:

- **UCI moves**: `<e2e4>`, `<d1d4>`, `<g1f3>`, etc.
- **Thinking tokens**: `<think>`, `</think>`, `<branch>`
- **Special tokens**: `<SOS>`, `<EOS>`, `<PAD>`

## Thinking Block

- `<think>` starts a thinking block
- `</think>` ends a thinking block
- The move **right after** `</think>` is the played move
- All moves **inside** the thinking block are NOT played — they represent the search tree

## Search Tree Structure

Tokens inside `<think>...</think>` form a list of candidate branches:

- Each branch starts from the original position at the start of the thinking block
- Each move inside a branch must be legal from that branch's current position
- `<branch>` ends the current candidate line and starts a new one from the think-root

### Example

Position after White plays `e2e4`:

```
Input: <e2e4>
Output: <think> <c7c5> <g1f3> <branch> <e7e5> <g1f3> <branch> <d7d5> <g1f3> </think> <c7c5>
```

Interpretation:
1. Engine (Black) considers the first candidate line `...c5, Nf3`
2. Starts a new branch from the think-root
3. Engine considers another candidate line `...e5, Nf3`
4. Starts a third branch from the think-root
5. Engine considers a final candidate line `...d5, Nf3`
6. Plays `...c5` after `</think>` as the chosen move

## Validation Rules

1. **Thinking block**: must form valid legal move sequence from the current position
2. **Played move**: move after `</think>` must be legal from the original position (ignoring thinking block)
