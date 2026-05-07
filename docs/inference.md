# Inference API

## Core Design
Three distinct layers:
1. `Model`: Pure NN. Tensor in, tensor out. Domain-agnostic.
2. `Game`: Chess board, rules, and token synchronization. First-class state object.
3. `InferenceSession`: Runtime execution bridging `Model` and `Game`.

## API

```rust
enum EloToken {
    Unknown,
    Below1000,
    Elo1000To1499,
    Elo1500To1999,
    Elo2000To2499,
    Elo2500To2999,
    Above3000,
}

enum TargetOutcome {
    Unknown,
    WhiteWon,
    BlackWon,
    Draw,
}

struct Model {
    pub fn forward(input_ids: Tensor) -> Tensor
}

struct Game {
    white_elo: EloToken,
    black_elo: EloToken,
    target_outcome: TargetOutcome,

    board: Board,
    moves_uci: Vec<String>,
    tokens: Vec<int>,

    pub fn new(
        white_elo: EloToken,
        black_elo: EloToken,
        target_outcome: TargetOutcome,
    ) -> Self

    pub fn feed_uci(&mut self, uci: &str) -> Result<()>
    pub fn feed_token(&mut self, token: int) -> Result<()>
    pub fn context_tokens(&self) -> Vec<int>
    pub fn legal_moves(&self) -> Vec<String>
    pub fn legal_tokens(&self) -> Vec<int>
}

struct InferenceSession {
    model: Model,
    game: Game,
    kv_cache: Option<KvCache>,

    pub fn new(model: Model, game: Game) -> Self
    pub fn new_game(&mut self, game: Game)
    pub fn get_raw_logits(&mut self) -> Tensor
    pub fn get_legal_logits(&mut self) -> Tensor
    pub fn get_raw_probs(&mut self) -> Tensor
    pub fn get_legal_probs(&mut self) -> Tensor
    pub fn feed_uci(&mut self, uci: &str) -> Result<()>
    pub fn feed_token(&mut self, token: int) -> Result<()>
}

struct StatelessBatchInferenceSession {
    model: Model,

    pub fn get_raw_logits_batch(&self, games: &[Game]) -> Tensor
    pub fn get_legal_logits_batch(&self, games: &[Game]) -> Tensor
}
```

## Contracts & Invariants

- **`Game`**: Owns prompt metadata, board state, move history, and token list. If the game exists, its internal state is perfectly synchronized and legal. Mutations fail if they break these invariants.
- **`InferenceSession`**: Handles tensorization, window truncation, optional KV-caching, and legality masking (masking illegal move logits to `-inf`).
- **`Model`**: Owns NN weights and computation. Strictly avoids token decoding, board state, or move validation logic.
- **Vocabulary**: Model artifacts carry `move_vocab.json`. Inference loads it at startup and verifies that `piece_aware_moves` and `side_prefixed_moves` match the artifact config before play begins.
- **Failure handling**: unrecoverable provider/model errors are converted to a UCI resignation (`bestmove resign`) instead of guessing from a corrupted state.
