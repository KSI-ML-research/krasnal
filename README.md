# Krasnal ♟️

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/KSI-ML-research/krasnal)
[![GitHub Stars](https://img.shields.io/github/stars/KSI-ML-research/krasnal)](https://github.com/KSI-ML-research/krasnal/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/KSI-ML-research/krasnal)](https://github.com/KSI-ML-research/krasnal/network)

**Wrocław-based chess engine powered by Transformer architecture.**

## 1. Project Goal

Krasnal is a Transformer-based chess engine. It aims to play strong, human-like chess — balancing the intuition of Maya Chess with the strength of Stockfish.

## 2. System Architecture

The architecture is documented using the [C4 model](https://c4model.com/).

### C4: Context Diagram

```mermaid
flowchart LR
    subgraph External["External"]
        LichessAPI[Lichess API<br/>Games Database]
        LichessOrg[lichess.org<br/>Chess Server]
        LichessBot[lichess-bot<br/>Bot Client]
        User(👤 Player)
    end

    subgraph Krasnal["System: Krasnal"]
        DataIngestion[DuckDB + Aix<br/>Data Ingestion]
        Training[Training Pipeline]
        Inference[UCI Engine]
        Model[Transformer Model]
    end

    LichessAPI -->|"PGN"| DataIngestion
    DataIngestion -->|"Parquet"| Training
    Training -->|"Model weights"| Model
    User -->|"plays"| LichessOrg
    LichessOrg <-->|"UCI"| LichessBot
    LichessBot <--"UCI"--> Inference
    Inference --> Model
```

### C4: Container Diagram

```mermaid
flowchart TD
    subgraph Data["Data Ingestion (Python)"]
        AixDB[Aix Database<br/>HuggingFace]
        DuckDB[Aix DuckDB<br/>Extension]
        ParquetRaw[("Parquet<br/>Filtered games")]
    end

    subgraph Preprocess["Preprocessing (Python)"]
        Tokenizer[Tokenizer]
        Conditioning[Outcome Conditioning<br/>ELO + Result tokens]
        ParquetTokenized[("Parquet<br/>Tokenized training data")]
    end

    subgraph Training["Model Training (Python + PyTorch)"]
        WAndB[W&B<br/>Logging]
        GPTTraining[GPT Training]
        Artifacts[("Model artifacts<br/>.pt + config")]
    end

    subgraph Inference["Inference (Python)"]
        UCI[UCI Parser]
        Provider[Model Provider]
        Generator[Move Generator]
    end

    subgraph External["External"]
        LichessBot[lichess-bot]
    end

    AixDB -->|"Parquet"| DuckDB -->|"UCI + FEN"| ParquetRaw
    ParquetRaw --> Tokenizer --> Conditioning --> ParquetTokenized
    ParquetTokenized --> GPTTraining --> Artifacts
    GPTTraining -.->|".pt + evals"| WAndB
    Artifacts --> Provider
    LichessBot <--"stdin/stdout (UCI)"--> UCI
    UCI --> Provider --> Generator
```

---

## 3. Documentation

Detailed guides for developers and users:

-   [**Installation Guide**](docs/INSTALLATION.md) - How to set up the environment (Python, uv).
-   [**Training Pipeline**](docs/training_pipeline.md) - Download games, preprocess, and pretrain.
-   [**Contributing Guide**](docs/CONTRIBUTING.md) - Code standards, pre-commit hooks and development process.
-   [**Research Notes**](docs/RESEARCH.md) - Summary of tested architecture variants and experiments.
-   [**Outcome conditioning**](docs/outcome_conditioning.md) - Prefix with a win/loss token so the model can be steered toward playing for White or Black.
-   [**Chain-of-thought (WIP)**](docs/cot.md) - CoT training notes.
-   [**Weights & Biases**](docs/wandb.md) - Experiment logging.
-   [**Lichess bot (local)**](docs/lichess_bot_local_setup.md) - Run the bot from your machine.
-   [**Bot implementation plan**](docs/bot_implementation_plan.md) - Lichess integration architecture.

## 4. Configuration Layout

-   Hydra configs for training and generation live in `config/`.
-   Lichess bot template config lives at `config/config.yml.example`.
-   Existing `just` command usage remains the same.
