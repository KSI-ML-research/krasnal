# Lichess bot local setup

You can run our bot from your machine.

Lichess bot recipes are defined in `Makefile` (not the justfile).

Read docs/bot_implementation_plan.md for architecture details.

### Bot account setup:
Lichess requires a few steps to set up a bot account.
- [step 1 - create the account & API key](https://github.com/lichess-bot-devs/lichess-bot/wiki/How-to-create-a-Lichess-OAuth-token)
- [step 2 - upgrade to bot account](https://lichess.org/api#tag/bot/POST/api/bot/account/upgrade)

### Setup:
1. run setup from docs/INSTALLATION.md
2. Create a .env file by copying the .env.example file. Update the LICHESS_BOT_TOKEN in .env file with the actual token from lichess.
3. setup lichess-bot (code responsible for integration with lichess api)
```bash
   make bot-setup
   ```
4. run the actual bot. As long as this process is running, you can play the bot on lichess
```bash
   make bot-run MODEL_PATH=artifacts/pretrain/...
   ```

5. optionally clean up the local setup
```bash
   make bot-clean
   ```

### Engine subprocess (`make bot-run MODEL_PATH=artifacts/...`)

The bot passes `KRASNAL_MODEL_ARTIFACT_DIR` as an absolute path. The UCI entrypoint bootstraps `sys.path` so `import krasnal` works when lichess-bot runs `python ../src/krasnal/uci_engine/run.py` with cwd `lichess-bot/`.

The `uci` handshake returns before heavy model loading. The provider is loaded on `isready` or the first command that needs it (`setoption`, `ucinewgame`, or `go`), so python-chess gets `uciok` without waiting on torch import + checkpoint I/O.

In `config.yml`, set **`engine.silence_stderr: true`** to **show** the engine’s stderr (lichess-bot’s flag is inverted: `false` sends engine stderr to `/dev/null`). The UCI entrypoint always attaches loguru to stderr at **ERROR** (stack traces on failures); use **`KRASNAL_UCI_VERBOSE=1`** for DEBUG. Failures on the first `go` also emit chunked **`info string krasnal-uci …`** lines plus `bestmove (none)` (the bot then aborts or resigns).
