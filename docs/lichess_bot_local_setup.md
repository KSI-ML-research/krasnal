# Lichess bot local setup

You can run our bot from your machine.

Lichess recipes are defined directly in the root `justfile`.

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
   just bot-setup
   ```
4. run the actual bot. As long as this process is running, you can play the bot on lichess
```bash
   just bot-run
   ```

5. optionally clean up the local setup
```bash
   just bot-clean
   ```

### Engine subprocess (`just bot-run artifacts/...`)

The bot passes `KRASNAL_MODEL_ARTIFACT_DIR` as an absolute path. The UCI entrypoint bootstraps `sys.path` so `import krasnal` works when lichess-bot runs `python ../src/krasnal/uci_engine/run.py` with cwd `lichess-bot/`.

Heavy model loading runs **on the first `uci` line** (after the process has started reading stdin), so python-chess’s handshake does not wait on torch import + checkpoint I/O before the child process is considered alive.

In `config.yml`, set **`engine.silence_stderr: true`** to **show** the engine’s stderr (lichess-bot’s flag is inverted: `false` sends engine stderr to `/dev/null`). The UCI entrypoint always attaches loguru to stderr at **ERROR** (stack traces on failures); use **`KRASNAL_UCI_VERBOSE=1`** for DEBUG. Failures on the first `go` also emit chunked **`info string krasnal-uci …`** lines plus `bestmove (none)` (the bot then aborts or resigns).
