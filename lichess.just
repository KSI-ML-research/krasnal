# Lichess Bot Integration

LICHESS_BOT_REPO := "https://github.com/lichess-bot-devs/lichess-bot.git"
# pinned lichess bot commit so that the setup is deterministic
LICHESS_BOT_REF := "96a8f74d87a42db8039e847548fec0d9528bb079"

# Download and setup lichess-bot client
bot-setup:
    @if [ ! -d "lichess-bot" ]; then \
        echo "Cloning lichess-bot..."; \
        git clone {{LICHESS_BOT_REPO}}; \
    fi
    @echo "Pinning lichess-bot to {{LICHESS_BOT_REF}}..."
    @cd lichess-bot && git fetch --depth 1 origin {{LICHESS_BOT_REF}} && git checkout --detach FETCH_HEAD
    @echo "Creating isolated virtual environment for lichess-bot..."
    @cd lichess-bot && uv venv .venv
    @echo "Installing lichess-bot dependencies into lichess-bot/.venv..."
    @cd lichess-bot && uv pip install --python .venv/bin/python -r requirements.txt

# Run the bot locally (requires .env with LICHESS_BOT_TOKEN)
bot-run:
    @if [ ! -x "lichess-bot/.venv/bin/python" ]; then \
        echo "Missing lichess-bot venv. Run: just bot-setup"; \
        exit 1; \
    fi
    @if [ ! -x ".venv/bin/python" ]; then \
        echo "Missing project venv for engine. Run: just setup"; \
        exit 1; \
    fi
    @echo "Preparing configuration..."
    @cp config.yml.example lichess-bot/config.yml
    @sed -i '' "s|TOKEN_PLACEHOLDER|${LICHESS_BOT_TOKEN}|g" lichess-bot/config.yml
    @sed -i '' "s|ENGINE_INTERPRETER_PLACEHOLDER|../.venv/bin/python|g" lichess-bot/config.yml
    @echo "Starting bot..."
    @cd lichess-bot && .venv/bin/python lichess-bot.py

# Remove everything related to local lichess-bot setup
bot-clean:
    @echo "Cleaning lichess-bot runtime artifacts and repository..."
    @rm -rf lichess-bot
