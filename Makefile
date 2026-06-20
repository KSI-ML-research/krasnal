# Load .env variables if file exists
-include .env
export

LICHESS_BOT_REPO := https://github.com/lichess-bot-devs/lichess-bot.git
# pinned lichess bot commit so that the setup is deterministic
LICHESS_BOT_REF := 96a8f74d87a42db8039e847548fec0d9528bb079

.PHONY: bot-setup bot-run bot-clean

# Download and setup lichess-bot client
bot-setup:
	@if [ ! -d "lichess-bot" ]; then \
		echo "Cloning lichess-bot..."; \
		git clone $(LICHESS_BOT_REPO); \
	fi
	@echo "Pinning lichess-bot to $(LICHESS_BOT_REF)..."
	@cd lichess-bot && git fetch --depth 1 origin $(LICHESS_BOT_REF) && git checkout --detach FETCH_HEAD
	@echo "Creating isolated virtual environment for lichess-bot..."
	@cd lichess-bot && uv venv .venv
	@echo "Installing lichess-bot dependencies into lichess-bot/.venv..."
	@cd lichess-bot && uv pip install --python .venv/bin/python -r requirements.txt
	@echo "Upgrading account to bot..."
	@curl -s -X POST https://lichess.org/api/bot/account/upgrade \
		-H "Authorization: Bearer $(LICHESS_BOT_TOKEN)" \
		|| echo "Account may already be a bot or token is invalid"

# Run the bot locally (requires LICHESS_BOT_TOKEN env var)
# Usage:
#   make bot-run MODEL_PATH=artifacts/pretrain/...
# Notes:
#   - MODEL_PATH must be a directory with model.pt, config.json, move_vocab.json
#   - path is resolved relative to project root, then passed as absolute path
bot-run:
	@if [ ! -x "lichess-bot/.venv/bin/python" ]; then \
		echo "Missing lichess-bot venv. Run: make bot-setup"; \
		exit 1; \
	fi
	@if [ ! -x ".venv/bin/python" ]; then \
		echo "Missing project venv for engine. Run: uv sync"; \
		exit 1; \
	fi
	@if [ "$$(uname)" = "Darwin" ]; then \
		cp config/lichess_config.yml lichess-bot/config.yml && \
		sed -i '' "s|TOKEN_PLACEHOLDER|$(LICHESS_BOT_TOKEN)|g" lichess-bot/config.yml && \
		sed -i '' "s|ENGINE_INTERPRETER_PLACEHOLDER|../.venv/bin/python|g" lichess-bot/config.yml; \
	else \
		cp config/lichess_config.yml lichess-bot/config.yml && \
		sed -i "s|TOKEN_PLACEHOLDER|$(LICHESS_BOT_TOKEN)|g" lichess-bot/config.yml && \
		sed -i "s|ENGINE_INTERPRETER_PLACEHOLDER|../.venv/bin/python|g" lichess-bot/config.yml; \
	fi
	@echo "Starting bot..."
	@if [ "$(MODEL_PATH)" = "" ]; then \
		cd lichess-bot && LICHESS_BOT_TOKEN=$(LICHESS_BOT_TOKEN) \
			KRASNAL_ENGINE_PROVIDER=mock \
			.venv/bin/python lichess-bot.py; \
	else \
		bot_model_path=$$(realpath $(MODEL_PATH)) && \
		cd lichess-bot && LICHESS_BOT_TOKEN=$(LICHESS_BOT_TOKEN) \
			KRASNAL_MODEL_ARTIFACT_DIR=$${bot_model_path} \
			KRASNAL_ENGINE_PROVIDER=model \
			.venv/bin/python lichess-bot.py; \
	fi

# Remove everything related to local lichess-bot setup
bot-clean:
	@echo "Cleaning lichess-bot runtime artifacts and repository..."
	@rm -rf lichess-bot
