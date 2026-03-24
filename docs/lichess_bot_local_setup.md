# Lichess bot local setup

You can run our bot from your machine.

Lichess recipes are maintained in a dedicated file (`lichess.just`) and imported by the root `justfile`, so commands below stay the same.

Read docs/bot_implementation_plan.md for architecture details.

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
