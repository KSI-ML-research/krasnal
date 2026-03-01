#!/usr/bin/env python3
import chess
import chess.engine
import chess.pgn
import sys
import os
import datetime

def main():
    engine_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "engine", "run.py"))
    
    print("Starting UCI engines (White vs Black)...")
    try:
        engine_white = chess.engine.SimpleEngine.popen_uci([sys.executable, engine_path])
        engine_black = chess.engine.SimpleEngine.popen_uci([sys.executable, engine_path])
    except Exception as e:
        print(f"Failed to start engines: {e}")
        return

    board = chess.Board()
    
    # Initialize PGN headers for game record
    game = chess.pgn.Game()
    game.headers["Event"] = "Local Bot vs Bot Match"
    game.headers["Site"] = "Krasnal Local Environment"
    game.headers["Date"] = datetime.datetime.now().strftime("%Y.%m.%d")
    game.headers["White"] = "Krasnal Mock (White)"
    game.headers["Black"] = "Krasnal Mock (Black)"
    
    node = game
    
    print("Match started! Playing until checkmate or draw.")
    print("Press Ctrl+C to interrupt.")
    
    try:
        while not board.is_game_over():
            # Limit to 0.01s - mock responds with random moves anyway
            if board.turn == chess.WHITE:
                result = engine_white.play(board, chess.engine.Limit(time=0.01))
            else:
                result = engine_black.play(board, chess.engine.Limit(time=0.01))
                
            if result.move:
                board.push(result.move)
                node = node.add_variation(result.move)
                # Update progress bar on a single line
                print(f"Move {len(board.move_stack)}: {result.move.uci()}   ", end="\\r")
            else:
                print("\\nEngine did not return a move! Interrupting.")
                break
                
        print("\\n\\n" + "="*30)
        print("Match over!")
        
        # Final position and result
        print("\\nFinal position:")
        print(board)
        
        result = board.result()
        game.headers["Result"] = result
        print(f"\\nResult: {result}")
        
        # Save to PGN file
        games_dir = "local_games"
        os.makedirs(games_dir, exist_ok=True)
        pgn_file = os.path.join(games_dir, "bot_vs_bot_match.pgn")
        with open(pgn_file, "w") as f:
            f.write(str(game))
        print(f"\\nPGN game record saved to: {pgn_file}")
            
    except KeyboardInterrupt:
        print("\\n\\nMatch interrupted by user (Ctrl+C).")
    finally:
        engine_white.quit()
        engine_black.quit()

if __name__ == "__main__":
    main()
