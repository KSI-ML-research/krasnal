#!/usr/bin/env python3
import chess
import chess.engine
import sys
import os

def main():
    engine_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "engine", "run.py"))
    
    print("Starting UCI engine...")
    try:
        # sys.executable ensures using the same python that started the script (uv environment)
        engine = chess.engine.SimpleEngine.popen_uci([sys.executable, engine_path])
    except Exception as e:
        print(f"Failed to start engine: {e}")
        return

    board = chess.Board()
    
    user_color_input = input("Do you want to play as White or Black? (w/b) [default: w]: ").strip().lower()
    user_is_white = user_color_input != 'b'
    
    print("\\nGame started! Enter moves in UCI format (e.g., e2e4, g1f3).")
    print("Enter 'quit' or 'q' to exit.")

    try:
        while not board.is_game_over():
            print("\\n" + "="*30)
            # Print board with coordinates (always from White's perspective)
            board_lines = str(board).splitlines()
            for i, line in enumerate(board_lines):
                print(f"{8 - i} | {line}")
            print("  +-----------------------")
            print("    a b c d e f g h")
            print("="*30)

            if board.turn == (chess.WHITE if user_is_white else chess.BLACK):
                # Player's move
                while True:
                    move_str = input("Your move: ").strip()
                    if move_str.lower() in ["quit", "exit", "q"]:
                        print("Game interrupted.")
                        return
                    
                    try:
                        move = chess.Move.from_uci(move_str)
                        if move in board.legal_moves:
                            board.push(move)
                            break
                        else:
                            print("Illegal move! Try again.")
                    except ValueError:
                        print("Invalid format! Use UCI format (e.g., e2e4).")
            else:
                # Bot's move
                print("Bot is thinking...")
                # Time limit for move (for mock it will be instantaneous anyway)
                result = engine.play(board, chess.engine.Limit(time=0.1))
                if result.move:
                    print(f"Bot plays: {result.move.uci()}")
                    board.push(result.move)
                else:
                    print("Bot did not return any move!")
                    break
                    
        print("\\n" + "="*30)
        print("Game over!")
        print(f"Result: {board.result()}")
        
    except KeyboardInterrupt:
        print("\\nGame interrupted (Ctrl+C).")
    finally:
        engine.quit()

if __name__ == "__main__":
    main()
