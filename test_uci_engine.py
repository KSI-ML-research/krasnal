import asyncio
import chess.engine
import sys

async def test_krasnal():
    print("Uruchamiam silnik Krasnal...")
    
    # Uruchamiamy nasz silnik dokładnie tak, jak zrobi to Lichess.
    # Musimy dodać 'src' do ścieżki PYTHONPATH, żeby importy 'engine.xxx' działały.
    transport, engine = await chess.engine.popen_uci(
        [sys.executable, "-c", "import sys; sys.path.insert(0, 'src'); from engine import run; run.main()"]
    )

    print("✅ Silnik poprawnie odpowiedział na handshake'a (uci, uciok, isready, readyok)!")

    # Testowanie losowego, ale legalnego ruchu
    board = chess.Board()
    print("Testuję wygenerowanie ruchu z pozycji startowej...")
    result = await engine.play(board, chess.engine.Limit(time=0.1))
    print(f"✅ Krasnal zagrał ze startu: {result.move}")
    
    # Ręczne dodanie historii ruchów (symulacja rozgrywki z przeciwnikiem)
    board.push(result.move)
    board.push(chess.Move.from_uci("e7e5"))  # Przykładowy ruch czarnych

    print(f"Testuję wygenerowanie kolejnego ruchu dla historii: {board.move_stack}...")
    result = await engine.play(board, chess.engine.Limit(time=0.1))
    print(f"✅ Krasnal zagrał z dalszej pozycji: {result.move}")

    # Poprawne wysłanie polecenia wyłączenia silnika (quit)
    await engine.quit()
    print("✅ Pętla UCI poprawnie zamknięta po otrzymaniu komendy 'quit'.")

if __name__ == "__main__":
    asyncio.run(test_krasnal())
