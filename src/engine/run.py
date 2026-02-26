#!/usr/bin/env python3

from engine.mock_provider import RandomMockProvider
from engine.uci_parser import UCIParser

def main():
    """
    Entrypoint do silnika UCI bota "Krasnal". Skleja mocka z pętlą UCI.
    """
    # TODO: Zmienna środowiskowa `ENGINE_ENV` determinująca wstrzyknięcie 
    # odpowiedniego Providera (Model PyTorch lub MockProvider).
    # Na ten moment sztywno ustawiamy używanie Mocka.

    # 1. Wstrzyknięcie implementacji "ChessModelProvider"
    provider = RandomMockProvider()

    # 2. Zainicjalizowanie pętli UCI naszym Mockiem.
    uci = UCIParser(provider)

    # 3. Uruchomienie parsera oczekującego na wejście Lichess-bota (stdin)
    # i na jego podstawie przekazującego komendy Providerowi.
    uci.run()

if __name__ == "__main__":
    main()
