# Krasnal

## 1. Cel Projektu

Stworzenie Wrocławskiego silnika szachowego ("Krasnal") opartego na architekturze Transformer (GPT-style), który potrafi generować legalne i sensowne ruchy szachowe, ucząc się na bazie gier arcymistrzów i silnych amatorów.

## 2. Architektura Systemu

System będzie się składać z trzech głównych modułów:

1. **Data Ingestion (Rust):** Pobieranie, filtrowanie, parsowanie SAN -> UCI, Tokenizacja -> `.parquet`.
2. **Model Training (Python/PyTorch):** Ładowanie danych (Polars), trening modelu Transformer.
3. **Inference (Python/Rust):** Generowanie ruchów przez wytrenowany model.

---

## 3. Data Pipeline

### 3.1 Źródło i Filtrowanie

- **Źródło:** Lichess Open Database (format PGN).
- **Wolumen:** 1 000 0000+ partii.

### 3.2 Filtrowanie partii

- `Elo` >= 2000 (dla obu graczy).
- `TimeControl`: min. 300s (5 minut) - odrzucamy Bullet/Blitz.
- `Result`: Tylko wygrane (1-0 lub 0-1). Odrzucamy remisy, chyba że dla remisów zwiększymy minimalne ELO (do decyzji w implementacji).
- `Termination`: Odrzucamy wygrane przez czas (`Time forfeit`), chyba że utniemy ostatnie 5 ruchów (do decyzji w implementacji).

### 3.3 Tokenizacja

Tokenami będą wszystkie możliwe unikalne ruchy w notacji UCI (np. `a2a3`, `h7h8q`). Jest ich dokładnie [**1968**](https://gist.github.com/void4/11b1623128c9a97ff461eef81edae665). Dodatkowo dodajemy tokeny specjalne `<SOS>` (Start of Sequence) i `<EOS>` (End of Sequence).

- `a2a3` -> `0`, `a2a4` -> `1`, ..., `h7h8q` -> `1967`
- `<SOS>` (Start of Sequence) - początek partii.
- `<EOS>` (End of Sequence) - koniec partii (mat/poddanie).

### 3.4 Przetwarzanie danych

Skrypt w Rustcie odpowiada za ciężką pracę CPU:

1. De-kompresja `.zst` (zstandard) z Lichess.
2. Parsowanie PGN, filtrowanie partii według kryteriów.
3. Symulacja gry (`shakmaty`) w celu konwersji SAN -> UCI.
4. Lookup w `vocab.json` -> zamiana UCI na `u16`.
5. Zapis do `.parquet` w blokach po K partii (stała do ustalenia).

---

## 4. Architektura Modelu

Model typu **Decoder-only Transformer** (architektura GPT).
Reszta do ustalenia.

---

# 5. Oczekiwania

- Plan minimum - model nauczy się z dużym prawdopodobieństwem generować legalne ruchy, ale może nie będą one zawsze sensowne.
- Plan maksimum - wgnieść w ziemię "Bestie z Wrocławia" (2100 ELO)

Plan maksimum powinien być możliwy - zespół Google Deepmind osiągnął ELO na poziomie 2025 (+/- 18) dla Transformera o rozmiarze 9M parametrów.

