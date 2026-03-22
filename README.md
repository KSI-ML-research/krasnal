# Krasnal ♟️

**Wrocławski silnik szachowy oparty na architekturze Transformer.**

## 1. Cel Projektu

Stworzenie silnika szachowego ("Krasnal") opartego na architekturze **Transformer (GPT-style)**, który potrafi generować legalne i sensowne ruchy szachowe, ucząc się bezpośrednio na bazie gier arcymistrzów i silnych amatorów.

Plan maksimum? Wgnieść w ziemię "Bestie z Wrocławia" (2100 ELO)!

---

## 2. Architektura Systemu

System składa się z trzech głównych modułów:

1.  **Data Ingestion (Rust):** Wydajne pobieranie, filtrowanie i tokenizacja gier (PGN -> UCI -> Parquet).
2.  **Model Training (Python/PyTorch):** Trening modelu Decoder-only Transformer przy użyciu biblioteki Polars do szybkiego ładowania danych.
3.  **Inference (Python/Rust):** Generowanie ruchów przez wytrenowany model zgodnie z protokołem UCI.

Szczegółowe informacje o architekturze modelu i potoku danych znajdziesz w [**docs/bot_implementation_plan.md**](docs/bot_implementation_plan.md).

---

## 3. Dokumentacja

Dla deweloperów i użytkowników przygotowaliśmy szczegółowe przewodniki:

-   [**Installation Guide**](docs/INSTALLATION.md) - Jak skonfigurować środowisko (Python, Rust, uv).
-   [**Contributing Guide**](docs/CONTRIBUTING.md) - Standardy kodu, pre-commit hooki i proces rozwoju projektu.
-   [**Experiment Notes**](docs/EXPERIMENT_NOTES.md) - Krotkie podsumowanie przetestowanych wariantow architektury.
-   [**Cloud Docker Guide**](docs/CLOUD_DOCKER.md) - Szybkie uruchamianie treningu i inferencji w kontenerach.

## 4. Oczekiwania

Zespół Google DeepMind osiągnął ELO na poziomie 2025 (+/- 18) dla Transformera o rozmiarze 9M parametrów. Naszym celem jest udowodnienie, że architektura GPT świetnie radzi sobie z logiką szachową bez tradycyjnych funkcji ewaluacyjnych.
