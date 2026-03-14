use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

const MIN_RATING: u32 = 2000;
const INPUT_PATH: &str = "data/lichess_db_puzzle.csv.zst";
const OUTPUT_PATH: &str = "data/puzzles_filtered.jsonl";

#[allow(dead_code)]
enum TokenFormat {
    Uci,
    EnrichedUci,
    San,
}

impl TokenFormat {
    fn format_move(&self, raw_uci: &str, _fen: &str) -> String {
        match self {
            TokenFormat::Uci => raw_uci.to_string(),
            TokenFormat::EnrichedUci => {
                // TODO: use shakmaty to parse FEN + enrich move with piece type / capture flag
                unimplemented!("EnrichedUci requires shakmaty integration")
            }
            TokenFormat::San => {
                // TODO: use shakmaty to parse FEN + convert UCI move to Standard Algebraic Notation
                unimplemented!("San requires shakmaty integration")
            } // TODO: other token handling options
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let format = TokenFormat::Uci;

    let input_file = File::open(Path::new(INPUT_PATH))?;
    let decoder = zstd::stream::Decoder::new(input_file)?;

    let output_file = File::create(Path::new(OUTPUT_PATH))?;
    let mut writer = BufWriter::new(output_file);

    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(decoder);

    let mut total: u64 = 0;
    let mut written: u64 = 0;

    for result in reader.records() {
        total += 1;

        let record = match result {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Skipping malformed record #{total}: {e}");
                continue;
            }
        };

        let fen = match record.get(1) {
            Some(v) if !v.is_empty() => v.to_string(),
            _ => {
                eprintln!("Skipping record #{total}: missing FEN");
                continue;
            }
        };

        // Lichess `Moves` field layout:
        //   - token 0: "setup" move (opponent's last move before the puzzle starts)
        //   - tokens 1..: solution moves for the side to play
        //
        // Here we intentionally keep ONLY the first solution move (token 1). Many
        // puzzles have multi-move solutions, but this script evaluates just the
        // first move; consumers of the output should not assume a full line.
        let solution_first_move_uci =
            match record.get(2).and_then(|m| m.split_whitespace().nth(1)) {
                Some(m) => m.to_string(),
                None => {
                    eprintln!("Skipping record #{total}: missing Moves");
                    continue;
                }
            };

        let rating: u32 = match record.get(3).and_then(|r| r.parse().ok()) {
            Some(r) => r,
            None => {
                eprintln!("Skipping record #{total}: unparseable Rating");
                continue;
            }
        };

        if rating < MIN_RATING {
            continue;
        }

        let formatted_solution_first_move =
            format.format_move(&solution_first_move_uci, &fen);

        // Note: `"solution"` here contains only the first move of the puzzle solution,
        // not the entire multi-move sequence.
        let entry = serde_json::json!({
            "fen": fen,
            "solution": formatted_solution_first_move,
            "rating": rating,
        });

        writeln!(writer, "{entry}")?;
        written += 1;
    }

    writer.flush()?;

    println!(
        "Done. Processed {total} records, wrote {written} puzzles (rating >= {MIN_RATING}) to {OUTPUT_PATH}"
    );

    Ok(())
}
