//! Filter the Lichess puzzle database by rating and export to JSONL.
//!
//! Reads the compressed CSV from `data/lichess_db_puzzle.csv.zst`, filters
//! puzzles by minimum rating (default: 1000), and writes the results to
//! `data/puzzles_filtered.jsonl`.
//!
//! Each output record contains:
//! - `fen`: board position at the start of the puzzle
//! - `solution`: first solution move in UCI notation (e.g. `e2e4`)
//! - `rating`: puzzle difficulty rating
//! - `game_url`: link to the original Lichess game
//!
//! Usage:
//! ```bash
//! just download-puzzles   # download the raw CSV (~1 GB)
//! just prepare-puzzles    # run this binary
//! ```

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use indicatif::{ProgressBar, ProgressStyle};

const MIN_RATING: u32 = 1000;
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
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let token_format = TokenFormat::Uci;

    let input_file = File::open(Path::new(INPUT_PATH))?;
    let decoder = zstd::stream::Decoder::new(input_file)?;

    let output_file = File::create(Path::new(OUTPUT_PATH))?;
    let mut writer = BufWriter::new(output_file);

    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(decoder);

    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} [{elapsed_precise}] {pos} records processed ({msg})")
            .unwrap(),
    );
    pb.set_message("filtering...");

    let mut total: u64 = 0;
    let mut written: u64 = 0;

    for result in reader.records() {
        total += 1;
        if total.is_multiple_of(10_000) {
            pb.set_position(total);
            pb.set_message(format!("{written} written so far"));
        }

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
        //   - token 0: first solution move for the puzzle
        //   - tokens 1..: remaining solution moves
        //
        // We keep ONLY the first solution move (token 0). Many puzzles have
        // multi-move solutions, but this script evaluates just the first move.
        let solution_first_move_uci = match record.get(2).and_then(|m| m.split_whitespace().next()) {
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

        let game_url = record.get(8).unwrap_or("").to_string();

        let formatted_solution = token_format.format_move(&solution_first_move_uci, &fen);

        // Note: "solution" contains only the first move of the puzzle solution.
        let entry = serde_json::json!({
            "fen": fen,
            "solution": formatted_solution,
            "rating": rating,
            "game_url": game_url,
        });

        writeln!(writer, "{entry}")?;
        written += 1;
    }

    writer.flush()?;
    pb.finish_with_message(format!("done - {written}/{total} puzzles written"));

    println!(
        "Done. Processed {total} records, wrote {written} puzzles (rating >= {MIN_RATING}) to {OUTPUT_PATH}"
    );

    Ok(())
}
