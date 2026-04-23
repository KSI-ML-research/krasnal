//! Download and convert chess games for specific players from PGN Mentor.
//!
//! Fetches PGN ZIP archives, converts moves to UCI notation using shakmaty,
//! and saves one JSONL file per player under `data/players/`.
//!
//! Each output record contains:
//! - `moves`: space-separated UCI move string
//! - `result`: "1-0", "0-1", or "1/2-1/2"
//! - `white`: white player name
//! - `black`: black player name
//! - `year`: year the game was played
//! - `player`: the player this file is associated with
//!
//! Usage:
//! ```bash
//! just download-player-games              # interactive
//! just download-player-games -- --all     # all players
//! just download-player-games -- tal carlsen  # specific players
//! ```

use std::collections::HashMap;
use std::fmt::Write as FmtWrite;
use std::fs::{self, File};
use std::io::{BufWriter, Cursor, Read, Write};
use std::ops::ControlFlow;
use std::path::Path;

use indicatif::{ProgressBar, ProgressStyle};
use pgn_reader::{RawTag, SanPlus, Skip, Visitor};
use shakmaty::{CastlingMode, Chess, Position};
use zip::ZipArchive;

const OUTPUT_DIR: &str = "data/players";

fn players() -> Vec<(&'static str, &'static str)> {
    vec![
        ("tal", "https://www.pgnmentor.com/players/Tal.zip"),
        ("fischer", "https://www.pgnmentor.com/players/Fischer.zip"),
        ("kasparov", "https://www.pgnmentor.com/players/Kasparov.zip"),
        ("karpov", "https://www.pgnmentor.com/players/Karpov.zip"),
        (
            "capablanca",
            "https://www.pgnmentor.com/players/Capablanca.zip",
        ),
        ("carlsen", "https://www.pgnmentor.com/players/Carlsen.zip"),
        ("nakamura", "https://www.pgnmentor.com/players/Nakamura.zip"),
        ("caruana", "https://www.pgnmentor.com/players/Caruana.zip"),
        ("polgar", "https://www.pgnmentor.com/players/Polgar.zip"),
        (
            "petrosian",
            "https://www.pgnmentor.com/players/Petrosian.zip",
        ),
    ]
}

#[derive(Default)]
struct GameData {
    moves: String,
    result: String,
    white: String,
    black: String,
    year: Option<String>,
}

struct PlayerVisitor {
    current_game: GameData,
    position: Chess,
    skip_game: bool,
    ply_count: usize,
    records: Vec<GameData>,
}

impl PlayerVisitor {
    fn new() -> Self {
        Self {
            current_game: GameData::default(),
            position: Chess::default(),
            skip_game: false,
            ply_count: 0,
            records: Vec::new(),
        }
    }
}

impl Visitor for PlayerVisitor {
    type Tags = ();
    type Movetext = ();
    type Output = ();

    fn begin_tags(&mut self) -> ControlFlow<Self::Output, Self::Tags> {
        self.current_game = GameData::default();
        self.current_game.moves.reserve(512);
        self.position = Chess::default();
        self.skip_game = false;
        self.ply_count = 0;
        ControlFlow::Continue(())
    }

    fn tag(
        &mut self,
        _tags: &mut Self::Tags,
        key: &[u8],
        value: RawTag<'_>,
    ) -> ControlFlow<Self::Output> {
        if self.skip_game {
            return ControlFlow::Continue(());
        }

        let val_str = std::str::from_utf8(value.as_bytes())
            .unwrap_or_default()
            .trim()
            .trim_matches('"');

        match key {
            b"Result" => match val_str {
                "1-0" | "0-1" | "1/2-1/2" => self.current_game.result = val_str.to_string(),
                _ => self.skip_game = true,
            },
            b"White" => self.current_game.white = val_str.to_string(),
            b"Black" => self.current_game.black = val_str.to_string(),
            b"Date" | b"UTCDate"
                if self.current_game.year.is_none()
                    && val_str
                        .split('.')
                        .next()
                        .is_some_and(|y| y != "????" && y != "?") =>
            {
                if let Some(year) = val_str.split('.').next() {
                    self.current_game.year = Some(year.to_string());
                }
            }
            _ => {}
        }
        ControlFlow::Continue(())
    }

    fn begin_movetext(&mut self, _tags: Self::Tags) -> ControlFlow<Self::Output, Self::Movetext> {
        if self.skip_game || self.current_game.result.is_empty() {
            return ControlFlow::Break(());
        }
        ControlFlow::Continue(())
    }

    fn begin_variation(
        &mut self,
        _movetext: &mut Self::Movetext,
    ) -> ControlFlow<Self::Output, Skip> {
        ControlFlow::Continue(Skip(true))
    }

    fn san(
        &mut self,
        _movetext: &mut Self::Movetext,
        san_plus: SanPlus,
    ) -> ControlFlow<Self::Output> {
        let m = match san_plus.san.to_move(&self.position) {
            Ok(m) => m,
            Err(_) => return ControlFlow::Break(()),
        };
        if !self.current_game.moves.is_empty() {
            self.current_game.moves.push(' ');
        }
        let _ = write!(
            self.current_game.moves,
            "{}",
            m.to_uci(CastlingMode::Standard)
        );
        self.position.play_unchecked(m);
        self.ply_count += 1;
        ControlFlow::Continue(())
    }

    fn end_game(&mut self, _movetext: Self::Movetext) -> Self::Output {
        if !self.skip_game && !self.current_game.moves.is_empty() {
            self.records.push(std::mem::take(&mut self.current_game));
        }
    }
}

fn fetch_pgn(url: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let client = reqwest::blocking::Client::builder()
        .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        .build()?;

    let response = client.get(url).send()?;

    if !response.status().is_success() {
        return Err(format!("Błąd pobierania, status HTTP: {}", response.status()).into());
    }

    Ok(response.bytes()?.to_vec())
}

fn extract_pgn_from_zip(zip_bytes: Vec<u8>) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let cursor = Cursor::new(zip_bytes);
    let mut archive = ZipArchive::new(cursor)?;

    // find largest PGN file by iterating indices
    let mut best_idx = None;
    let mut best_size = 0u64;
    for i in 0..archive.len() {
        let file = archive.by_index(i)?;
        if file.name().ends_with(".pgn") && file.size() > best_size {
            best_size = file.size();
            best_idx = Some(i);
        }
    }

    let idx = best_idx.ok_or("No PGN file found in ZIP")?;
    let mut file = archive.by_index(idx)?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;
    Ok(buf)
}

fn process_player(player: &str, url: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n[{}]", player.to_uppercase());
    println!("  Downloading {}...", url);

    let zip_bytes = fetch_pgn(url)?;
    let pgn_bytes = extract_pgn_from_zip(zip_bytes)?;

    println!("  Converting PGN to UCI...");

    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} [{elapsed_precise}] {pos} games processed")
            .unwrap(),
    );

    let mut visitor = PlayerVisitor::new();
    let mut reader = pgn_reader::Reader::new(pgn_bytes.as_slice());

    let mut game_count: u64 = 0;
    while reader.read_game(&mut visitor)?.is_some() {
        game_count += 1;
        if game_count.is_multiple_of(100) {
            pb.set_position(game_count);
        }
    }
    pb.finish_and_clear();

    let records = std::mem::take(&mut visitor.records);
    let valid = records.len();

    if valid == 0 {
        println!("  No valid games found.");
        return Ok(());
    }

    fs::create_dir_all(OUTPUT_DIR)?;
    let output_path = Path::new(OUTPUT_DIR).join(format!("{player}.jsonl"));
    let file = File::create(&output_path)?;
    let mut writer = BufWriter::new(file);

    for record in &records {
        let entry = serde_json::json!({
            "moves": record.moves,
            "result": record.result,
            "white": record.white,
            "black": record.black,
            "year": record.year,
            "player": player,
        });
        writeln!(writer, "{entry}")?;
    }
    writer.flush()?;

    println!("  Saved {valid} games to {}", output_path.display());
    Ok(())
}

fn interactive_select(player_list: &[(&str, &str)]) -> Vec<String> {
    println!("\nAvailable players:");
    for (i, (name, _)) in player_list.iter().enumerate() {
        println!("  {:2}. {}", i + 1, name);
    }
    println!("\nEnter numbers separated by spaces (e.g. 1 3 5), or 'all':");
    print!("> ");
    std::io::stdout().flush().unwrap();

    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();
    let input = input.trim().to_lowercase();

    if input == "all" {
        return player_list
            .iter()
            .map(|(name, _)| name.to_string())
            .collect();
    }

    let name_set: HashMap<&str, bool> = player_list.iter().map(|(n, _)| (*n, true)).collect();
    let mut selected = Vec::new();

    for token in input.split_whitespace() {
        if let Ok(idx) = token.parse::<usize>() {
            if idx >= 1 && idx <= player_list.len() {
                selected.push(player_list[idx - 1].0.to_string());
            } else {
                eprintln!("  Skipping out-of-range: {token}");
            }
        } else if name_set.contains_key(token) {
            selected.push(token.to_string());
        } else {
            eprintln!("  Unknown player: {token}");
        }
    }

    selected
}

fn parse_args() -> (bool, Vec<String>) {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--all") {
        return (true, Vec::new());
    }
    let players: Vec<String> = args.into_iter().filter(|a| !a.starts_with('-')).collect();
    (false, players)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let player_list = players();
    let player_map: HashMap<&str, &str> = player_list.iter().map(|(n, u)| (*n, *u)).collect();

    let (all, explicit) = parse_args();

    let selected: Vec<String> = if all {
        player_list.iter().map(|(n, _)| n.to_string()).collect()
    } else if !explicit.is_empty() {
        explicit
    } else {
        interactive_select(&player_list)
    };

    if selected.is_empty() {
        println!("No players selected, exiting.");
        return Ok(());
    }

    println!("\nSelected: {}", selected.join(", "));

    for player in &selected {
        match player_map.get(player.as_str()) {
            Some(url) => {
                if let Err(e) = process_player(player, url) {
                    eprintln!("  Error processing {player}: {e}");
                }
            }
            None => eprintln!("  Unknown player: {player}"),
        }
    }

    println!("\nDone.");
    Ok(())
}
