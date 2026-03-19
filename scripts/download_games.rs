use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use pgn_reader::{RawTag, Reader, SanPlus, Visitor};
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use shakmaty::{CastlingMode, Chess, Position};
use std::collections::HashMap;
use std::fmt::Write;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::ops::ControlFlow;
use std::path::Path;

#[derive(Serialize, Deserialize, Default, Clone)]
pub struct ManifestEntry {
    pub status: String, // "completed" or "partial"
    pub games_count: usize,
}

#[derive(Serialize, Deserialize, Default)]
pub struct Manifest {
    pub links: HashMap<String, ManifestEntry>,
}

impl Manifest {
    pub fn load(path: &str) -> Self {
        File::open(path)
            .map(|f| serde_json::from_reader(f).unwrap_or_default())
            .unwrap_or_default()
    }

    pub fn save(&self, path: &str) -> Result<()> {
        let file = File::create(path)?;
        serde_json::to_writer_pretty(file, self)?;
        Ok(())
    }
}

pub struct Config {
    pub min_elo: i32,
    pub min_plys: usize,
    pub max_elo_diff: i32,
    pub min_base_time_s: i32,
    pub include_draws: bool,
    pub batch_size: usize,
    pub target_games: usize,
    pub links_path: &'static str,
    pub output_dir: &'static str,
    pub manifest_path: &'static str,
}

pub const CONFIG: Config = Config {
    min_elo: 2000,
    min_plys: 20,
    max_elo_diff: 3000,
    min_base_time_s: 300,
    include_draws: false,
    batch_size: 50_000,
    target_games: 10_000_000,
    links_path: "data/download_links.txt",
    output_dir: "data/raw",
    manifest_path: "data/manifest.json",
};

#[derive(Default, Clone)]
pub struct GameData {
    pub white_elo: i32,
    pub black_elo: i32,
    pub result: Option<i8>,
    pub moves: String,
    pub opening: String,
}

pub struct FilteredVisitor {
    pub current_game: GameData,
    pub skip_game: bool,
    pub valid_time_control: bool,
    pub ply_count: usize,
    pub position: Chess,
}

impl Default for FilteredVisitor {
    fn default() -> Self {
        Self::new()
    }
}

impl FilteredVisitor {
    pub fn new() -> Self {
        Self {
            current_game: GameData::default(),
            skip_game: false,
            valid_time_control: false,
            ply_count: 0,
            position: Chess::default(),
        }
    }
}

impl Visitor for FilteredVisitor {
    type Tags = ();
    type Movetext = ();
    type Output = Option<GameData>;

    fn begin_tags(&mut self) -> ControlFlow<Self::Output, Self::Tags> {
        self.current_game = GameData::default();
        self.current_game.moves.reserve(512);
        self.skip_game = false;
        self.valid_time_control = false;
        self.ply_count = 0;
        self.position = Chess::default();
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
        let key_str = std::str::from_utf8(key).unwrap_or_default();
        let val_bytes = value.as_bytes();
        let val_str = std::str::from_utf8(val_bytes)
            .unwrap_or_default()
            .trim()
            .trim_matches('"');

        match key_str {
            "WhiteElo" => self.current_game.white_elo = val_str.parse().unwrap_or(0),
            "BlackElo" => self.current_game.black_elo = val_str.parse().unwrap_or(0),
            "Result" => {
                self.current_game.result = match val_str {
                    "1-0" => Some(1),
                    "0-1" => Some(-1),
                    "1/2-1/2" => Some(0),
                    _ => None,
                };
            }
            "Opening" => self.current_game.opening = val_str.to_string(),
            "TimeControl" => {
                if let Some(secs) = val_str
                    .split('+')
                    .next()
                    .and_then(|base| base.parse::<i32>().ok())
                {
                    if secs >= CONFIG.min_base_time_s {
                        self.valid_time_control = true;
                    } else {
                        self.skip_game = true;
                    }
                }
            }
            _ => {}
        }
        ControlFlow::Continue(())
    }

    fn begin_movetext(&mut self, _tags: Self::Tags) -> ControlFlow<Self::Output, Self::Movetext> {
        if self.skip_game || !self.valid_time_control {
            return ControlFlow::Break(None);
        }
        if self.current_game.white_elo < CONFIG.min_elo
            || self.current_game.black_elo < CONFIG.min_elo
        {
            return ControlFlow::Break(None);
        }
        if (self.current_game.white_elo - self.current_game.black_elo).abs() > CONFIG.max_elo_diff {
            return ControlFlow::Break(None);
        }
        match self.current_game.result {
            Some(1) | Some(-1) => {}
            Some(0) => {
                if !CONFIG.include_draws {
                    return ControlFlow::Break(None);
                }
            }
            _ => return ControlFlow::Break(None),
        }
        ControlFlow::Continue(())
    }

    fn san(
        &mut self,
        _movetext: &mut Self::Movetext,
        san_plus: SanPlus,
    ) -> ControlFlow<Self::Output> {
        let m = match san_plus.san.to_move(&self.position) {
            Ok(m) => m,
            Err(_) => return ControlFlow::Break(None),
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
        if self.ply_count < CONFIG.min_plys {
            return None;
        }
        Some(std::mem::take(&mut self.current_game))
    }
}

pub fn save_batch(games: &[GameData], filename: &str) -> Result<()> {
    if games.is_empty() {
        return Ok(());
    }
    let h = games.len();
    let mut w_elo = Vec::with_capacity(h);
    let mut b_elo = Vec::with_capacity(h);
    let mut res = Vec::with_capacity(h);
    let mut mvs = Vec::with_capacity(h);
    let mut ops = Vec::with_capacity(h);

    for g in games {
        w_elo.push(g.white_elo);
        b_elo.push(g.black_elo);
        res.push(g.result);
        mvs.push(g.moves.clone());
        ops.push(g.opening.clone());
    }

    let mut df = DataFrame::new(
        h,
        vec![
            Series::new("white_elo".into(), w_elo).into(),
            Series::new("black_elo".into(), b_elo).into(),
            Series::new("result".into(), res).into(),
            Series::new("moves".into(), mvs).into(),
            Series::new("opening".into(), ops).into(),
        ],
    )?;

    let mut file = File::create(filename)?;
    ParquetWriter::new(&mut file).finish(&mut df)?;
    Ok(())
}

pub fn get_file_prefix(link: &str) -> String {
    Path::new(link)
        .file_name()
        .and_then(|n| n.to_str())
        .map(|s| s.replace(".pgn.zst", ""))
        .unwrap_or_else(|| "unknown".to_string())
}

fn main() -> Result<()> {
    std::fs::create_dir_all(CONFIG.output_dir)?;

    let manifest_path = CONFIG.manifest_path;
    let mut manifest = Manifest::load(manifest_path);

    let links_file = std::path::Path::new(CONFIG.links_path);
    if !links_file.exists() {
        println!("Links file not found at {}", CONFIG.links_path);
        return Ok(());
    }

    let mut links: Vec<String> = BufReader::new(File::open(links_file)?)
        .lines()
        .map_while(Result::ok)
        .filter(|l| !l.trim().is_empty())
        .collect();

    // Sort to process links from oldest to newest
    links.sort();

    let mut total_games = 0;
    for entry in manifest.links.values() {
        total_games += entry.games_count;
    }

    println!("Initial total games: {}", total_games);
    if total_games >= CONFIG.target_games {
        println!("Target reached! ({}/{})", total_games, CONFIG.target_games);
        return Ok(());
    }

    let pb = ProgressBar::new(CONFIG.target_games as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")?
            .progress_chars("#>-"),
    );
    pb.set_position(total_games as u64);

    for (i, link) in links.iter().enumerate() {
        if total_games >= CONFIG.target_games {
            pb.finish_with_message("Target games reached!");
            break;
        }

        if manifest
            .links
            .get(link)
            .is_some_and(|entry| entry.status == "completed")
        {
            continue;
        }

        pb.set_message(format!(
            "Processing {} ({}/{})",
            get_file_prefix(link),
            i + 1,
            links.len()
        ));

        match process_link(link, &mut total_games, &pb) {
            Ok((count, fully_completed)) => {
                if fully_completed {
                    manifest.links.insert(
                        link.clone(),
                        ManifestEntry {
                            status: "completed".to_string(),
                            games_count: count,
                        },
                    );
                    manifest.save(manifest_path)?;
                    pb.println(format!(
                        "✓ Finished: {} (+{} games)",
                        get_file_prefix(link),
                        count
                    ));
                } else {
                    pb.println(format!(
                        "! Limit reached at: {} (+{} games)",
                        get_file_prefix(link),
                        count
                    ));
                }
            }
            Err(e) => {
                pb.println(format!("✗ Error processing {}: {}", link, e));
            }
        }
    }

    pb.finish_with_message(format!("Done! Total games: {}", total_games));
    Ok(())
}

fn process_link(link: &str, total_games: &mut usize, pb: &ProgressBar) -> Result<(usize, bool)> {
    let response = reqwest::blocking::get(link)?;
    let decoder = zstd::stream::read::Decoder::new(BufReader::with_capacity(512 * 1024, response))?;
    let mut pgn_reader = Reader::new(BufReader::with_capacity(256 * 1024, decoder));
    let mut visitor = FilteredVisitor::new();

    let prefix = get_file_prefix(link);
    let mut batch = Vec::with_capacity(CONFIG.batch_size);
    let mut games_in_link = 0;
    let mut part_idx = 0;
    let mut fully_read = true;

    while let Some(game) = pgn_reader.read_game(&mut visitor)? {
        if let Some(valid_game) = game {
            batch.push(valid_game);
            games_in_link += 1;
            *total_games += 1;
            pb.inc(1);

            if batch.len() >= CONFIG.batch_size {
                let filename =
                    format!("{}/{}_part_{}.parquet", CONFIG.output_dir, prefix, part_idx);
                save_batch(&batch, &filename)?;
                batch.clear();
                part_idx += 1;
            }

            if *total_games >= CONFIG.target_games {
                fully_read = false;
                break;
            }
        }
    }

    if !batch.is_empty() {
        let filename = format!("{}/{}_part_{}.parquet", CONFIG.output_dir, prefix, part_idx);
        save_batch(&batch, &filename)?;
    }

    Ok((games_in_link, fully_read))
}
