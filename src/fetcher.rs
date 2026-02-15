use anyhow::Result;
use pgn_reader::{RawTag, SanPlus, Visitor};
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use shakmaty::{CastlingMode, Chess, Position};
use std::collections::HashMap;
use std::fmt::Write;
use std::fs::File;
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
            .and_then(|f| Ok(serde_json::from_reader(f).unwrap_or_default()))
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
    target_games: 5_000_000,
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
                if let Some(base) = val_str.split('+').next() {
                    if let Ok(secs) = base.parse::<i32>() {
                        if secs >= CONFIG.min_base_time_s {
                            self.valid_time_control = true;
                        } else {
                            self.skip_game = true;
                        }
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
