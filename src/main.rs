use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use pgn_reader::{RawTag, Reader, SanPlus, Visitor};
use polars::prelude::*;
use rayon::prelude::*;
use std::fmt::Write;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::ops::ControlFlow;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

struct Config {
    min_elo: i32,
    min_plys: usize,
    max_elo_diff: i32,
    min_base_time_s: i32,
    include_draws: bool,
    batch_size: usize,
    local_batch_size: usize,
    target_games: usize,
    links_path: &'static str,
    output_dir: &'static str,
}

const CONFIG: Config = Config {
    min_elo: 1800,
    min_plys: 20,
    max_elo_diff: 500,
    min_base_time_s: 300,
    include_draws: true,
    batch_size: 10_000,
    local_batch_size: 100,
    target_games: 50_000,
    links_path: "data/download_links.txt",
    output_dir: "data/parquet",
};

struct SharedState {
    buffer: Mutex<Vec<GameData>>,
    total_collected: AtomicUsize,
    file_counter: AtomicUsize,
}

impl SharedState {
    fn add_games(&self, mut games: Vec<GameData>) -> Result<usize> {
        let count = games.len();
        let mut to_save: Option<Vec<GameData>> = None;
        let mut part_idx: usize = 0;

        {
            let mut buffer = self
                .buffer
                .lock()
                .map_err(|e| anyhow::anyhow!("Mutex poisoned: {}", e))?;

            if self.total_collected.load(Ordering::SeqCst) >= CONFIG.target_games {
                return Ok(0);
            }

            buffer.append(&mut games);
            self.total_collected.fetch_add(count, Ordering::SeqCst);

            if buffer.len() >= CONFIG.batch_size {
                part_idx = self.file_counter.fetch_add(1, Ordering::SeqCst);
                to_save = Some(buffer.drain(0..CONFIG.batch_size).collect());
            }
        }

        if let Some(data) = to_save {
            save_batch(data, part_idx)?;
        }

        Ok(count)
    }

    fn should_stop(&self) -> bool {
        self.total_collected.load(Ordering::SeqCst) >= CONFIG.target_games
    }
}

fn save_batch(games: Vec<GameData>, part_idx: usize) -> Result<()> {
    let height = games.len();
    if height == 0 {
        return Ok(());
    }

    let mut w_elo = Vec::with_capacity(height);
    let mut b_elo = Vec::with_capacity(height);
    let mut res = Vec::with_capacity(height);
    let mut mvs = Vec::with_capacity(height);
    let mut ops = Vec::with_capacity(height);

    for g in games {
        w_elo.push(g.white_elo);
        b_elo.push(g.black_elo);
        res.push(g.result);
        mvs.push(g.moves);
        ops.push(g.opening);
    }

    let mut df = DataFrame::new(
        height,
        vec![
            Series::new("white_elo".into(), w_elo).into(),
            Series::new("black_elo".into(), b_elo).into(),
            Series::new("result".into(), res).into(),
            Series::new("moves".into(), mvs).into(),
            Series::new("opening".into(), ops).into(),
        ],
    )?;

    let filename = format!("{}/dataset_part_{}.parquet", CONFIG.output_dir, part_idx);
    let mut file = std::fs::File::create(&filename)?;
    ParquetWriter::new(&mut file).finish(&mut df)?;
    file.sync_all()?;
    Ok(())
}

#[derive(Default, Clone)]
struct GameData {
    white_elo: i32,
    black_elo: i32,
    result: Option<i8>,
    moves: String,
    opening: String,
}

struct FilteredVisitor {
    current_game: GameData,
    skip_game: bool,
    valid_time_control: bool,
    ply_count: usize,
}

impl FilteredVisitor {
    fn new() -> Self {
        Self {
            current_game: GameData::default(),
            skip_game: false,
            valid_time_control: false,
            ply_count: 0,
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
        if self.skip_game {
            return ControlFlow::Break(None);
        }

        if !self.valid_time_control {
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
        if !self.current_game.moves.is_empty() {
            self.current_game.moves.push(' ');
        }
        let _ = write!(self.current_game.moves, "{}", san_plus);
        self.ply_count += 1;
        ControlFlow::Continue(())
    }

    fn end_game(&mut self, _movetext: Self::Movetext) -> Self::Output {
        if self.ply_count < CONFIG.min_plys {
            return None;
        }

        let game = std::mem::take(&mut self.current_game);
        Some(game)
    }
}

fn main() -> Result<()> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build_global()?;

    let path = Path::new(CONFIG.links_path);
    if !path.exists() {
        return Ok(());
    }
    let file = File::open(&path)?;
    let reader = BufReader::new(file);
    let links: Vec<String> = reader
        .lines()
        .filter_map(|l| l.ok())
        .filter(|l| !l.trim().is_empty())
        .collect();

    let state = Arc::new(SharedState {
        buffer: Mutex::new(Vec::with_capacity(CONFIG.batch_size)),
        total_collected: AtomicUsize::new(0),
        file_counter: AtomicUsize::new(0),
    });

    let pb = ProgressBar::new(CONFIG.target_games as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")?
        .progress_chars("#>-"));

    links.par_iter().for_each(|link| {
        if state.should_stop() {
            return;
        }

        if let Err(e) = process_link(link, &state, &pb) {
            pb.println(format!("Error processing {}: {}", link, e));
        }
    });

    // Flush remaining games
    let mut buffer = state.buffer.lock().unwrap();
    if !buffer.is_empty() {
        let part_idx = state.file_counter.fetch_add(1, Ordering::SeqCst);
        let to_save: Vec<GameData> = buffer.drain(..).collect();
        let _ = save_batch(to_save, part_idx);
    }

    pb.finish_with_message(format!(
        "Done! Saved {} games.",
        state.total_collected.load(Ordering::SeqCst)
    ));
    Ok(())
}

fn process_link(link: &str, state: &SharedState, pb: &ProgressBar) -> Result<()> {
    let response = reqwest::blocking::get(link)?;
    // Wrap the network stream in a large buffer to smooth out chunks
    let buf_response = BufReader::with_capacity(128 * 1024, response);
    let decoder = zstd::stream::read::Decoder::new(buf_response)?;
    let reader = BufReader::with_capacity(64 * 1024, decoder);

    let mut pgn_reader = Reader::new(reader);
    let mut visitor = FilteredVisitor::new();

    let mut games_batch = Vec::with_capacity(CONFIG.local_batch_size);

    while let Some(game) = pgn_reader.read_game(&mut visitor)? {
        if let Some(valid_game) = game {
            games_batch.push(valid_game);

            if games_batch.len() >= CONFIG.local_batch_size {
                let saved = state.add_games(games_batch)?;
                if saved > 0 {
                    pb.inc(saved as u64);
                }
                games_batch = Vec::with_capacity(CONFIG.local_batch_size);
            }
        }
        if state.should_stop() {
            break;
        }
    }

    if !games_batch.is_empty() {
        let saved = state.add_games(games_batch)?;
        if saved > 0 {
            pb.inc(saved as u64);
        }
    }

    Ok(())
}
